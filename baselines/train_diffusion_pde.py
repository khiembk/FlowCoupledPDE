"""
EDM-based conditional diffusion baseline for coupled PDE prediction.

Trains DiffusionPDE (SongUNet + EDM preconditioning) on GS / MPF / LV datasets.
Conditioning: source channels are concatenated with the noisy target.

LV (1D, len=256): reshaped to 16×16 internally so the 2D SongUNet can be reused.
Evaluation metrics are computed in the original 1D space.

Usage (from baselines/):
    python train_diffusion_pde.py \
        --data_path /scratch/user/u.kt348068/PDE_data/gs_512.pt \
        --output_dir /scratch/user/u.kt348068/ckpt/gs512_diffusion_pde \
        --dataset grayscott \
        --train_ratio 0.6305 --val_ratio 0.1232 \
        --epochs 500 --lr 2e-4 --batch_size 32
"""
import argparse
import datetime
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# ── project imports ───────────────────────────────────────────────────────────
_BASELINES = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASELINES.parent / "meanflow"))  # data_loaders.*
sys.path.insert(0, str(_BASELINES))                       # models.DiffusionPDE
from data_loaders.grayscott_loader import build_grayscott_dataloader
from data_loaders.lv_loader import build_lv_dataloader
from data_loaders.bz_loader import build_bz_dataloader
from data_loaders.thm_loader import build_thm_dataloader
from data_loaders.gs_well_loader import build_gs_well_dataloader
from data_loaders.dr2d_loader import build_dr2d_dataloader

from models import DiffusionPDE

logger = logging.getLogger(__name__)

_1D_SEQ_LEN = 256   # 16×16 = 256
_1D_RES     = 16


# ── data helpers ──────────────────────────────────────────────────────────────

def build_dataloaders(args):
    base_kw = dict(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        horizon=1,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    # grayscott/lv loaders accept normalize; bz/thm do not
    norm_kw = {**base_kw, "normalize": False}
    has_test = (args.train_ratio + args.val_ratio) < 1.0

    if args.dataset in ("grayscott", "multiphase"):
        kw = norm_kw
        _, train_loader = build_grayscott_dataloader(split="train", shuffle=True, **kw)
        _, val_loader   = build_grayscott_dataloader(split="val",   shuffle=False, drop_last=False, **kw)
        if has_test:
            _, test_loader = build_grayscott_dataloader(split="test", shuffle=False, drop_last=False, **kw)
        else:
            test_loader = val_loader
    elif args.dataset == "lv":
        kw = norm_kw
        _, train_loader = build_lv_dataloader(split="train", shuffle=True, **kw)
        _, val_loader   = build_lv_dataloader(split="val",   shuffle=False, drop_last=False, **kw)
        if has_test:
            _, test_loader = build_lv_dataloader(split="test", shuffle=False, drop_last=False, **kw)
        else:
            test_loader = val_loader
    elif args.dataset == "bz":
        kw = base_kw
        _, train_loader = build_bz_dataloader(split="train", shuffle=True, **kw)
        _, val_loader   = build_bz_dataloader(split="val",   shuffle=False, drop_last=False, **kw)
        if has_test:
            _, test_loader = build_bz_dataloader(split="test", shuffle=False, drop_last=False, **kw)
        else:
            test_loader = val_loader
    elif args.dataset == "thm":
        kw = base_kw
        _, train_loader = build_thm_dataloader(split="train", shuffle=True, **kw)
        _, val_loader   = build_thm_dataloader(split="val",   shuffle=False, drop_last=False, **kw)
        if has_test:
            _, test_loader = build_thm_dataloader(split="test", shuffle=False, drop_last=False, **kw)
        else:
            test_loader = val_loader
    elif args.dataset == "gs_well":
        kw = dict(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            horizon=1,
            time_stride=getattr(args, "time_stride", 10),
            resolution=64,
        )
        _, train_loader = build_gs_well_dataloader(split="train", shuffle=True,  drop_last=True,  **kw)
        _, val_loader   = build_gs_well_dataloader(split="valid", shuffle=False, drop_last=False, **kw)
        _, test_loader  = build_gs_well_dataloader(split="test",  shuffle=False, drop_last=False, **kw)
    elif args.dataset == "dr2d":
        kw = dict(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            horizon=1,
            time_stride=getattr(args, "time_stride", 1),
            resolution=64,
        )
        _, train_loader = build_dr2d_dataloader(split="train", shuffle=True,  drop_last=True,  **kw)
        _, val_loader   = build_dr2d_dataloader(split="val",   shuffle=False, drop_last=False, **kw)
        _, test_loader  = build_dr2d_dataloader(split="test",  shuffle=False, drop_last=False, **kw)
    else:
        raise NotImplementedError(args.dataset)

    return train_loader, val_loader, test_loader


def batch_to_xy(batch, is_1d: bool = False):
    """(z1_1, z1_2, z0_1, z0_2) → source, target  [B, n_proc, H, W]
    For LV (is_1d=True): input [B,2,256] is reshaped to [B,2,16,16].
    """
    n_proc = len(batch) // 2
    src = torch.cat(list(batch[:n_proc]),  dim=1)
    tgt = torch.cat(list(batch[n_proc:]),  dim=1)
    if is_1d:
        B, C, L = src.shape
        src = src.reshape(B, C, _1D_RES, _1D_RES)
        tgt = tgt.reshape(B, C, _1D_RES, _1D_RES)
    return src, tgt


def compute_norm_stats(loader, n_proc: int, device, is_1d: bool = False):
    """Compute per-channel mean and std over the training source (x) tensors."""
    sum_x  = torch.zeros(n_proc, device=device)
    sum_x2 = torch.zeros(n_proc, device=device)
    count  = 0
    for batch in loader:
        src, _ = batch_to_xy(batch, is_1d)
        src = src.to(device)
        B = src.shape[0]
        flat = src.view(B, n_proc, -1)
        sum_x  += flat.sum(dim=(0, 2))
        sum_x2 += (flat ** 2).sum(dim=(0, 2))
        count  += B * flat.shape[2]
    mean = sum_x / count
    std  = ((sum_x2 / count - mean ** 2).clamp_min(1e-8)).sqrt()
    return mean, std


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: DiffusionPDE, loader, device, num_steps: int,
             is_1d: bool = False) -> dict:
    model.eval()
    n_proc = model.n_proc
    nums = [0.0] * n_proc
    dens = [0.0] * n_proc

    for batch in loader:
        src, tgt = batch_to_xy(batch, is_1d)
        src, tgt = src.to(device), tgt.to(device)
        pred = model(src, num_steps=num_steps)   # [B, n_proc, H, W] or 16×16
        if is_1d:
            B, C = pred.shape[:2]
            pred = pred.reshape(B, C, _1D_SEQ_LEN)
            tgt  = tgt.reshape(B, C, _1D_SEQ_LEN)
        for i in range(n_proc):
            nums[i] += ((pred[:, i] - tgt[:, i]) ** 2).sum().item()
            dens[i] += (tgt[:, i] ** 2).sum().item()

    model.train()
    rel_l2s = [(n / max(d, 1e-8)) ** 0.5 for n, d in zip(nums, dens)]
    return {
        "rel_l2": sum(rel_l2s) / len(rel_l2s),
        **{f"rel_l2_{i+1}": v for i, v in enumerate(rel_l2s)},
    }


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, args):
    ckpt = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch":     epoch,
        "args":      vars(args),
    }
    torch.save(ckpt, os.path.join(args.output_dir, "checkpoint-last.pth"))


def load_checkpoint(model, optimizer, scheduler, args):
    last = os.path.join(args.output_dir, "checkpoint-last.pth")
    if os.path.isfile(last) and getattr(args, "auto_resume", False):
        logger.info(f"Resuming from {last}")
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt["epoch"] + 1
    return 0


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, args,
                    log_writer, is_1d: bool = False):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for step, batch in enumerate(loader):
        if step > 0 and args.test_run:
            break
        src, tgt = batch_to_xy(batch, is_1d)
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(src, tgt)

        if not math.isfinite(loss.item()):
            raise ValueError(f"Loss is {loss.item()} at epoch {epoch} step {step}, stopping.")

        loss.backward()

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

        global_step = epoch * len(loader) + step
        if log_writer is not None and (global_step + 1) % args.log_per_step == 0:
            avg = total_loss / n_batches
            lr  = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch} [{step}/{len(loader)}]: loss={avg:.6f}  lr={lr:.6e}")
            log_writer.add_scalar("train/loss", avg, global_step)
            log_writer.add_scalar("train/lr",   lr,  global_step)
            total_loss = 0.0
            n_batches  = 0


# ── arg parser ────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("DiffusionPDE training for coupled PDEs")

    # dataset
    p.add_argument("--dataset",     default="grayscott", choices=["grayscott", "multiphase", "lv", "bz", "thm", "gs_well", "dr2d"])
    p.add_argument("--data_path",   required=True)
    p.add_argument("--train_ratio", type=float, default=0.6305)
    p.add_argument("--val_ratio",   type=float, default=0.1232)
    p.add_argument("--n_proc",      type=int,   default=2)

    # model
    p.add_argument("--img_resolution", type=int,   default=64)
    p.add_argument("--model_channels", type=int,   default=64)
    p.add_argument("--num_blocks",     type=int,   default=4)
    p.add_argument("--dropout",        type=float, default=0.10)
    # EDM
    p.add_argument("--sigma_data", type=float, default=0.5)
    p.add_argument("--sigma_min",  type=float, default=0.002)
    p.add_argument("--sigma_max",  type=float, default=80.0)
    p.add_argument("--P_mean",     type=float, default=-1.2)
    p.add_argument("--P_std",      type=float, default=1.2)
    # inference
    p.add_argument("--num_steps",      type=int, default=20, help="ODE steps at test time")
    p.add_argument("--eval_num_steps", type=int, default=10,
                   help="Faster ODE steps during training validation (set lower to speed up eval)")

    # training
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--epochs",         type=int,   default=500)
    p.add_argument("--warmup_epochs",  type=int,   default=50)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--weight_decay",   type=float, default=0.0)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="cuda")

    # logging / checkpointing
    p.add_argument("--output_dir",      default="./output")
    p.add_argument("--eval_frequency",  type=int, default=20)
    p.add_argument("--log_per_step",    type=int, default=10)
    p.add_argument("--auto_resume",     action="store_true")
    p.add_argument("--eval_only",       action="store_true")
    p.add_argument("--test_run",        action="store_true")

    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    is_1d = args.dataset in ("lv", "bz")

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(args)
    logger.info(
        f"Dataset: {args.dataset}  "
        f"train={len(train_loader.dataset)}  "
        f"val={len(val_loader.dataset)}  "
        f"test={len(test_loader.dataset)}"
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = DiffusionPDE(
        n_proc         = args.n_proc,
        img_resolution = args.img_resolution,
        model_channels = args.model_channels,
        num_blocks     = args.num_blocks,
        dropout        = args.dropout,
        sigma_data     = args.sigma_data,
        sigma_min      = args.sigma_min,
        sigma_max      = args.sigma_max,
        P_mean         = args.P_mean,
        P_std          = args.P_std,
        num_steps      = args.num_steps,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"DiffusionPDE  parameters: {n_params:,}")

    # ── normalisation stats (computed from train set source data) ─────────────
    norm_stats_path = os.path.join(args.output_dir, "norm_stats.pt")
    if os.path.isfile(norm_stats_path):
        stats = torch.load(norm_stats_path, map_location="cpu")
        mean, std = stats["mean"].to(device), stats["std"].to(device)
        logger.info(f"Loaded norm stats from {norm_stats_path}")
    else:
        logger.info("Computing normalisation stats from training set …")
        mean, std = compute_norm_stats(train_loader, args.n_proc, device, is_1d)
        torch.save({"mean": mean.cpu(), "std": std.cpu()}, norm_stats_path)
        logger.info(f"  mean={mean.tolist()}  std={std.tolist()}")

    model.data_mean.copy_(mean)
    model.data_std.copy_(std)

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer    = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_iters  = args.epochs * len(train_loader)
    warmup_iters = args.warmup_epochs * len(train_loader)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6 / args.lr, end_factor=1.0, total_iters=warmup_iters,
    )
    main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters - warmup_iters, eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_iters]
    )

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch = load_checkpoint(model, optimizer, scheduler, args)
    # re-apply norm stats in case they were overwritten by checkpoint load
    model.data_mean.copy_(mean)
    model.data_std.copy_(std)

    log_writer = SummaryWriter(log_dir=args.output_dir)

    if args.eval_only:
        metrics = evaluate(model, test_loader, device, args.num_steps, is_1d)
        logger.info(f"[eval only] test: {metrics}")
        return

    # ── training loop ─────────────────────────────────────────────────────────
    best_val = float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler,
                        device, epoch, args, log_writer, is_1d)

        if (epoch + 1) % args.eval_frequency == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate(model, val_loader, device, args.eval_num_steps, is_1d)
            logger.info(
                f"Epoch {epoch + 1}  val: "
                + "  ".join(f"{k}={v:.6f}" for k, v in val_metrics.items())
            )
            for k, v in val_metrics.items():
                log_writer.add_scalar(f"val/{k}", v, epoch + 1)

            save_checkpoint(model, optimizer, scheduler, epoch, args)

            if val_metrics["rel_l2"] < best_val:
                best_val = val_metrics["rel_l2"]
                best_path = os.path.join(args.output_dir, "checkpoint-best.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_rel_l2": best_val}, best_path)
                logger.info(f"  -> new best val rel-L2: {best_val:.6f}")

        if args.test_run:
            break

    # ── final test evaluation ──────────────────────────────────────────────────
    best_ckpt = os.path.join(args.output_dir, "checkpoint-best.pth")
    if os.path.isfile(best_ckpt):
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.data_mean.copy_(mean)
        model.data_std.copy_(std)
        logger.info(f"Loaded best checkpoint (epoch {state['epoch']}).")

    test_metrics = evaluate(model, test_loader, device, args.num_steps, is_1d)
    logger.info("Final test: " + "  ".join(f"{k}={v:.6f}" for k, v in test_metrics.items()))
    for k, v in test_metrics.items():
        log_writer.add_scalar(f"test/{k}", v, args.epochs)

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Total training time: {elapsed}")
    log_writer.close()


if __name__ == "__main__":
    main()
