"""
Naive (independent) Mean-Flow training script for coupled PDE benchmarks.

Each process is modelled by its own lightweight UNet (~5–7 M params) that
receives only its own field as input — no cross-process coupling.
Training uses the same MeanFlow local (JVP) + global objectives as the
coupled CoupledFlow baseline.

Usage (run from baselines/):

  # Gray-Scott 512 (2-D, 2 processes):
  python train_naive_flow.py \\
      --dataset grayscott --data_path <path>/gs_512.pt \\
      --output_dir ./output/naive_gs512 \\
      --batch_size 32 --lr 6e-4 --epochs 500 --warmup_epochs 100 \\
      --train_ratio 0.6305 --val_ratio 0.1232 \\
      --dropout 0.2 --ema_decay 0.9999 --ema_decays 0.99995 0.9996 \\
      --seq_loss --auto_resume

  # Lotka-Volterra 512 (1-D, 2 processes):
  python train_naive_flow.py \\
      --dataset lv --data_path <path>/lv_512.pt \\
      --output_dir ./output/naive_lv512 \\
      --batch_size 32 --lr 6e-4 --epochs 500 --warmup_epochs 100 \\
      --train_ratio 0.6305 --val_ratio 0.1232 \\
      --dropout 0.2 --ema_decay 0.9999 --ema_decays 0.99995 0.9996 \\
      --seq_loss --auto_resume

  # BZ 512 (1-D, 3 processes):
  python train_naive_flow.py \\
      --dataset bz --data_path <path>/bz_512.pt \\
      --n_proc 3 \\
      --output_dir ./output/naive_bz512 \\
      --batch_size 32 --lr 6e-4 --epochs 500 --warmup_epochs 100 \\
      --train_ratio 0.6305 --val_ratio 0.1232 \\
      --dropout 0.1 --ema_decay 0.9999 --ema_decays 0.99995 0.9996 \\
      --seq_loss --auto_resume

  # THM 512 (2-D, 5 processes):
  python train_naive_flow.py \\
      --dataset thm --data_path <path>/thm_512.pt \\
      --n_proc 5 \\
      --output_dir ./output/naive_thm512 \\
      --batch_size 16 --lr 6e-4 --epochs 500 --warmup_epochs 100 \\
      --train_ratio 0.6305 --val_ratio 0.1232 \\
      --dropout 0.1 --ema_decay 0.9999 --ema_decays 0.99995 0.9996 \\
      --seq_loss --auto_resume
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

# ── project path setup ────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MEANFLOW_DIR = str(_REPO_ROOT / "meanflow")
if _MEANFLOW_DIR not in sys.path:
    sys.path.insert(0, _MEANFLOW_DIR)

from data_loaders.grayscott_loader import build_grayscott_dataloader
from data_loaders.lv_loader import build_lv_dataloader
from data_loaders.bz_loader import build_bz_dataloader
from data_loaders.thm_loader import build_thm_dataloader
from data_loaders.dr2d_loader import build_dr2d_dataloader

# Import naive_flow directly from baselines/models/ without triggering
# baselines/models/__init__.py (which imports compol.py that uses Py3.10+ syntax).
_MODELS_DIR = str(Path(__file__).resolve().parent / "models")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)
from naive_flow import build_naive_flow_model

logger = logging.getLogger(__name__)


# ── dataset dimension / process-count registry ────────────────────────────────

DATASET_DIM = {
    "grayscott":  "2d",
    "multiphase": "2d",
    "thm":        "2d",
    "dr2d":       "2d",
    "lv":         "1d",
    "bz":         "1d",
}

DATASET_N_PROC = {
    "grayscott":  2,
    "lv":         2,
    "multiphase": 2,
    "bz":         3,
    "thm":        5,
    "dr2d":       2,
}


# ── data helpers ──────────────────────────────────────────────────────────────

def build_dataloaders(args):
    """Return (train_loader, val_loader, test_loader)."""
    kw_base = dict(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        horizon=1,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    has_test = (args.train_ratio + args.val_ratio) < 1.0

    if args.dataset in ("grayscott", "multiphase"):
        kw = dict(**kw_base, normalize=False)
        _, train_loader = build_grayscott_dataloader(split="train", shuffle=True, **kw)
        _, val_loader   = build_grayscott_dataloader(split="val",   shuffle=False, drop_last=False, **kw)
        _, test_loader  = build_grayscott_dataloader(split="test",  shuffle=False, drop_last=False, **kw) \
                          if has_test else (None, val_loader)
        return train_loader, val_loader, test_loader

    if args.dataset == "lv":
        kw = dict(**kw_base, normalize=False)
        _, train_loader = build_lv_dataloader(split="train", shuffle=True, **kw)
        _, val_loader   = build_lv_dataloader(split="val",   shuffle=False, drop_last=False, **kw)
        _, test_loader  = build_lv_dataloader(split="test",  shuffle=False, drop_last=False, **kw) \
                          if has_test else (None, val_loader)
        return train_loader, val_loader, test_loader

    if args.dataset == "bz":
        _, train_loader = build_bz_dataloader(split="train", shuffle=True, **kw_base)
        _, val_loader   = build_bz_dataloader(split="val",   shuffle=False, drop_last=False, **kw_base)
        _, test_loader  = build_bz_dataloader(split="test",  shuffle=False, drop_last=False, **kw_base) \
                          if has_test else (None, val_loader)
        return train_loader, val_loader, test_loader

    if args.dataset == "thm":
        _, train_loader = build_thm_dataloader(split="train", shuffle=True, **kw_base)
        _, val_loader   = build_thm_dataloader(split="val",   shuffle=False, drop_last=False, **kw_base)
        _, test_loader  = build_thm_dataloader(split="test",  shuffle=False, drop_last=False, **kw_base) \
                          if has_test else (None, val_loader)
        return train_loader, val_loader, test_loader

    if args.dataset == "dr2d":
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
        return train_loader, val_loader, test_loader

    raise NotImplementedError(f"Dataset {args.dataset!r} not supported.")


def split_batch(batch, n_proc):
    """Split (source_1, …, source_N, target_1, …, target_N) into two lists."""
    sources = list(batch[:n_proc])
    targets = list(batch[n_proc:])
    return sources, targets


# ── per-process normaliser ────────────────────────────────────────────────────

class Normalizer:
    """
    Per-process z-score normalizer computed from one pass over the training loader.
    Fixes training instability when data has large values / step differences (e.g. LV, BZ).
    """
    def __init__(self, loader, n_proc, device):
        sums      = [0.0] * n_proc
        sq_sums   = [0.0] * n_proc
        counts    = [0]   * n_proc
        for batch in loader:
            for i in range(n_proc):
                x = batch[i].float()
                sums[i]    += x.sum().item()
                sq_sums[i] += (x ** 2).sum().item()
                counts[i]  += x.numel()
        means = [s / c for s, c in zip(sums, counts)]
        stds  = [max((sq / c - m ** 2) ** 0.5, 1e-8)
                 for sq, c, m in zip(sq_sums, counts, means)]
        self.means = torch.tensor(means, dtype=torch.float32, device=device)
        self.stds  = torch.tensor(stds,  dtype=torch.float32, device=device)
        logger.info(
            "Normalizer stats: "
            + "  ".join(f"proc{i+1} mean={means[i]:.4f} std={stds[i]:.4f}"
                        for i in range(n_proc))
        )

    def normalize(self, tensors):
        return [(t - self.means[i]) / self.stds[i] for i, t in enumerate(tensors)]

    def denormalize(self, tensors):
        return [t * self.stds[i] + self.means[i] for i, t in enumerate(tensors)]


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, nets=None, args=None, normalizer=None):
    """
    Compute per-process and average relative L2 error in the original data space.

    If normalizer is provided, inputs are normalized before the model and
    predictions are denormalized before computing rel-L2.
    """
    model.eval()
    n_proc = model.n_proc
    nums = [0.0] * n_proc
    dens = [0.0] * n_proc

    for batch in loader:
        batch = [x.to(device, non_blocking=True) for x in batch]
        sources, targets = split_batch(batch, n_proc)

        model_sources = normalizer.normalize(sources) if normalizer else sources
        preds = model.sample(model_sources, nets=nets)
        if normalizer:
            preds = normalizer.denormalize(preds)

        for i in range(n_proc):
            nums[i] += ((preds[i] - targets[i]) ** 2).sum().item()
            dens[i] += (targets[i] ** 2).sum().item()
        if args is not None and getattr(args, "test_run", False):
            break

    model.train()
    rel_l2s = [(n / max(d, 1e-8)) ** 0.5 for n, d in zip(nums, dens)]
    return {
        "rel_l2": sum(rel_l2s) / len(rel_l2s),
        **{f"rel_l2_{i + 1}": v for i, v in enumerate(rel_l2s)},
    }


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, args,
                    log_writer, trainable_params, normalizer=None):
    model.train()
    n_proc = model.n_proc
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        if step > 0 and args.test_run:
            break

        batch = [x.to(device, non_blocking=True) for x in batch]
        sources, targets = split_batch(batch, n_proc)

        if normalizer is not None:
            sources = normalizer.normalize(sources)
            targets = normalizer.normalize(targets)

        optimizer.zero_grad(set_to_none=True)

        loss = model.forward_loss(sources, targets)
        loss.backward()
        loss_val = loss.detach().item()

        if not math.isfinite(loss_val):
            raise ValueError(f"Loss is {loss_val}, stopping training.")

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

        optimizer.step()
        model.update_ema()
        scheduler.step()

        total_loss += loss_val
        n_batches += 1

        global_step = epoch * len(loader) + step
        if log_writer is not None and (global_step + 1) % args.log_per_step == 0:
            avg = total_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} [{step}/{len(loader)}]: "
                f"loss={avg:.6f}  lr={lr:.6e}"
            )
            log_writer.add_scalar("train/loss", avg, global_step)
            log_writer.add_scalar("train/lr", lr, global_step)
            total_loss = 0.0
            n_batches = 0


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, args):
    ckpt = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch":     epoch,
        "args":      vars(args),
    }
    last_path = os.path.join(args.output_dir, "checkpoint-last.pth")
    torch.save(ckpt, last_path)
    if (epoch + 1) % 1000 == 0:
        torch.save(ckpt, os.path.join(args.output_dir, f"checkpoint-{epoch}.pth"))


def load_checkpoint(model, optimizer, scheduler, args):
    resume = getattr(args, "resume", None)
    if not resume:
        last = os.path.join(args.output_dir, "checkpoint-last.pth")
        if os.path.isfile(last) and getattr(args, "auto_resume", False):
            resume = last
    if resume and os.path.isfile(resume):
        logger.info(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        if getattr(args, "reset_optimizer", False):
            logger.info("reset_optimizer=True: skipping optimizer/scheduler restore.")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt["epoch"] + 1
    return 0


# ── argument parser ────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("Naive (independent) Mean-Flow for coupled PDEs")

    # dataset
    p.add_argument("--dataset", required=True,
                   choices=["grayscott", "lv", "multiphase", "bz", "thm", "dr2d"])
    p.add_argument("--data_path", required=True)
    p.add_argument("--n_proc", type=int, default=None,
                   help="Number of processes. Defaults to the dataset's standard value "
                        "(2 for GS/LV/MPF, 3 for BZ, 5 for THM).")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio",   type=float, default=0.1)

    # model
    p.add_argument("--dropout",   type=float, default=0.2)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_decays", nargs="+", type=float, default=[0.99995, 0.9996],
                   help="Additional EMA decay rates evaluated at checkpoints.")

    # MeanFlow time sampler
    p.add_argument("--tr_sampler", default="v1", choices=["v0", "v1"])
    p.add_argument("--ratio",    type=float, default=0.75,
                   help="Probability of sampling r ≠ t.")
    p.add_argument("--P_mean_t", type=float, default=-0.6)
    p.add_argument("--P_std_t",  type=float, default=1.6)
    p.add_argument("--P_mean_r", type=float, default=-4.0)
    p.add_argument("--P_std_r",  type=float, default=1.6)
    p.add_argument("--norm_p",   type=float, default=0.75,
                   help="Power for adaptive loss weighting.")
    p.add_argument("--norm_eps", type=float, default=1e-3)

    # training
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--epochs",         type=int,   default=500)
    p.add_argument("--warmup_epochs",  type=int,   default=100)
    p.add_argument("--lr",             type=float, default=6e-4)
    p.add_argument("--weight_decay",   type=float, default=0.0)
    p.add_argument("--grad_clip",      type=float, default=1.0,
                   help="Max gradient norm (0 = disabled).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cuda")

    # logging / checkpointing
    p.add_argument("--output_dir",     default="./output")
    p.add_argument("--eval_frequency", type=int, default=20,
                   help="Evaluate every N epochs.")
    p.add_argument("--log_per_step",   type=int, default=10)
    p.add_argument("--resume",         default="",
                   help="Path to checkpoint to resume from.")
    p.add_argument("--auto_resume",    action="store_true")
    p.add_argument("--reset_optimizer", action="store_true",
                   help="When resuming, reset optimizer/scheduler to use current --lr.")
    p.add_argument("--eval_only",  action="store_true")
    p.add_argument("--test_run",   action="store_true",
                   help="Run one batch only (debugging).")
    p.add_argument("--normalize",  action="store_true",
                   help="Apply per-process z-score normalisation (recommended for LV/BZ).")

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
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Resolve n_proc and dim
    if args.n_proc is None:
        args.n_proc = DATASET_N_PROC[args.dataset]
    dim = DATASET_DIM[args.dataset]
    logger.info(f"Dataset: {args.dataset}  n_proc={args.n_proc}  dim={dim}")

    # ── data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(args)
    logger.info(
        f"Splits: train={len(train_loader.dataset)}  "
        f"val={len(val_loader.dataset)}  "
        f"test={len(test_loader.dataset)}"
    )

    # ── model ────────────────────────────────────────────────────────────────
    model = build_naive_flow_model(n_proc=args.n_proc, dim=dim, args=args).to(device)

    trainable_params = []
    for i in range(1, args.n_proc + 1):
        trainable_params.extend(model._modules[f"net{i}"].parameters())

    n_params_per_net = sum(p.numel() for p in model._modules["net1"].parameters())
    n_params_total   = sum(p.numel() for p in trainable_params)
    logger.info(
        f"NaiveMeanFlow: {args.n_proc} nets × {n_params_per_net:,} params = "
        f"{n_params_total:,} trainable params"
    )

    # ── optimiser + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    warmup_iters = args.warmup_epochs * len(train_loader)
    total_iters  = args.epochs * len(train_loader)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8 / args.lr,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    main_sched = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=total_iters - warmup_iters,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_iters],
    )

    # ── resume ───────────────────────────────────────────────────────────────
    start_epoch = load_checkpoint(model, optimizer, scheduler, args)

    # ── normaliser ───────────────────────────────────────────────────────────
    normalizer = None
    if getattr(args, "normalize", False):
        logger.info("Building per-process normalizer from training data …")
        normalizer = Normalizer(train_loader, args.n_proc, device)

    # ── tensorboard ──────────────────────────────────────────────────────────
    log_writer = SummaryWriter(log_dir=args.output_dir)

    # ── eval only ────────────────────────────────────────────────────────────
    if args.eval_only:
        _run_eval(model, test_loader, device, args, log_writer, epoch=start_epoch - 1,
                  split="test", normalizer=normalizer)
        return

    # ── training loop ────────────────────────────────────────────────────────
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, args, log_writer, trainable_params,
            normalizer=normalizer,
        )

        if (epoch + 1) % args.eval_frequency == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, scheduler, epoch, args)
            _run_eval(model, test_loader, device, args, log_writer, epoch, split="test",
                      normalizer=normalizer)

        if args.test_run:
            break

    # ── final test ───────────────────────────────────────────────────────────
    last_ckpt = os.path.join(args.output_dir, "checkpoint-last.pth")
    if os.path.isfile(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        logger.info("Loaded last checkpoint for final evaluation.")

    _run_eval(model, test_loader, device, args, log_writer, args.epochs, split="test (final)",
              normalizer=normalizer)

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Total training time: {elapsed}")
    log_writer.close()


def _run_eval(model, loader, device, args, log_writer, epoch, split="test", normalizer=None):
    """Evaluate all EMA variants and log results."""
    n_proc = model.n_proc

    # Build list of (label, nets)
    eval_variants = [
        ("noema", [model._modules[f"net{i}"] for i in range(1, n_proc + 1)]),
        ("ema",   [model._modules[f"net{i}_ema"] for i in range(1, n_proc + 1)]),
    ]
    for j, decay in enumerate(model.ema_decays, start=1):
        eval_variants.append((
            f"ema{decay}",
            [model._modules[f"net{i}_ema_extra{j}"] for i in range(1, n_proc + 1)],
        ))

    for suffix, nets in eval_variants:
        metrics = evaluate(model, loader, device, nets=nets, args=args, normalizer=normalizer)
        per_proc = "  ".join(
            f"p{i + 1}={metrics[f'rel_l2_{i + 1}']:.6f}" for i in range(n_proc)
        )
        logger.info(
            f"Eval epoch {epoch + 1} [{suffix}] ({split}): "
            f"rel-L2={metrics['rel_l2']:.6f}  {per_proc}"
        )
        if log_writer is not None:
            log_writer.add_scalar(f"rel_L2/{suffix}", metrics["rel_l2"], epoch + 1)
            for i in range(n_proc):
                log_writer.add_scalar(
                    f"rel_L2_proc{i + 1}/{suffix}", metrics[f"rel_l2_{i + 1}"], epoch + 1
                )


if __name__ == "__main__":
    main()
