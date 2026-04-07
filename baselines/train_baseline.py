"""
Baseline training script for coupled PDE benchmarks.

Usage (from this directory):
    python train_baseline.py --model fno2d \
        --data_path ../data_generator/gray-scott/tiny_set/gs.pt \
        --output_dir ./output_fno2d \
        --batch_size 32 --epochs 400

All baselines receive the concatenated input  [B, n_proc, H, W]
and predict the concatenated target  [B, n_proc, H, W].
The evaluation metric is the relative L2 error (per process and averaged).
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

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "meanflow"))
from data_loaders.grayscott_loader import build_grayscott_dataloader

from models import (
    FNO1d, FNO2d,
    UFNO1d, UFNO2d,
    DeepONet1d, DeepONet2d,
    Transolver1d, Transolver2d,
    CMWNO1d, CMWNO2d,
    COMPOL1d, COMPOL2d,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_model(args) -> nn.Module:
    n = args.n_proc
    c_in = args.in_channels_per_proc
    c_out = args.out_channels_per_proc
    name = args.model.lower()

    if name == "fno2d":
        return FNO2d(
            modes1=args.modes, modes2=args.modes,
            width=args.width, in_channels=n * c_in,
            out_channels=n * c_out, n_layers=args.n_layers,
            padding=args.padding,
        )
    if name == "fno1d":
        return FNO1d(
            modes=args.modes, width=args.width,
            in_channels=n * c_in, out_channels=n * c_out,
            n_layers=args.n_layers,
        )
    if name == "ufno2d":
        return UFNO2d(
            modes1=args.modes, modes2=args.modes,
            width=args.width, in_channels=n * c_in,
            out_channels=n * c_out, n_layers=args.n_layers,
            padding=args.padding,
        )
    if name == "ufno1d":
        return UFNO1d(
            modes=args.modes, width=args.width,
            in_channels=n * c_in, out_channels=n * c_out,
            n_layers=args.n_layers,
        )
    if name == "deeponet2d":
        return DeepONet2d(
            in_channels=n * c_in, out_channels=n * c_out,
            p=args.deeponet_p, width=args.width,
        )
    if name == "deeponet1d":
        return DeepONet1d(
            in_channels=n * c_in, out_channels=n * c_out,
            L=args.seq_len, p=args.deeponet_p,
        )
    if name == "transolver2d":
        return Transolver2d(
            in_channels=n * c_in, out_channels=n * c_out,
            dim=args.width, n_slices=args.n_slices,
            n_heads=args.n_heads, n_layers=args.n_layers,
        )
    if name == "transolver1d":
        return Transolver1d(
            in_channels=n * c_in, out_channels=n * c_out,
            dim=args.width, n_slices=args.n_slices,
            n_heads=args.n_heads, n_layers=args.n_layers,
        )
    if name == "cmwno2d":
        return CMWNO2d(
            in_channels=c_in, out_channels=c_out, n_proc=n,
            width=args.width, n_layers=args.n_layers, k=args.wavelet_k,
        )
    if name == "cmwno1d":
        return CMWNO1d(
            in_channels=c_in, out_channels=c_out, n_proc=n,
            width=args.width, n_layers=args.n_layers, k=args.wavelet_k,
        )
    if name == "compol2d":
        return COMPOL2d(
            in_channels=c_in, out_channels=c_out, n_proc=n,
            modes1=args.modes, modes2=args.modes,
            width=args.width, n_layers=args.n_layers,
            n_heads=args.n_heads, padding=args.padding,
        )
    if name == "compol1d":
        return COMPOL1d(
            in_channels=c_in, out_channels=c_out, n_proc=n,
            modes=args.modes, width=args.width,
            n_layers=args.n_layers, n_heads=args.n_heads,
        )
    raise ValueError(f"Unknown model: {args.model}")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_dataloaders(args):
    """Return (train_loader, val_loader, test_loader) for the selected dataset."""
    if args.dataset == "grayscott":
        kw = dict(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            horizon=1,
            normalize=False,
        )
        _, train_loader = build_grayscott_dataloader(split="train", shuffle=True, **kw)
        _, val_loader = build_grayscott_dataloader(split="val", shuffle=False,
                                                    drop_last=False, **kw)
        _, test_loader = build_grayscott_dataloader(split="test", shuffle=False,
                                                     drop_last=False, **kw)
        return train_loader, val_loader, test_loader
    raise NotImplementedError(f"Dataset {args.dataset!r} not supported yet. "
                              "Add a loader in build_dataloaders().")


def batch_to_xy(batch):
    """
    Convert a batch tuple (z1_1, z1_2, z0_1, z0_2) to
    concatenated input x [B, n_proc, *spatial] and
    concatenated target y [B, n_proc, *spatial].
    """
    z1_1, z1_2, z0_1, z0_2 = batch
    x = torch.cat([z1_1, z1_2], dim=1)
    y = torch.cat([z0_1, z0_2], dim=1)
    return x, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> dict:
    model.eval()
    n_proc = 2  # TODO: generalise
    num = [0.0] * n_proc
    den = [0.0] * n_proc

    for batch in loader:
        x, y = batch_to_xy(batch)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x)
        C_per = pred.shape[1] // n_proc
        for i in range(n_proc):
            p_i = pred[:, i * C_per:(i + 1) * C_per]
            y_i = y[:, i * C_per:(i + 1) * C_per]
            num[i] += ((p_i - y_i) ** 2).sum().item()
            den[i] += (y_i ** 2).sum().item()

    model.train()
    rel_l2s = [(n / max(d, 1e-8)) ** 0.5 for n, d in zip(num, den)]
    return {
        "rel_l2": sum(rel_l2s) / len(rel_l2s),
        **{f"rel_l2_{i+1}": v for i, v in enumerate(rel_l2s)},
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, args, log_writer):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        if step > 0 and args.test_run:
            break
        x, y = batch_to_xy(batch)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)

        if not math.isfinite(loss.item()):
            raise ValueError(f"Loss is {loss.item()}, stopping.")

        loss.backward()

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
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


def save_checkpoint(model, optimizer, scheduler, epoch, args):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }
    path = os.path.join(args.output_dir, f"checkpoint-{epoch:04d}.pth")
    torch.save(ckpt, path)
    # Keep a "last" symlink for easy resuming
    last_path = os.path.join(args.output_dir, "checkpoint-last.pth")
    torch.save(ckpt, last_path)


def load_checkpoint(model, optimizer, scheduler, args):
    resume = getattr(args, "resume", None)
    if not resume:
        last = os.path.join(args.output_dir, "checkpoint-last.pth")
        if os.path.isfile(last) and getattr(args, "auto_resume", False):
            resume = last
    if resume and os.path.isfile(resume):
        logger.info(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt["epoch"] + 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("Baseline training for coupled PDEs")

    # ── dataset ──────────────────────────────────────────────────────────────
    p.add_argument("--dataset", default="grayscott", choices=["grayscott"])
    p.add_argument("--data_path", required=True)
    p.add_argument("--n_proc", type=int, default=2,
                   help="Number of coupled processes (channels in the dataset).")
    p.add_argument("--in_channels_per_proc", type=int, default=1)
    p.add_argument("--out_channels_per_proc", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=256,
                   help="Sequence length for 1-D datasets (used by DeepONet1d).")

    # ── model ────────────────────────────────────────────────────────────────
    p.add_argument("--model", required=True,
                   choices=[
                       "fno2d", "fno1d",
                       "ufno2d", "ufno1d",
                       "deeponet2d", "deeponet1d",
                       "transolver2d", "transolver1d",
                       "cmwno2d", "cmwno1d",
                       "compol2d", "compol1d",
                   ])
    # Shared hyper-parameters
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--modes", type=int, default=12,
                   help="Number of Fourier/wavelet modes (FNO/UFNO/COMPOL).")
    p.add_argument("--padding", type=int, default=9,
                   help="Zero-padding for FNO/UFNO/COMPOL spectral layers.")
    # Transolver-specific
    p.add_argument("--n_slices", type=int, default=32,
                   help="Number of physics tokens (Transolver).")
    p.add_argument("--n_heads", type=int, default=4,
                   help="Attention heads (Transolver, COMPOL, CMWNO coupling).")
    # DeepONet-specific
    p.add_argument("--deeponet_p", type=int, default=128,
                   help="Basis dimension p for DeepONet.")
    # CMWNO-specific
    p.add_argument("--wavelet_k", type=int, default=2,
                   help="Wavelet filter size k (must be power of 2) for CMWNO.")

    # ── training ─────────────────────────────────────────────────────────────
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--warmup_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Max gradient norm (0 = disabled).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")

    # ── logging / checkpointing ───────────────────────────────────────────────
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--eval_frequency", type=int, default=20,
                   help="Evaluate every N epochs.")
    p.add_argument("--log_per_step", type=int, default=10)
    p.add_argument("--resume", default="", help="Path to checkpoint to resume from.")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--test_run", action="store_true",
                   help="Run one batch only (for debugging).")

    return p.parse_args()


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

    # ── data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(args)
    logger.info(
        f"Dataset: {args.dataset}  "
        f"train={len(train_loader.dataset)}  "
        f"val={len(val_loader.dataset)}  "
        f"test={len(test_loader.dataset)}"
    )

    # ── model ────────────────────────────────────────────────────────────────
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model}  parameters: {n_params:,}")

    # ── optimiser + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_iters = args.epochs * len(train_loader)
    warmup_iters = args.warmup_epochs * len(train_loader)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6 / args.lr, end_factor=1.0,
        total_iters=warmup_iters,
    )
    main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters - warmup_iters, eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_iters]
    )

    # ── resume ───────────────────────────────────────────────────────────────
    start_epoch = load_checkpoint(model, optimizer, scheduler, args)

    # ── tensorboard ──────────────────────────────────────────────────────────
    log_writer = SummaryWriter(log_dir=args.output_dir)

    # ── eval only ────────────────────────────────────────────────────────────
    if args.eval_only:
        metrics = evaluate(model, test_loader, device)
        logger.info(f"[eval only] test: {metrics}")
        return

    # ── training loop ────────────────────────────────────────────────────────
    best_val_rel_l2 = float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, args, log_writer,
        )

        if (epoch + 1) % args.eval_frequency == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate(model, val_loader, device)
            logger.info(
                f"Epoch {epoch + 1}  val: "
                + "  ".join(f"{k}={v:.6f}" for k, v in val_metrics.items())
            )
            for k, v in val_metrics.items():
                log_writer.add_scalar(f"val/{k}", v, epoch + 1)

            # Save checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch, args)

            # Save best model
            if val_metrics["rel_l2"] < best_val_rel_l2:
                best_val_rel_l2 = val_metrics["rel_l2"]
                best_path = os.path.join(args.output_dir, "checkpoint-best.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_rel_l2": best_val_rel_l2}, best_path)
                logger.info(f"  -> new best val rel-L2: {best_val_rel_l2:.6f}")

        if args.test_run:
            break

    # ── final test evaluation ─────────────────────────────────────────────────
    best_ckpt = os.path.join(args.output_dir, "checkpoint-best.pth")
    if os.path.isfile(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        logger.info(f"Loaded best checkpoint (epoch {state['epoch']}).")

    test_metrics = evaluate(model, test_loader, device)
    logger.info(
        "Final test: " + "  ".join(f"{k}={v:.6f}" for k, v in test_metrics.items())
    )
    for k, v in test_metrics.items():
        log_writer.add_scalar(f"test/{k}", v, args.epochs)

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Total training time: {elapsed}")
    log_writer.close()


if __name__ == "__main__":
    main()
