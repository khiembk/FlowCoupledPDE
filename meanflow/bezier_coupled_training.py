"""
Training script for CoupledFlowBezier (quadratic Bezier trajectory coupling).

Usage (GrayScott 2D):
    python bezier_coupled_training.py \
        --dataset=grayscott --data_path=<path>/gs_512.pt \
        --output_dir=./output --arch=unet \
        --batch_size=32 --lr=0.0006 --phi_lr=0.0002 --epochs=500 \
        --warmup_epochs=100 --dropout=0.1 \
        --train_ratio=0.6305 --val_ratio=0.1232 \
        --bezier_hidden=64 \
        --seq_loss --auto_resume

Usage (LV 1D):
    python bezier_coupled_training.py \
        --dataset=lv --data_path=<path>/lv_512.pt \
        --arch=unet1d ...

Meta-gradient update (Algorithm 1 faithful, 2× memory):
    add --meta_grad
"""

import argparse
import datetime
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.aggregation import MeanMetric

from data_loaders.grayscott_loader import build_grayscott_dataloader
from data_loaders.lv_loader import build_lv_dataloader
from models.model_configs import instantiate_bezier_coupled_model
from train_arg_parser import get_args_parser
from training import distributed_mode
from training.load_and_save import load_model, save_model
from training.bezier_training_loop import (
    train_bezier_one_epoch,
    train_bezier_step,
    train_bezier_meta_step,
    evaluate_bezier_rel_l2,
)

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Arg parser extension
# ─────────────────────────────────────────────────────────────────────────────

def get_bezier_args_parser():
    parser = get_args_parser()

    parser.add_argument(
        "--phi_lr", default=None, type=float,
        help="Learning rate for the encoding network φ. Defaults to --lr if not set.",
    )
    parser.add_argument(
        "--bezier_hidden", default=64, type=int,
        help="Hidden channels in BezierEncodingNet (default 64).",
    )
    parser.add_argument(
        "--meta_grad", action="store_true", default=False,
        help="Use true Algorithm 1 meta-gradient for φ update (2× memory vs. default).",
    )
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_model(model):
    logger.info("=" * 91)
    num_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params += param.numel()
            logger.info(f"{name:56} | {str(list(param.shape)):24} | std={param.std().item():.4f}")
    logger.info("=" * 91)
    logger.info(f"Total trainable params: {num_params:,}")


def get_data_loaders(args):
    kwargs = dict(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        horizon=1,
        train_ratio=getattr(args, "train_ratio", 0.8),
        val_ratio=getattr(args, "val_ratio", 0.1),
    )

    if args.dataset == "grayscott":
        _, train_loader = build_grayscott_dataloader(split="train", normalize=False, **kwargs)
        _, val_loader   = build_grayscott_dataloader(split="test",  normalize=False,
                                                     shuffle=False, drop_last=False, **kwargs)
    elif args.dataset == "lv":
        _, train_loader = build_lv_dataloader(split="train", **kwargs)
        _, val_loader   = build_lv_dataloader(split="test",  shuffle=False, drop_last=False, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported dataset: {args.dataset}")

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    distributed_mode.init_distributed_mode(args)

    if distributed_mode.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logger.addHandler(logging.NullHandler())

    logger.info("{}".format(args).replace(", ", ",\n"))

    if distributed_mode.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    device = torch.device(args.device)
    torch.manual_seed(args.seed + distributed_mode.get_rank())
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset}")
    data_loader_train, data_loader_val = get_data_loaders(args)
    logger.info(f"Train batches: {len(data_loader_train)},  Val batches: {len(data_loader_val)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Instantiating CoupledFlowBezier")
    model = instantiate_bezier_coupled_model(args)
    model.to(device)

    model_without_ddp = model
    print_model(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,
            gradient_as_bucket_view=True,
        )
        model_without_ddp = model.module

    # ── Optimizer: two param groups (θ and φ with separate LRs) ──────────────
    phi_lr = args.phi_lr if args.phi_lr is not None else args.lr

    theta_params = (
        list(model_without_ddp.net1.parameters())
        + list(model_without_ddp.net2.parameters())
    )
    phi_params = list(model_without_ddp.encoding_net.parameters())

    optimizer = torch.optim.Adam(
        [
            {"params": theta_params, "lr": args.lr},
            {"params": phi_params,   "lr": phi_lr},
        ],
        betas=args.optimizer_betas,
        weight_decay=0.0,
    )

    # ── LR schedule (linear warmup then constant) ─────────────────────────────
    total_iters   = args.epochs * len(data_loader_train)
    warmup_iters  = args.warmup_epochs * len(data_loader_train)
    warmup_sched  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8 / args.lr, end_factor=1.0, total_iters=warmup_iters,
    )
    main_sched    = torch.optim.lr_scheduler.ConstantLR(
        optimizer, total_iters=total_iters, factor=1.0,
    )
    lr_schedule   = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_iters],
    )

    # ── Auto-resume ───────────────────────────────────────────────────────────
    if getattr(args, "auto_resume", False) and not args.resume:
        last_ckpt = os.path.join(args.output_dir, "checkpoint-last.pth")
        if os.path.isfile(last_ckpt):
            args.resume = last_ckpt
            logger.info(f"Auto-resuming from {last_ckpt}")
        else:
            logger.info("No checkpoint-last.pth found, starting from scratch.")

    load_model(args=args, model_without_ddp=model_without_ddp,
               optimizer=optimizer, lr_schedule=lr_schedule)

    # ── Train step ────────────────────────────────────────────────────────────
    if args.meta_grad:
        logger.info("Using true Algorithm 1 meta-gradient: θ updated by L_local+L_global, φ by meta-chain (2× memory)")
        _step = train_bezier_meta_step  # theta_lr passed dynamically per step from optimizer
    else:
        _step = train_bezier_step

    compiled_train_step = torch.compile(_step, disable=not getattr(args, "compile", False))

    # ── Meters ────────────────────────────────────────────────────────────────
    batch_loss = MeanMetric().to(device, non_blocking=True)
    batch_time = MeanMetric().to(device, non_blocking=True)
    meters = {"batch_loss": batch_loss, "batch_time": batch_time}

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"Training epochs {args.start_epoch} → {args.epochs}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        if not getattr(args, "eval_only", False):
            train_bezier_one_epoch(
                model=model,
                compiled_train_step=compiled_train_step,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                log_writer=log_writer,
                args=args,
                meters=meters,
            )

        eval_due = (
            args.eval_frequency > 0
            and (epoch + 1) % args.eval_frequency == 0
        )
        if eval_due or getattr(args, "eval_only", False) or getattr(args, "test_run", False):
            if not getattr(args, "eval_only", False):
                save_model(args=args, model_without_ddp=model_without_ddp,
                           optimizer=optimizer, lr_schedule=lr_schedule, epoch=epoch)

            # Evaluate all EMA variants
            eval_nets = [
                ("noema", model_without_ddp.net1, model_without_ddp.net2),
                ("ema",   model_without_ddp.net1_ema, model_without_ddp.net2_ema),
            ]
            for i, decay in enumerate(model_without_ddp.ema_decays):
                eval_nets.append((
                    f"ema{decay}",
                    model_without_ddp._modules[f"net1_ema{i+1}"],
                    model_without_ddp._modules[f"net2_ema{i+1}"],
                ))

            for suffix, n1, n2 in eval_nets:
                rel_l2, r1, r2 = evaluate_bezier_rel_l2(
                    model_without_ddp, data_loader_val, device, n1, n2, args,
                )
                logger.info(
                    f"Epoch {epoch+1} [{suffix}] val: "
                    f"rel_l2={rel_l2:.6f}  rel_l2_1={r1:.6f}  rel_l2_2={r2:.6f}"
                )
                if log_writer is not None:
                    log_writer.add_scalar(f"rel_L2_{suffix}", rel_l2, epoch + 1)

        if getattr(args, "test_run", False) or getattr(args, "eval_only", False):
            break

    total = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Training time {total}")


if __name__ == "__main__":
    parser = get_bezier_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
