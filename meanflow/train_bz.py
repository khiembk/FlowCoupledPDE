"""
Training script for BZ (Belousov-Zhabotinsky) 3-process coupled flow.

Completely separate from train_coupled.py — does NOT touch any code path
used by GS or LV training.

Usage (from meanflow/):
    python train_bz.py \
        --data_path /scratch/.../bz_512.pt \
        --output_dir /scratch/.../ckpt/bz512_gp \
        --train_ratio 0.6305 --val_ratio 0.1232 \
        --batch_size 32 --epochs 500 --warmup_epochs 100 \
        --lr 0.0006 --dropout 0.1 \
        --ema_decay 0.9999 --ema_decays 0.99995 0.9996 \
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
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.aggregation import MeanMetric

from data_loaders.bz_loader import build_bz_dataloader
from models.model_configs import instantiate_coupled_bz_model
from training import distributed_mode
from training.load_and_save import load_model, save_model
from training.bz_training_loop import (
    train_bz_one_epoch,
    bz_seq_loss_step,
    bz_combined_loss_step,
    evaluate_bz_rel_l2,
)

torch.set_float32_matmul_precision('high')
logger = logging.getLogger(__name__)


def get_args_parser():
    p = argparse.ArgumentParser("BZ coupled flow training", add_help=True)
    # Data
    p.add_argument("--data_path", required=True)
    p.add_argument("--train_ratio", type=float, default=0.6305)
    p.add_argument("--val_ratio",   type=float, default=0.1232)
    # Training
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--epochs",         type=int,   default=500)
    p.add_argument("--warmup_epochs",  type=int,   default=100)
    p.add_argument("--lr",             type=float, default=6e-4)
    p.add_argument("--optimizer_betas", type=float, nargs=2, default=[0.9, 0.999])
    p.add_argument("--dropout",        type=float, default=0.1)
    # EMA
    p.add_argument("--ema_decay",      type=float, default=0.9999)
    p.add_argument("--ema_decays",     type=float, nargs="*", default=[0.99995, 0.9996])
    # Loss / sampling
    p.add_argument("--seq_loss",       action="store_true")
    p.add_argument("--tr_sampler",     default="v1", choices=["v0", "v1"])
    p.add_argument("--ratio",          type=float, default=0.75)
    p.add_argument("--P_mean_t",       type=float, default=-0.6)
    p.add_argument("--P_std_t",        type=float, default=1.6)
    p.add_argument("--P_mean_r",       type=float, default=-4.0)
    p.add_argument("--P_std_r",        type=float, default=1.6)
    p.add_argument("--norm_p",         type=float, default=0.75)
    p.add_argument("--norm_eps",       type=float, default=1e-3)
    # Logging / checkpointing
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--eval_frequency", type=int,   default=10)
    p.add_argument("--log_per_step",   type=int,   default=100)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--pin_mem",        action="store_true")
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--auto_resume",    action="store_true")
    p.add_argument("--resume",         default="")
    p.add_argument("--start_epoch",    type=int,   default=0)
    p.add_argument("--test_run",       action="store_true")
    p.add_argument("--eval_only",      action="store_true")
    p.add_argument("--compile",        action="store_true")
    # Distributed (parsed by distributed_mode)
    p.add_argument("--dist_url",       default="env://")
    p.add_argument("--dist_on_itp",    action="store_true")
    p.add_argument("--local_rank",     type=int,   default=-1)
    return p


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
    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.output_dir) if distributed_mode.is_main_process() else None

    device = torch.device(args.device)
    torch.manual_seed(args.seed + distributed_mode.get_rank())
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # ── data ────────────────────────────────────────────────────────────────
    kw = dict(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        horizon=1,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    _, train_loader = build_bz_dataloader(split="train", shuffle=True, pin_memory=args.pin_mem, **kw)
    _, test_loader  = build_bz_dataloader(split="test",  shuffle=False, drop_last=False, **kw)
    logger.info(f"Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")

    # ── model ───────────────────────────────────────────────────────────────
    # args.arch is not used here — always unet1d_bz
    args.arch = "unet1d_bz"
    model = instantiate_coupled_bz_model(args)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,
        )
    model_without_ddp = model if not args.distributed else model.module

    opt_params = (
        list(model_without_ddp.net1.parameters())
        + list(model_without_ddp.net2.parameters())
        + list(model_without_ddp.net3.parameters())
    )
    optimizer = torch.optim.Adam(opt_params, lr=args.lr, betas=args.optimizer_betas, weight_decay=0.0)

    warmup_iters = args.warmup_epochs * len(train_loader)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8 / args.lr, end_factor=1.0, total_iters=warmup_iters
    )
    main_sched = torch.optim.lr_scheduler.ConstantLR(
        optimizer, total_iters=args.epochs * len(train_loader), factor=1.0
    )
    lr_schedule = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_iters]
    )

    # ── auto-resume ─────────────────────────────────────────────────────────
    if getattr(args, "auto_resume", False) and not args.resume:
        last_ckpt = os.path.join(args.output_dir, "checkpoint-last.pth")
        if os.path.isfile(last_ckpt):
            args.resume = last_ckpt
            logger.info(f"Auto-resuming from {last_ckpt}")

    load_model(args=args, model_without_ddp=model_without_ddp,
               optimizer=optimizer, lr_schedule=lr_schedule)

    _step_fn = bz_seq_loss_step if args.seq_loss else bz_combined_loss_step
    compiled_train_step = torch.compile(_step_fn, disable=not args.compile)

    meters = {
        "batch_loss": MeanMetric().to(device),
        "batch_time": MeanMetric().to(device),
    }

    # ── training loop ───────────────────────────────────────────────────────
    logger.info(f"Start training BZ: epochs {args.start_epoch}–{args.epochs}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if not args.eval_only:
            train_bz_one_epoch(
                model=model,
                compiled_train_step=compiled_train_step,
                data_loader=train_loader,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                log_writer=log_writer,
                args=args,
                meters=meters,
            )

        do_eval = (
            args.eval_frequency > 0
            and (epoch + 1) % args.eval_frequency == 0
        ) or args.eval_only or args.test_run

        if args.output_dir and do_eval:
            if not args.eval_only:
                save_model(args=args, model_without_ddp=model_without_ddp,
                           optimizer=optimizer, lr_schedule=lr_schedule, epoch=epoch)
                logger.info(f"Saved checkpoint to {args.output_dir}")

            eval_nets = [
                ("ema",   model_without_ddp.net1_ema, model_without_ddp.net2_ema, model_without_ddp.net3_ema),
                ("noema", model_without_ddp.net1,     model_without_ddp.net2,     model_without_ddp.net3),
            ]
            for i, decay in enumerate(model_without_ddp.ema_decays):
                eval_nets.append((
                    f"ema{decay}",
                    model_without_ddp._modules[f"net1_ema{i+1}"],
                    model_without_ddp._modules[f"net2_ema{i+1}"],
                    model_without_ddp._modules[f"net3_ema{i+1}"],
                ))

            for suffix, n1, n2, n3 in eval_nets:
                rl, r1, r2, r3 = evaluate_bz_rel_l2(
                    model_without_ddp, test_loader, device, n1, n2, n3, args
                )
                logger.info(
                    f"Eval epoch {epoch+1} [{suffix}] (test): "
                    f"rel-L2 = {rl:.6f} (u: {r1:.6f}, v: {r2:.6f}, w: {r3:.6f})"
                )
                if log_writer is not None:
                    log_writer.add_scalar(f"rel_L2_{suffix}",   rl, epoch + 1)
                    log_writer.add_scalar(f"rel_L2_u_{suffix}", r1, epoch + 1)
                    log_writer.add_scalar(f"rel_L2_v_{suffix}", r2, epoch + 1)
                    log_writer.add_scalar(f"rel_L2_w_{suffix}", r3, epoch + 1)

        if args.test_run or args.eval_only:
            break

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
