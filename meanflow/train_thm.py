"""
Training script for THM (thermo-mechanical) 5-process coupled flow.

Uses CoupledFlowNProc with the lightweight unet_lite architecture (~30M trainable
params for 5 processes, vs ~250M for the full-size unet).
Completely separate from train_coupled.py / train_bz.py — does not touch
any code path used by GS, LV, or BZ training.

Usage (from meanflow/):
    python train_thm.py \\
        --data_path /scratch/.../thm_512.pt \\
        --output_dir /scratch/.../ckpt/thm512_meanflow \\
        --train_ratio 0.6305 --val_ratio 0.1232 \\
        --batch_size 16 --epochs 500 --warmup_epochs 100 \\
        --lr 6e-4 --dropout 0.1 \\
        --ema_decay 0.9999 --ema_decays 0.99995 0.9996 \\
        --seq_loss --use_gp --auto_resume

    # For THM-1024:
    python train_thm.py \\
        --data_path /scratch/.../thm_1024.pt \\
        --output_dir /scratch/.../ckpt/thm1024_meanflow \\
        --train_ratio 0.7734 --val_ratio 0.0755 \\
        [... same flags ...]

    # Multi-GPU:
    torchrun --standalone --nproc_per_node=4 train_thm.py [args]
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

from data_loaders.thm_loader import build_thm_dataloader
from models.model_configs import instantiate_coupled_nproc_model
from training import distributed_mode
from training.load_and_save import load_model, save_model
from training.nproc_training_loop import (
    train_nproc_one_epoch,
    nproc_seq_loss_step,
    nproc_combined_loss_step,
    evaluate_nproc_rel_l2,
)

torch.set_float32_matmul_precision('high')
logger = logging.getLogger(__name__)

N_PROC = 5  # THM always has 5 processes


def get_args_parser():
    p = argparse.ArgumentParser("THM coupled flow training (CoupledFlowNProc)", add_help=True)
    # Data
    p.add_argument("--data_path",    required=True)
    p.add_argument("--train_ratio",  type=float, default=0.6305)
    p.add_argument("--val_ratio",    type=float, default=0.1232)
    # Training
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--epochs",          type=int,   default=500)
    p.add_argument("--warmup_epochs",   type=int,   default=100)
    p.add_argument("--lr",              type=float, default=6e-4)
    p.add_argument("--optimizer_betas", type=float, nargs=2, default=[0.9, 0.999])
    p.add_argument("--dropout",         type=float, default=0.1)
    # EMA
    p.add_argument("--ema_decay",   type=float, default=0.9999)
    p.add_argument("--ema_decays",  type=float, nargs="*", default=[0.99995, 0.9996])
    # Loss / time sampling
    p.add_argument("--seq_loss",    action="store_true")
    p.add_argument("--tr_sampler",  default="v1", choices=["v0", "v1"])
    p.add_argument("--ratio",       type=float, default=0.75)
    p.add_argument("--P_mean_t",    type=float, default=-0.6)
    p.add_argument("--P_std_t",     type=float, default=1.6)
    p.add_argument("--P_mean_r",    type=float, default=-4.0)
    p.add_argument("--P_std_r",     type=float, default=1.6)
    p.add_argument("--norm_p",      type=float, default=0.75)
    p.add_argument("--norm_eps",    type=float, default=1e-3)
    # GP coupling
    p.add_argument("--use_gp",               action="store_true")
    p.add_argument("--gp_log_length_scale",  type=float, default=0.0)
    # Logging / checkpointing
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--eval_frequency", type=int,   default=10)
    p.add_argument("--log_per_step",   type=int,   default=100)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--pin_mem",        action="store_true")
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--auto_resume",      action="store_true")
    p.add_argument("--resume",           default="")
    p.add_argument("--reset_optimizer",  action="store_true")
    p.add_argument("--start_epoch",      type=int,   default=0)
    p.add_argument("--test_run",       action="store_true")
    p.add_argument("--eval_only",      action="store_true")
    p.add_argument("--compile",        action="store_true")
    # Distributed
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
    _, train_loader = build_thm_dataloader(split="train", shuffle=True,  pin_memory=args.pin_mem, **kw)
    _, test_loader  = build_thm_dataloader(split="test",  shuffle=False, drop_last=False, **kw)
    logger.info(f"Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")

    # ── model ───────────────────────────────────────────────────────────────
    model = instantiate_coupled_nproc_model(args, n_proc=N_PROC)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters (trainable): {num_params:,}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,
        )
    model_without_ddp = model if not args.distributed else model.module

    # Optimizer covers all trainable nets + GP modules
    opt_params = [p for net in model_without_ddp.nets for p in net.parameters()]
    if getattr(args, "use_gp", False):
        for i in range(N_PROC):
            opt_params += list(model_without_ddp._modules[f"gp_{i+1}"].parameters())

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

    _step_fn = nproc_seq_loss_step if args.seq_loss else nproc_combined_loss_step
    compiled_train_step = torch.compile(_step_fn, disable=not args.compile)

    meters = {
        "batch_loss": MeanMetric().to(device),
        "batch_time": MeanMetric().to(device),
    }

    # ── EMA variant specs for evaluation ────────────────────────────────────
    def _get_eval_nets(suffix, nets_list):
        return (suffix, nets_list)

    def build_eval_variants():
        variants = [
            ("noema", list(model_without_ddp.nets)),
            ("ema",   list(model_without_ddp.nets_ema)),
        ]
        for ei, decay in enumerate(model_without_ddp.ema_decays):
            nets = [model_without_ddp._modules[f"net{pi+1}_ema{ei+1}"] for pi in range(N_PROC)]
            variants.append((f"ema{decay}", nets))
        return variants

    # ── training loop ────────────────────────────────────────────────────────
    logger.info(f"Start training THM: epochs {args.start_epoch}–{args.epochs}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if not args.eval_only:
            train_nproc_one_epoch(
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
                n_proc=N_PROC,
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

            chan_names = ["T", "p", "eps_xx", "eps_yy", "eps_xy"]

            for suffix, nets in build_eval_variants():
                avg_rl, per_proc = evaluate_nproc_rel_l2(
                    model_without_ddp, test_loader, device, nets, N_PROC, args
                )
                per_str = ", ".join(f"{c}: {v:.6f}" for c, v in zip(chan_names, per_proc))
                logger.info(
                    f"Eval epoch {epoch+1} [{suffix}] (test): "
                    f"rel-L2 = {avg_rl:.6f}  ({per_str})"
                )
                if log_writer is not None:
                    log_writer.add_scalar(f"rel_L2_{suffix}", avg_rl, epoch + 1)
                    for c, v in zip(chan_names, per_proc):
                        log_writer.add_scalar(f"rel_L2_{c}_{suffix}", v, epoch + 1)

        if args.test_run or args.eval_only:
            break

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
