"""
Training loop for N-process CoupledFlowNProc.
Separate from coupled_training_loop.py and bz_training_loop.py.
"""
import gc
import logging
import math
import time
from typing import Iterable, Any, Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_nproc_rel_l2(model_without_ddp, data_loader, device, nets, n_proc, args=None):
    """Compute per-process relative L2 error over a dataloader."""
    model_without_ddp.eval()
    nums = [0.0] * n_proc
    dens = [0.0] * n_proc

    for batch in data_loader:
        tensors = [x.to(device, non_blocking=True) for x in batch]
        sources = tensors[:n_proc]
        targets = tensors[n_proc:]
        preds = model_without_ddp.sample(sources, nets=nets)
        for i, (pred, tgt) in enumerate(zip(preds, targets)):
            nums[i] += ((pred - tgt) ** 2).sum().item()
            dens[i] += (tgt ** 2).sum().item()
        if args is not None and getattr(args, "test_run", False):
            break

    model_without_ddp.train()
    rl = [(n / max(d, 1e-8)) ** 0.5 for n, d in zip(nums, dens)]
    return sum(rl) / n_proc, rl


def synchronize_gradients(model: torch.nn.Module):
    if not isinstance(model, DistributedDataParallel):
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for param in model.module.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()


def nproc_seq_loss_step(model_without_ddp, sources, targets, aug_cond=None):
    """Sequential backward: local then global (halves peak activation memory)."""
    local = model_without_ddp.forward_local_loss(sources, targets)
    local.backward()
    global_ = model_without_ddp.forward_global_loss(sources, targets)
    global_.backward()
    return (local + global_).detach()


def nproc_combined_loss_step(model_without_ddp, sources, targets, aug_cond=None):
    loss = model_without_ddp.forward_combined_loss(sources, targets)
    loss.backward()
    return loss.detach()


def train_nproc_one_epoch(
    model: torch.nn.Module,
    compiled_train_step: Callable,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    log_writer: Any,
    args,
    meters: dict,
    n_proc: int,
):
    gc.collect()
    model.train(True)

    batch_loss = meters["batch_loss"]
    batch_time = meters["batch_time"]
    model_without_ddp = model if not isinstance(model, DistributedDataParallel) else model.module

    tic = time.time()
    for data_iter_step, batch in enumerate(data_loader):
        steps = data_iter_step + len(data_loader) * epoch
        optimizer.zero_grad(set_to_none=True)

        if data_iter_step > 0 and args.test_run:
            break

        tensors = [x.to(device, non_blocking=True) for x in batch]
        sources = tensors[:n_proc]
        targets = tensors[n_proc:]

        loss = compiled_train_step(model_without_ddp, sources, targets, None)

        synchronize_gradients(model)

        loss_value = loss.item()
        batch_loss.update(loss_value)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        optimizer.step()
        model_without_ddp.update_ema()

        toc = time.time()
        batch_time.update(toc - tic)
        tic = toc

        lr = optimizer.param_groups[0]["lr"]
        lr_schedule.step()

        if (steps + 1) % args.log_per_step == 0:
            loss_ave = batch_loss.compute().detach().cpu().item()
            sec_per_iter = batch_time.compute()
            batch_time.reset()
            batch_loss.reset()

            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: "
                f"loss = {loss_ave:.6f}, lr = {lr:.6f}, "
                f"steps = {steps}, sec_per_iter = {sec_per_iter:.4f}"
            )
            epoch_1000x = int(steps / len(data_loader) * 1000)
            if log_writer is not None:
                log_writer.add_scalar("ep_loss", loss_ave, epoch_1000x)
                log_writer.add_scalar("ep_lr",   lr,       epoch_1000x)
