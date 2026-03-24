import argparse
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


def synchronize_gradients(model: torch.nn.Module):
    
    if not isinstance(model, DistributedDataParallel):
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for param in model.module.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()


def gradient_sanity_check(model: torch.nn.Module):
    
    if not isinstance(model, DistributedDataParallel):
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for name, p in model.module.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        if len(p.shape) <= 1:
            continue

        monitor = p.grad.norm()
        monitor_list = [torch.zeros_like(monitor) for _ in range(dist.get_world_size())]
        dist.all_gather(monitor_list, monitor)
        monitor_tensor = torch.stack(monitor_list)

        ref = monitor_tensor[0]
        for i, m in enumerate(monitor_tensor):
            if not torch.isclose(m, ref, rtol=1e-5, atol=1e-7):
                raise RuntimeError(
                    f"Gradient norm mismatch for {name} at rank {i}: {m} vs rank 0: {ref}"
                )


def get_compiled_counts():
    metrics = torch._dynamo.utils.get_compilation_metrics()
    return len(metrics)


def train_combined_loss_step(model_without_ddp, *args, **kwargs):
    
    loss = model_without_ddp.forward_combined_loss(*args, **kwargs)
    loss.backward(create_graph=False)
    return loss

def train_local_loss_step(model_without_ddp, *args, **kwargs):
    
    loss = model_without_ddp.forward_local_loss(*args, **kwargs)
    loss.backward(create_graph=False)
    return loss


def maybe_normalize(x: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    
    if getattr(args, "input_scale", None) is not None:
        x = x / args.input_scale
    return x


def move_batch_to_device(batch, device):
    return [x.to(device, non_blocking=True) for x in batch]


def train_one_epoch(
    model: torch.nn.Module,
    compiled_train_step: Callable,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    log_writer: Any,
    args: argparse.Namespace,
    meters: dict[str, MeanMetric],
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

        # Expected:
        # batch = (z1_1, z1_2, z0_1, z0_2)
        if len(batch) != 4:
            raise ValueError(
                "Expected batch format (z1_1, z1_2, z0_1, z0_2), "
                f"but got length {len(batch)}"
            )

        z1_1, z1_2, z0_1, z0_2 = move_batch_to_device(batch, device)

        # PDE-specific normalization hook
        z1_1 = maybe_normalize(z1_1, args)
        z1_2 = maybe_normalize(z1_2, args)
        z0_1 = maybe_normalize(z0_1, args)
        z0_2 = maybe_normalize(z0_2, args)

        # Usually no EDM augmentation for coupled PDE training
        aug_cond = None

        if args.compile and epoch == args.start_epoch and data_iter_step == 0:
            logger.info("Compiling the first train step, this may take a while...")

        loss = compiled_train_step(
            model_without_ddp,
            z1_1,
            z1_2,
            z0_1,
            z0_2,
            aug_cond,
        )

        if args.compile:
            assert get_compiled_counts() > 0, "Compilation not triggered."

        synchronize_gradients(model)

        if getattr(args, "grad_sanity_check", False):
            if (epoch - args.start_epoch) % 50 == 0 and data_iter_step < 2:
                gradient_sanity_check(model)

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
            metrics = {
                "loss": loss_ave,
                "lr": lr,
                "epoch": steps / len(data_loader),
                "steps": steps,
                "sec_per_iter": sec_per_iter,
            }

            if log_writer is not None:
                for k, v in metrics.items():
                    log_writer.add_scalar(f"ep_{k}", v, epoch_1000x)

    return