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

## help me implement beizer tranining loop follow the algo in paper

@torch.no_grad()
def evaluate_coupled_rel_l2(model_without_ddp, data_loader, device, net1, net2, args=None):
    """Compute relative L2 error for both coupled components over a dataloader.

    Returns:
        (rel_l2, rel_l2_1, rel_l2_2): averaged and per-component relative L2 errors.
    """
    model_without_ddp.eval()
    l2_num_1 = l2_den_1 = 0.0
    l2_num_2 = l2_den_2 = 0.0

    for batch in data_loader:
        source_1, source_2, target_1, target_2 = [x.to(device, non_blocking=True) for x in batch]
        pred_1, pred_2 = model_without_ddp.sample(source_1, source_2, net1=net1, net2=net2)
        l2_num_1 += ((pred_1 - target_1) ** 2).sum().item()
        l2_den_1 += (target_1 ** 2).sum().item()
        l2_num_2 += ((pred_2 - target_2) ** 2).sum().item()
        l2_den_2 += (target_2 ** 2).sum().item()
        if args is not None and getattr(args, "test_run", False):
            break

    model_without_ddp.train()
    rel_l2_1 = (l2_num_1 / max(l2_den_1, 1e-8)) ** 0.5
    rel_l2_2 = (l2_num_2 / max(l2_den_2, 1e-8)) ** 0.5
    rel_l2 = (rel_l2_1 + rel_l2_2) / 2
    return rel_l2, rel_l2_1, rel_l2_2


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


def train_combined_loss_step(model_without_ddp, source_1, source_2, target_1, target_2, aug_cond=None):
    loss = model_without_ddp.forward_combined_loss(source_1, source_2, target_1, target_2)
    loss.backward()
    return loss.detach()

def train_combine_loss_squence_step(model_without_ddp, source_1, source_2, target_1, target_2, aug_cond=None):
    # Backprop local and global losses separately so their activation graphs
    # are never both live simultaneously (halves peak activation memory).
    local_loss = model_without_ddp.forward_local_loss(source_1, source_2, target_1, target_2)
    local_loss.backward()

    global_loss = model_without_ddp.forward_global_loss(source_1, source_2, target_1, target_2)
    global_loss.backward()

    return (local_loss + global_loss).detach()

def train_local_loss_step(model_without_ddp, *args, **kwargs):
    
    loss = model_without_ddp.forward_local_loss(*args, **kwargs)
    loss.backward(create_graph=False)
    return loss




def move_batch_to_device(batch, device):
    return [x.to(device, non_blocking=True) for x in batch]


def train_coupled_one_epoch(
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

        source_1, source_2, target_1, target_2 = move_batch_to_device(batch, device)


        # Usually no EDM augmentation for coupled PDE training
        aug_cond = None

        if args.compile and epoch == args.start_epoch and data_iter_step == 0:
            logger.info("Compiling the first train step, this may take a while...")

        loss = compiled_train_step(
            model_without_ddp,
            source_1,
            source_2,
            target_1,
            target_2,
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