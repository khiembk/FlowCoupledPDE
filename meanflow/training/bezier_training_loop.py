"""
Training loop for CoupledFlowBezier.

Two update modes:

  Default (seq_loss):
      Step 1 — backward(L_local):  accumulates grads on both θ (flow nets)
                                    and φ (encoding net) via z_t.
      Step 2 — backward(L_global): accumulates grads on θ only (endpoints
                                    are φ-independent).
      One optimizer.step() updates everything.

  Meta-gradient (--meta_grad, Algorithm 1 faithful):
      Step 1 — compute ∇_θ L_local with create_graph=True (keeps the graph).
      Step 2 — evaluate L_global with virtual θ̃ = θ - lr·∇_θ L_local using
               functional_call (no in-place param mutation, graph intact).
      backward(L_global) flows: L_global → θ̃ → ∇_θ L_local → z_t → φ.
      Memory cost: ~2× vs. the default step.
"""

import argparse
import gc
import logging
import math
import time
from typing import Callable, Iterable, Any

import torch
import torch.distributed as dist
from torch.func import functional_call
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_bezier_rel_l2(model_without_ddp, data_loader, device, net1, net2, args=None):
    """Relative L2 error for both processes."""
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
    return (rel_l2_1 + rel_l2_2) / 2, rel_l2_1, rel_l2_2


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers (same as coupled_training_loop)
# ─────────────────────────────────────────────────────────────────────────────

def synchronize_gradients(model: torch.nn.Module):
    if not isinstance(model, DistributedDataParallel):
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for param in model.module.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()


# ─────────────────────────────────────────────────────────────────────────────
# Training steps
# ─────────────────────────────────────────────────────────────────────────────

def train_bezier_step(model_without_ddp, source_1, source_2, target_1, target_2, aug_cond=None):
    """
    Practical two-step (Algorithm 1 approximation).

    Sequential backward passes so their activation graphs are never both live
    simultaneously (halves peak activation memory vs. summing first).

    φ receives gradient via:  L_local → z_t(φ) and v_t(φ)
    θ receives gradient via:  L_local (JVP path) + L_global (endpoint matching)
    """
    local_loss = model_without_ddp.forward_local_loss(source_1, source_2, target_1, target_2)
    local_loss.backward()

    global_loss = model_without_ddp.forward_global_loss(source_1, source_2, target_1, target_2)
    global_loss.backward()

    return (local_loss + global_loss).detach()


def train_bezier_meta_step(model_without_ddp, source_1, source_2, target_1, target_2,
                            theta_lr: float, aug_cond=None):
    """
    True Algorithm 1 meta-gradient update (Eq. 12–14 in the paper).

    φ receives gradient via the full chain:
        L_global(θ̃) → θ̃ → ∇_θ L_local → z_t(φ), v_t(φ)

    Implementation:
        1. Compute ∇_θ L_local with create_graph=True (keeps the inner graph).
        2. Build virtual θ̃ = θ - theta_lr · ∇_θ L_local  (in-graph tensors,
           no in-place mutation so the chain is intact).
        3. Evaluate L_global with functional_call(θ̃) — no param mutation.
        4. backward(L_global) propagates meta-gradient all the way to φ.

    Also backprop ∇_θ L_local manually so θ gets the standard local gradient.

    Memory: roughly 2× the standard step (create_graph keeps intermediate acts).
    """
    # ── Gather named theta parameters ────────────────────────────────────────
    net1_named = dict(model_without_ddp.net1.named_parameters())
    net2_named = dict(model_without_ddp.net2.named_parameters())
    theta_params = list(net1_named.values()) + list(net2_named.values())
    n1 = len(net1_named)

    # ── Step 1: local loss, keep graph for meta-gradient ─────────────────────
    local_loss = model_without_ddp.forward_local_loss(source_1, source_2, target_1, target_2)

    theta_grads = torch.autograd.grad(
        local_loss, theta_params,
        create_graph=True,   # keep graph so L_global can flow back through these grads to φ
        allow_unused=True,
    )

    # Assign local gradients to .grad so optimizer.step() picks them up
    for p, g in zip(theta_params, theta_grads):
        if g is not None:
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.add_(g.detach())

    # ── Step 2: virtual θ̃ and L_global via functional_call ──────────────────
    net1_grads = theta_grads[:n1]
    net2_grads = theta_grads[n1:]

    net1_tilde = {
        k: p - theta_lr * g if g is not None else p
        for (k, p), g in zip(net1_named.items(), net1_grads)
    }
    net2_tilde = {
        k: p - theta_lr * g if g is not None else p
        for (k, p), g in zip(net2_named.items(), net2_grads)
    }

    device = source_1.device
    dtype  = source_1.dtype
    bsz    = source_1.shape[0]

    t1   = torch.ones(bsz,  device=device, dtype=dtype)
    r0   = torch.zeros(bsz, device=device, dtype=dtype)
    t1_b = model_without_ddp._expand_time_like(t1, source_1)
    r0_b = model_without_ddp._expand_time_like(r0, source_1)
    h_b  = t1_b - r0_b
    inp  = torch.cat([source_1, source_2], dim=1)

    u1_tilde = functional_call(
        model_without_ddp.net1, net1_tilde,
        (inp, (t1_b.view(-1), h_b.view(-1))),
        kwargs={"aug_cond": None},
    )
    u2_tilde = functional_call(
        model_without_ddp.net2, net2_tilde,
        (inp, (t1_b.view(-1), h_b.view(-1))),
        kwargs={"aug_cond": None},
    )

    v1_global = source_1 - target_1
    v2_global = source_2 - target_2

    # Use adaptive weighting consistent with _adaptive_reduce
    args = model_without_ddp.args
    sq1 = ((u1_tilde - v1_global) ** 2).flatten(1).sum(dim=1)
    sq2 = ((u2_tilde - v2_global) ** 2).flatten(1).sum(dim=1)
    if getattr(args, "use_adaptive_weight", True):
        sq1 = sq1 / (sq1.detach() + args.norm_eps) ** args.norm_p
        sq2 = sq2 / (sq2.detach() + args.norm_eps) ** args.norm_p
    global_loss = sq1.mean() + sq2.mean()

    # backward flows: global_loss → θ̃ → ∇_θ L_local → z_t(φ) → φ
    global_loss.backward()

    return (local_loss + global_loss).detach()


# ─────────────────────────────────────────────────────────────────────────────
# Epoch loop
# ─────────────────────────────────────────────────────────────────────────────

def train_bezier_one_epoch(
    model: torch.nn.Module,
    compiled_train_step: Callable,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    log_writer: Any,
    args: argparse.Namespace,
    meters: dict,
):
    gc.collect()
    model.train(True)

    batch_loss = meters["batch_loss"]
    batch_time = meters["batch_time"]
    model_without_ddp = model if not isinstance(model, DistributedDataParallel) else model.module

    tic = time.time()
    for data_iter_step, batch in enumerate(data_loader):
        steps = data_iter_step + len(data_loader) * epoch

        if data_iter_step > 0 and getattr(args, "test_run", False):
            break

        optimizer.zero_grad(set_to_none=True)

        source_1, source_2, target_1, target_2 = [x.to(device, non_blocking=True) for x in batch]

        loss = compiled_train_step(
            model_without_ddp,
            source_1, source_2, target_1, target_2,
        )

        synchronize_gradients(model)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        batch_loss.update(loss_value)

        optimizer.step()
        model_without_ddp.update_ema()

        toc = time.time()
        batch_time.update(toc - tic)
        tic = toc

        lr = optimizer.param_groups[0]["lr"]
        lr_schedule.step()

        if (steps + 1) % args.log_per_step == 0:
            loss_ave     = batch_loss.compute().detach().cpu().item()
            sec_per_iter = batch_time.compute()
            batch_time.reset()
            batch_loss.reset()

            phi_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: "
                f"loss={loss_ave:.6f}  lr={lr:.2e}  phi_lr={phi_lr:.2e}"
            )

            if log_writer is not None:
                epoch_1000x = int(steps / len(data_loader) * 1000)
                log_writer.add_scalar("loss",    loss_ave,     epoch_1000x)
                log_writer.add_scalar("lr",      lr,           epoch_1000x)
                log_writer.add_scalar("phi_lr",  phi_lr,       epoch_1000x)
