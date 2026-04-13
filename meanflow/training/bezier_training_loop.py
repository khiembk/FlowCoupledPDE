"""
Training loop for CoupledFlowBezier.

Two update modes:

  Default (train_bezier_step):
      Step 1 — backward(L_local):  θ and φ both get gradients (φ via z_t).
      Step 2 — backward(L_global): θ gets additional gradient (endpoints).
      One optimizer.step() updates everything.

  Meta-gradient (--meta_grad, Algorithm 1 faithful):
      θ is updated TWICE sequentially (local then global), φ once (meta-gradient):

        Step 1 (θ ← L_local):
            theta_grads = ∇_θ L_local   [create_graph=True, keeps graph for Step 2]
            θ.grad += theta_grads        [first θ contribution]

        Step 2 (θ ← L_global, φ ← meta-chain):
            Build virtual θ̃ = θ - lr · theta_grads   (in-graph, no .data mutation)
            Evaluate L_global with functional_call(θ̃)
            Jointly compute:
              ∇_θ L_global(θ̃)  via autograd.grad(global_loss, theta_params)
                  chain: global_loss → θ̃ → theta_grads → θ
              ∇_φ L_global(θ̃)  via autograd.grad(global_loss, phi_params)
                  chain: global_loss → θ̃ → theta_grads → L_local → z_t(φ) → φ
            θ.grad += ∇_θ L_global(θ̃)  [second θ contribution]
            φ.grad  = ∇_φ L_global(θ̃)  [φ updated by meta-gradient only]

      lr passed dynamically from the optimizer (tracks the schedule correctly).
      Memory: ~2× the default step.
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
    True Algorithm 1 (Eq. 12–14 in the paper).

    θ is updated TWICE (local then global); φ is updated once (meta-gradient only):

      θ.grad = ∇_θ L_local  +  ∇_θ L_global(θ̃)    ← two sequential contributions
      φ.grad = ∇_φ L_global(θ̃)                      ← meta-chain only

    theta_lr must be the CURRENT lr from the optimizer (tracks the schedule).
    Memory: ~2× the default step (create_graph keeps the inner graph alive).
    """
    # ── Gather parameters ────────────────────────────────────────────────────
    net1_named   = dict(model_without_ddp.net1.named_parameters())
    net2_named   = dict(model_without_ddp.net2.named_parameters())
    theta_params = list(net1_named.values()) + list(net2_named.values())
    phi_params   = list(model_without_ddp.encoding_net.parameters())
    n1 = len(net1_named)

    # ── Step 1: ∇_θ L_local  (first θ contribution) ──────────────────────────
    local_loss = model_without_ddp.forward_local_loss(source_1, source_2, target_1, target_2)

    # create_graph=True keeps the graph so Step 2 can differentiate through
    # theta_grads to reach φ (meta-chain) and θ (second-order θ gradient).
    theta_grads_local = torch.autograd.grad(
        local_loss, theta_params,
        create_graph=True,
        allow_unused=True,
    )

    # ── Step 2: L_global(θ̃)  →  second θ contribution + φ meta-gradient ─────
    # Virtual θ̃ = θ - lr · ∇_θ L_local  (in-graph, no .data mutation)
    net1_tilde = {
        k: p - theta_lr * g if g is not None else p
        for (k, p), g in zip(net1_named.items(), theta_grads_local[:n1])
    }
    net2_tilde = {
        k: p - theta_lr * g if g is not None else p
        for (k, p), g in zip(net2_named.items(), theta_grads_local[n1:])
    }

    device = source_1.device
    dtype  = source_1.dtype
    bsz    = source_1.shape[0]

    t1   = torch.ones(bsz,  device=device, dtype=dtype)
    r0   = torch.zeros(bsz, device=device, dtype=dtype)
    t1_b = model_without_ddp._expand_time_like(t1, source_1)
    h_b  = t1_b - model_without_ddp._expand_time_like(r0, source_1)
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

    args = model_without_ddp.args
    sq1  = ((u1_tilde - v1_global) ** 2).flatten(1).sum(dim=1)
    sq2  = ((u2_tilde - v2_global) ** 2).flatten(1).sum(dim=1)
    if getattr(args, "use_adaptive_weight", True):
        sq1 = sq1 / (sq1.detach() + args.norm_eps) ** args.norm_p
        sq2 = sq2 / (sq2.detach() + args.norm_eps) ** args.norm_p
    global_loss = sq1.mean() + sq2.mean()

    # Jointly differentiate global_loss w.r.t. theta_params AND phi_params
    # in one autograd.grad call (avoids two separate backward passes on the
    # same graph and the need for retain_graph=True).
    #
    # theta chain: global_loss → θ̃ → theta_grads_local → theta_params
    #              (includes the Hessian term from create_graph=True)
    # phi chain:   global_loss → θ̃ → theta_grads_local → L_local → z_t(φ) → phi
    all_grads = torch.autograd.grad(
        global_loss, theta_params + phi_params,
        allow_unused=True,
    )
    theta_grads_global = all_grads[:len(theta_params)]
    phi_grads          = all_grads[len(theta_params):]

    # Write accumulated gradients into .grad
    # θ: sum of local contribution + global contribution (two sequential updates)
    for p, g_loc, g_glob in zip(theta_params, theta_grads_local, theta_grads_global):
        g = 0
        if g_loc  is not None: g = g + g_loc.detach()
        if g_glob is not None: g = g + g_glob.detach()
        p.grad = g if isinstance(g, torch.Tensor) else None

    # φ: meta-gradient only (no direct L_local contribution)
    for p, g in zip(phi_params, phi_grads):
        if g is not None:
            p.grad = g.detach().clone()

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

        # For the meta step, pass the current lr from the scheduler so that the
        # virtual θ̃ update uses the same step size as the real optimizer.
        if getattr(args, "meta_grad", False):
            current_lr = optimizer.param_groups[0]["lr"]
            loss = compiled_train_step(
                model_without_ddp,
                source_1, source_2, target_1, target_2,
                current_lr,
            )
        else:
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
