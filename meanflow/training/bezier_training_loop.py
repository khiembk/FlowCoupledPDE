"""
Training loop for CoupledFlowBezier.

Two update modes:

  Default (train_bezier_step):
      Step 1 — backward(L_local):  θ and φ both get gradients (φ via z_t).
      Step 2 — backward(L_global): θ gets additional gradient (endpoints).
      One optimizer.step() updates everything.

  Meta-gradient (--meta_grad, Algorithm 1 / eq. 32):
      Gradients computed via meta chain rule; one optimizer.step() applies them.

      θ.grad = ∇_θ L_local  +  ∇_θ L_global(θ̃)
      φ.grad = ∇_φ L_global(θ̃)   [eq. 32 chain rule]
             = ∇_θ̃ L_global(θ̃) · ∂θ̃/∂φ
             = ∇_θ̃ L_global(θ̃) · (-η · ∂_φ ∇_θ L_local)

      create_graph=True on ∇_θ L_local keeps the computation graph live so
      autograd can trace: L_global(θ̃) → θ̃ → ∇_θ L_local → L_local → z_t(φ) → φ.
      The virtual θ̃ = θ - η · ∇_θ L_local is required for this chain; η is the
      current optimizer lr.  Actual updates use optimizer.step() (Adam momentum).
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


def train_bezier_meta_step(model_without_ddp, optimizer,
                            source_1, source_2, target_1, target_2, aug_cond=None):
    """
    Algorithm 1 / eq. 32: meta-gradient update for φ via chain rule.

    ∇_φ L_global(θ̃) = ∇_θ̃ L_global · (-η · ∂_φ ∇_θ L_local)

    create_graph=True keeps the graph so autograd traces the full chain:
        L_global(θ̃) → θ̃ → ∇_θ L_local → L_local → z_t(φ) → φ

    Gradients are written to .grad; optimizer.step() applies them with
    Adam momentum / weight decay.
    """
    theta_lr = optimizer.param_groups[0]["lr"]

    net1_named   = dict(model_without_ddp.net1.named_parameters())
    net2_named   = dict(model_without_ddp.net2.named_parameters())
    theta_params = list(net1_named.values()) + list(net2_named.values())
    phi_params   = list(model_without_ddp.encoding_net.parameters())
    n1 = len(net1_named)

    # ── ∇_θ L_local  (create_graph keeps graph for chain rule to φ) ──────────
    local_loss = model_without_ddp.forward_local_loss(source_1, source_2, target_1, target_2)
    theta_grads_local = torch.autograd.grad(
        local_loss, theta_params, create_graph=True, allow_unused=True,
    )

    # ── Clip ∇_θ L_local (in-graph) before building θ̃ ───────────────────────
    # Scale by 1/norm (clamped to 1) so the virtual step is at most theta_lr.
    # Scaling in-graph preserves the meta-chain: global_loss → θ̃ →
    # (scale * theta_grads_local) → L_local → z_t(φ) → φ.
    _valid = [g for g in theta_grads_local if g is not None]
    if _valid:
        _total_norm = torch.stack([g.norm() for g in _valid]).norm()
        _scale = (1.0 / (_total_norm + 1e-6)).clamp(max=1.0)
    else:
        _scale = 1.0

    # ── Virtual θ̃ = θ - η · scale · ∇_θ L_local  (in-graph) ─────────────────
    net1_tilde = {
        k: p - theta_lr * _scale * g if g is not None else p
        for (k, p), g in zip(net1_named.items(), theta_grads_local[:n1])
    }
    net2_tilde = {
        k: p - theta_lr * _scale * g if g is not None else p
        for (k, p), g in zip(net2_named.items(), theta_grads_local[n1:])
    }

    # ── L_global(θ̃) via functional_call ──────────────────────────────────────
    device = source_1.device
    dtype  = source_1.dtype
    bsz    = source_1.shape[0]
    t1   = torch.ones(bsz,  device=device, dtype=dtype)
    r0   = torch.zeros(bsz, device=device, dtype=dtype)
    t1_b = model_without_ddp._expand_time_like(t1, source_1)
    r0_b = model_without_ddp._expand_time_like(r0, source_1)
    inp  = torch.cat([source_1, source_2], dim=1)

    u1_tilde = functional_call(
        model_without_ddp.net1, net1_tilde,
        (inp, (t1_b.view(-1), r0_b.view(-1))),
        kwargs={"aug_cond": None},
    )
    u2_tilde = functional_call(
        model_without_ddp.net2, net2_tilde,
        (inp, (t1_b.view(-1), r0_b.view(-1))),
        kwargs={"aug_cond": None},
    )

    v1_global = source_1 - target_1
    v2_global = source_2 - target_2
    args = model_without_ddp.args
    sq1 = ((u1_tilde - v1_global) ** 2).flatten(1).sum(dim=1)
    sq2 = ((u2_tilde - v2_global) ** 2).flatten(1).sum(dim=1)
    if getattr(args, "use_adaptive_weight", True):
        sq1 = sq1 / (sq1.detach() + args.norm_eps) ** args.norm_p
        sq2 = sq2 / (sq2.detach() + args.norm_eps) ** args.norm_p
    global_loss = sq1.mean() + sq2.mean()

    # ── ∇_φ L_global(θ̃) via eq. 32 chain rule ───────────────────────────────
    # chain: global_loss → θ̃ → theta_grads_local → L_local → z_t(φ) → φ
    all_grads = torch.autograd.grad(
        global_loss, theta_params + phi_params, allow_unused=True,
    )
    theta_grads_global = all_grads[:len(theta_params)]
    phi_grads          = all_grads[len(theta_params):]

    # ── Write .grad and update with optimizer ─────────────────────────────────
    optimizer.zero_grad(set_to_none=True)

    # θ: ∇_θ L_local  +  ∇_θ L_global(θ̃)
    for p, g_loc, g_glob in zip(theta_params, theta_grads_local, theta_grads_global):
        g = 0
        if g_loc  is not None: g = g + g_loc.detach()
        if g_glob is not None: g = g + g_glob.detach()
        p.grad = g if isinstance(g, torch.Tensor) else None

    # φ: ∇_φ L_global(θ̃) via eq. 32
    for p, g in zip(phi_params, phi_grads):
        p.grad = g.detach().clone() if g is not None else None

    # Clip final gradients before optimizer step (max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(theta_params + phi_params, max_norm=1.0)

    optimizer.step()

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

        source_1, source_2, target_1, target_2 = [x.to(device, non_blocking=True) for x in batch]

        if getattr(args, "meta_grad", False):
            # meta step owns zero_grad + optimizer.step(); handles eq. 32 chain rule for φ
            loss = train_bezier_meta_step(
                model_without_ddp, optimizer,
                source_1, source_2, target_1, target_2,
            )
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = compiled_train_step(
                model_without_ddp,
                source_1, source_2, target_1, target_2,
            )
            synchronize_gradients(model)
            optimizer.step()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")
        batch_loss.update(loss_value)

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
