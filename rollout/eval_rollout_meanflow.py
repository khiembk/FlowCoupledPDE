"""
Evaluate CoupledFlow (MeanFlow) models on multi-step GS rollout.

Rollout strategy: autoregressive 1-step chaining.
  ẑ_{1τ} = model.sample(z_0)
  ẑ_{2τ} = model.sample(ẑ_{1τ})
  ...

Ground truth from gs_rollout_gt.pt shape [N_test, 2, 6, H, W]:
  dim 2 = [0τ, 1τ, 2τ, 3τ, 4τ, 5τ]

Usage:
    python rollout/eval_rollout_meanflow.py \
        --gt_path  /scratch/user/u.kt348068/PDE_data/gs_rollout_gt.pt \
        --ckpt_dir /scratch/user/u.kt348068/ckpt \
        --models   gs_gp_512 naive_gs512 gs_gp_1024 naive_gs1024 \
        --device   cuda
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "meanflow"))

from models.model_configs import instantiate_coupled_model  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["args"]
    if isinstance(raw, dict):
        raw = argparse.Namespace(**raw)

    # fill in fields that older checkpoints may not have saved
    _defaults = dict(arch="unet", use_gp=False, dropout=0.2,
                     ema_decays=[0.99995, 0.9996], use_edm_aug=False,
                     seq_loss=False, n_proc=2)
    for k, v in _defaults.items():
        if not hasattr(raw, k):
            setattr(raw, k, v)

    model = instantiate_coupled_model(raw).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def _ema_variants(model):
    """Return list of (label, net1, net2) for all EMA copies."""
    variants = [("noema", model.net1, model.net2),
                ("ema",   model.net1_ema, model.net2_ema)]
    for i, decay in enumerate(model.ema_decays):
        variants.append((f"ema{decay}",
                         model._modules[f"net1_ema{i + 1}"],
                         model._modules[f"net2_ema{i + 1}"]))
    return variants


@torch.no_grad()
def _rollout(model, z0, n_steps, net1, net2):
    """
    z0   : [B, 2, H, W]
    returns list of n_steps tensors [B, 2, H, W], one per step
    """
    src1 = z0[:, 0:1]
    src2 = z0[:, 1:2]
    preds = []
    for _ in range(n_steps):
        p1, p2 = model.sample(src1, src2, net1=net1, net2=net2)
        out = torch.cat([p1, p2], dim=1)   # [B, 2, H, W]
        preds.append(out)
        src1, src2 = p1, p2
    return preds


def _rel_l2(pred, gt):
    """Relative L2 over the full batch."""
    num = ((pred - gt) ** 2).sum()
    den = (gt ** 2).sum().clamp_min(1e-8)
    return (num / den).sqrt().item()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path",   required=True,
                   help="Path to gs_rollout_gt.pt  [N_test,2,6,H,W]")
    p.add_argument("--ckpt_dir",  required=True,
                   help="Root checkpoint directory (one subdir per model)")
    p.add_argument("--models",    nargs="+",
                   default=["gs_gp_512", "naive_gs512"],
                   help="Subdir names under ckpt_dir")
    p.add_argument("--ckpt_name", default="checkpoint-last.pth")
    p.add_argument("--max_steps", type=int, default=5,
                   help="Number of rollout steps (1τ .. max_steps*τ)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",    default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── ground truth ──────────────────────────────────────────────────────────
    # shape: [N_test, 2, 6, H, W]  — dim 2: [0τ,1τ,2τ,3τ,4τ,5τ]
    gt = torch.load(args.gt_path, map_location="cpu")
    N_test, n_chan, n_t, H, W = gt.shape
    assert n_t >= args.max_steps + 1, \
        f"gt has only {n_t} time steps but max_steps={args.max_steps}"
    print(f"GT loaded: {tuple(gt.shape)}  (N={N_test}, steps=0τ..{n_t-1}τ)")

    # split GT into batches on CPU; move per-batch to device during eval
    z0_all = gt[:, :, 0, :, :]   # [N_test, 2, H, W]

    header = f"\n{'Model/Variant':<30}" + "".join(f"  {k+1}τ      " for k in range(args.max_steps))
    sep    = "-" * len(header)

    # ── evaluate each checkpoint ──────────────────────────────────────────────
    for model_name in args.models:
        ckpt_path = Path(args.ckpt_dir) / model_name / args.ckpt_name
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt_path} not found")
            continue
        print(f"\nLoading {model_name} …")
        model = _load_model(str(ckpt_path), device)

        print(header)
        print(sep)

        for label, net1, net2 in _ema_variants(model):
            # accumulate per-step numerator/denominator
            nums = [0.0] * args.max_steps
            dens = [0.0] * args.max_steps

            # process in batches
            for start in range(0, N_test, args.batch_size):
                end  = min(start + args.batch_size, N_test)
                z0_b = z0_all[start:end].to(device)

                preds = _rollout(model, z0_b, args.max_steps, net1, net2)

                for k, pred_k in enumerate(preds):
                    gt_k = gt[start:end, :, k + 1, :, :].to(device)
                    nums[k] += ((pred_k - gt_k) ** 2).sum().item()
                    dens[k] += (gt_k ** 2).sum().item()

            rel_l2s = [(n / max(d, 1e-8)) ** 0.5 for n, d in zip(nums, dens)]
            row = f"{model_name}/{label:<20}" + "".join(f"  {v:.6f}" for v in rel_l2s)
            print(row)

        print(sep)


if __name__ == "__main__":
    main()
