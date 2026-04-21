"""
Evaluate baseline neural operator models on multi-step GS rollout.

Rollout strategy: autoregressive 1-step chaining.
  ẑ_{kτ} = model(ẑ_{(k-1)τ})

Ground truth from gs_rollout_gt.pt shape [N_test, 2, 6, H, W]:
  dim 2 = [0τ, 1τ, 2τ, 3τ, 4τ, 5τ]

Usage:
    python rollout/eval_rollout_baselines.py \
        --gt_path  /scratch/user/u.kt348068/PDE_data/gs_rollout_gt.pt \
        --ckpt_dir /scratch/user/u.kt348068/ckpt \
        --dataset  gs512 \
        --device   cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "baselines"))

from models import (   # noqa: E402
    FNO2d, UFNO2d, DeepONet2d, Transolver2d, CMWNO2d, COMPOL2d,
)


# ── per-dataset model registry ───────────────────────────────────────────────
# Each entry: (ckpt_subdir, model_factory_fn)
# factory_fn() returns an nn.Module ready for load_state_dict

def _gs512_registry():
    n = 2   # n_proc

    def fno():
        return FNO2d(modes1=12, modes2=12, width=64,
                     in_channels=n, out_channels=n,
                     n_layers=4, padding=9)

    def ufno():
        return UFNO2d(modes1=12, modes2=12, width=64,
                      in_channels=n, out_channels=n,
                      n_layers=4, padding=9)

    def deeponet():
        # in_channels = n_proc * c_in = 2; p=128 (default deeponet_p)
        return DeepONet2d(in_channels=n, out_channels=n,
                          p=128, width=64)

    def transolver():
        # dim=width, n_slices=32 (default), n_heads=4, n_layers=4
        return Transolver2d(in_channels=n, out_channels=n,
                            dim=64, n_slices=32,
                            n_heads=4, n_layers=4)

    def cmwno():
        # per-process in/out, n_proc=2; wavelet_k=2 (default)
        return CMWNO2d(in_channels=1, out_channels=1,
                       n_proc=n, width=32, n_layers=4, k=2)

    def compol_rnn():
        return COMPOL2d(in_channels=1, out_channels=1, n_proc=n,
                        modes1=12, modes2=12, width=32, n_layers=4,
                        n_heads=4, padding=9, aggr_type="rnn")

    def compol_atn():
        return COMPOL2d(in_channels=1, out_channels=1, n_proc=n,
                        modes1=12, modes2=12, width=32, n_layers=4,
                        n_heads=4, padding=9, aggr_type="atn")

    return [
        ("gs512_fno2d",        fno),
        ("gs512_ufno2d",       ufno),
        ("gs512_deeponet2d",   deeponet),
        ("gs512_transolver2d", transolver),
        ("gs512_cmwno2d",      cmwno),
        ("gs512_compol2d_rnn", compol_rnn),
        ("gs512_compol2d_atn", compol_atn),
    ]


REGISTRIES = {
    "gs512": _gs512_registry,
}


# ── rollout helpers ───────────────────────────────────────────────────────────

@torch.no_grad()
def _rollout(model: nn.Module, z0: torch.Tensor, n_steps: int):
    """
    z0   : [B, 2, H, W]
    returns list of n_steps tensors [B, 2, H, W]
    """
    src = z0
    preds = []
    for _ in range(n_steps):
        pred = model(src)          # [B, 2, H, W]
        preds.append(pred)
        src = pred
    return preds


# ── key remapping for checkpoints saved with old naming conventions ───────────

def _remap_state_dict(name, state_dict):
    """Fix CMWNO2d checkpoints: net{1,2}_{row,col} → nets_{row,col}.{0,1}"""
    if "cmwno" not in name:
        return state_dict
    mapping = {"net1_row.": "nets_row.0.", "net2_row.": "nets_row.1.",
               "net1_col.": "nets_col.0.", "net2_col.": "nets_col.1."}
    new = {}
    for k, v in state_dict.items():
        for old_prefix, new_prefix in mapping.items():
            if k.startswith(old_prefix):
                k = new_prefix + k[len(old_prefix):]
                break
        new[k] = v
    return new


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path",    required=True,
                   help="Path to gs_rollout_gt.pt  [N_test,2,6,H,W]")
    p.add_argument("--ckpt_dir",   required=True,
                   help="Root checkpoint directory (one subdir per model)")
    p.add_argument("--dataset",    default="gs512",
                   choices=list(REGISTRIES.keys()))
    p.add_argument("--ckpt_name",  default="checkpoint-best.pth")
    p.add_argument("--max_steps",  type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── ground truth ──────────────────────────────────────────────────────────
    gt = torch.load(args.gt_path, map_location="cpu")
    N_test, n_chan, n_t, H, W = gt.shape
    assert n_t >= args.max_steps + 1
    print(f"GT loaded: {tuple(gt.shape)}  (N={N_test}, steps=0τ..{n_t-1}τ)")

    z0_all = gt[:, :, 0, :, :]    # [N_test, 2, H, W]

    header = f"\n{'Model':<30}" + "".join(f"  {k+1}τ      " for k in range(args.max_steps))
    sep    = "-" * len(header)
    print(header)
    print(sep)

    registry = REGISTRIES[args.dataset]()

    for name, factory in registry:
        ckpt_path = Path(args.ckpt_dir) / name / args.ckpt_name
        if not ckpt_path.exists():
            print(f"  [SKIP] {ckpt_path} not found")
            continue

        model = factory().to(device)
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = _remap_state_dict(name, ckpt["model"])
        model.load_state_dict(state, strict=True)
        model.eval()

        nums = [0.0] * args.max_steps
        dens = [0.0] * args.max_steps

        with torch.no_grad():
            for start in range(0, N_test, args.batch_size):
                end  = min(start + args.batch_size, N_test)
                z0_b = z0_all[start:end].to(device)

                preds = _rollout(model, z0_b, args.max_steps)

                for k, pred_k in enumerate(preds):
                    gt_k = gt[start:end, :, k + 1, :, :].to(device)
                    nums[k] += ((pred_k - gt_k) ** 2).sum().item()
                    dens[k] += (gt_k ** 2).sum().item()

        rel_l2s = [(n / max(d, 1e-8)) ** 0.5 for n, d in zip(nums, dens)]
        row = f"{name:<30}" + "".join(f"  {v:.6f}" for v in rel_l2s)
        print(row)

    print(sep)


if __name__ == "__main__":
    main()
