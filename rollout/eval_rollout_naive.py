"""
Evaluate NaiveMeanFlow models on multi-step GS rollout.

NaiveMeanFlow: each process is predicted independently (no cross-process
coupling). Uses the same MeanFlow sampling interface as CoupledFlow but
takes a list of per-process tensors.

Usage:
    python rollout/eval_rollout_naive.py \
        --gt_path  /scratch/user/u.kt348068/PDE_data/gs_rollout_gt.pt \
        --ckpt_dir /scratch/user/u.kt348068/ckpt \
        --models   naive_gs512 \
        --device   cuda
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
# baselines/ for NaiveMeanFlow; meanflow/ not needed here
sys.path.insert(0, str(REPO / "baselines"))
sys.path.insert(0, str(REPO / "meanflow"))

from models.naive_flow import build_naive_flow_model  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

DATASET_DIM = {
    "grayscott": "2d", "multiphase": "2d", "thm": "2d",
    "lv": "1d", "bz": "1d",
}


def _load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["args"]
    if isinstance(raw, dict):
        raw = argparse.Namespace(**raw)

    # fill in fields that older checkpoints may not have saved
    _defaults = dict(dropout=0.2, ema_decay=0.9999,
                     ema_decays=[0.99995, 0.9996],
                     norm_p=0.75, norm_eps=1e-3,
                     tr_sampler="v1", ratio=0.75,
                     P_mean_t=-0.6, P_std_t=1.6,
                     P_mean_r=-4.0, P_std_r=1.6)
    for k, v in _defaults.items():
        if not hasattr(raw, k):
            setattr(raw, k, v)

    dataset = getattr(raw, "dataset", "grayscott")
    dim     = DATASET_DIM.get(dataset, "2d")
    n_proc  = raw.n_proc

    model = build_naive_flow_model(n_proc=n_proc, dim=dim, args=raw).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def _ema_variants(model):
    """Return list of (label, nets_list) for all EMA copies."""
    n = model.n_proc
    noema = [model._modules[f"net{i+1}"]     for i in range(n)]
    ema   = [model._modules[f"net{i+1}_ema"] for i in range(n)]
    variants = [("noema", noema), ("ema", ema)]
    for j, decay in enumerate(model.ema_decays):
        nets = [model._modules[f"net{i+1}_ema{j+1}"] for i in range(n)]
        variants.append((f"ema{decay}", nets))
    return variants


@torch.no_grad()
def _rollout(model, z0, n_steps, nets):
    """
    z0   : [B, n_proc, H, W]
    nets : list of n_proc networks
    returns list of n_steps tensors [B, n_proc, *spatial]
    """
    n_proc = model.n_proc
    sources = [z0[:, i:i+1] for i in range(n_proc)]
    preds = []
    for _ in range(n_steps):
        pred_list = model.sample(sources, nets=nets)
        out = torch.cat(pred_list, dim=1)
        preds.append(out)
        sources = pred_list
    return preds


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path",    required=True)
    p.add_argument("--ckpt_dir",   required=True)
    p.add_argument("--models",     nargs="+", default=["naive_gs512"])
    p.add_argument("--ckpt_name",  default="checkpoint-last.pth")
    p.add_argument("--max_steps",  type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gt = torch.load(args.gt_path, map_location="cpu")
    N_test, n_chan, n_t, H, W = gt.shape
    assert n_t >= args.max_steps + 1
    print(f"GT loaded: {tuple(gt.shape)}  (N={N_test}, steps=0τ..{n_t-1}τ)")

    z0_all = gt[:, :, 0, :, :]   # [N_test, n_proc, H, W]

    header = f"\n{'Model/Variant':<35}" + "".join(f"  {k+1}τ      " for k in range(args.max_steps))
    sep    = "-" * len(header)

    for model_name in args.models:
        ckpt_path = Path(args.ckpt_dir) / model_name / args.ckpt_name
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt_path} not found")
            continue
        print(f"\nLoading {model_name} …")
        model = _load_model(str(ckpt_path), device)

        print(header)
        print(sep)

        for label, nets in _ema_variants(model):
            nums = [0.0] * args.max_steps
            dens = [0.0] * args.max_steps

            for start in range(0, N_test, args.batch_size):
                end  = min(start + args.batch_size, N_test)
                z0_b = z0_all[start:end].to(device)

                preds = _rollout(model, z0_b, args.max_steps, nets)

                for k, pred_k in enumerate(preds):
                    gt_k = gt[start:end, :, k + 1, :, :].to(device)
                    nums[k] += ((pred_k - gt_k) ** 2).sum().item()
                    dens[k] += (gt_k ** 2).sum().item()

            rel_l2s = [(n / max(d, 1e-8)) ** 0.5 for n, d in zip(nums, dens)]
            row = f"{model_name}/{label:<25}" + "".join(f"  {v:.6f}" for v in rel_l2s)
            print(row)

        print(sep)


if __name__ == "__main__":
    main()
