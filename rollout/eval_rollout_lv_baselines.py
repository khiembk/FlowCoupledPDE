"""
Evaluate baseline neural operator models on multi-step LV rollout.

Ground truth: lv_rollout_gt.pt  shape [N_test, 2, 11, 256]
  dim 2: [0tau, 1tau, ..., 10tau]

Models: FNO1d, UFNO1d, COMPOL1d_rnn, COMPOL1d_atn, DiffusionPDE (1D).
DiffusionPDE is trained with sequences reshaped to 16x16; rollout reshapes
input/output accordingly.

Usage:
    python rollout/eval_rollout_lv_baselines.py \
        --gt_path  /scratch/user/u.kt348068/PDE_data/lv_rollout_gt.pt \
        --ckpt_dir /scratch/user/u.kt348068/ckpt \
        --dataset  lv512 \
        --device   cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "baselines"))

from models import FNO1d, UFNO1d, COMPOL1d, DiffusionPDE  # noqa: E402

_1D_RES     = 16
_1D_SEQ_LEN = 256


def _lv512_registry():
    n = 2

    def fno():
        return FNO1d(modes=12, width=64, in_channels=n, out_channels=n, n_layers=4)

    def ufno():
        return UFNO1d(modes=12, width=64, in_channels=n, out_channels=n, n_layers=4)

    def compol_rnn():
        return COMPOL1d(in_channels=1, out_channels=1, n_proc=n,
                        modes=12, width=32, n_layers=4, aggr_type="rnn")

    def compol_atn():
        return COMPOL1d(in_channels=1, out_channels=1, n_proc=n,
                        modes=12, width=32, n_layers=4, aggr_type="atn")

    def diffusion_pde():
        return DiffusionPDE(n_proc=n, img_resolution=_1D_RES,
                            model_channels=64, num_steps=20)

    # (ckpt_subdir, factory, is_1d_reshape)
    return [
        ("lv512_fno1d",         fno,          False),
        ("lv512_ufno1d",        ufno,         False),
        ("lv512_compol1d_rnn",  compol_rnn,   False),
        ("lv512_compol1d_atn",  compol_atn,   False),
        ("lv512_diffusion_pde", diffusion_pde, True),
    ]


REGISTRIES = {
    "lv512": _lv512_registry,
}


@torch.no_grad()
def _rollout(model: nn.Module, z0: torch.Tensor, n_steps: int, is_1d: bool):
    # z0: [B, 2, 256]
    src = z0
    preds = []
    for _ in range(n_steps):
        if is_1d:
            inp  = src.view(src.shape[0], src.shape[1], _1D_RES, _1D_RES)
            out  = model(inp)
            pred = out.view(out.shape[0], out.shape[1], _1D_SEQ_LEN)
        else:
            pred = model(src)
        preds.append(pred)
        src = pred
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path",    required=True)
    p.add_argument("--ckpt_dir",   required=True)
    p.add_argument("--dataset",    default="lv512", choices=list(REGISTRIES.keys()))
    p.add_argument("--ckpt_name",  default="checkpoint-best.pth")
    p.add_argument("--max_steps",  type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gt = torch.load(args.gt_path, map_location="cpu")
    N_test, n_chan, n_t, L = gt.shape
    assert n_t >= args.max_steps + 1, \
        f"GT has {n_t} time steps but max_steps={args.max_steps}"
    print(f"GT loaded: {tuple(gt.shape)}  (N={N_test}, L={L}, steps=0τ..{n_t-1}τ)")

    z0_all = gt[:, :, 0, :]   # [N_test, 2, 256]

    header = f"\n{'Model':<35}" + "".join(f"  {k+1}τ      " for k in range(args.max_steps))
    sep    = "-" * len(header)
    print(header)
    print(sep)

    registry = REGISTRIES[args.dataset]()

    for name, factory, is_1d in registry:
        ckpt_path = Path(args.ckpt_dir) / name / args.ckpt_name
        if not ckpt_path.exists():
            print(f"  [SKIP] {ckpt_path} not found")
            continue

        model = factory().to(device)
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        nums = [0.0] * args.max_steps
        dens = [0.0] * args.max_steps

        with torch.no_grad():
            for start in range(0, N_test, args.batch_size):
                end  = min(start + args.batch_size, N_test)
                z0_b = z0_all[start:end].to(device)

                preds = _rollout(model, z0_b, args.max_steps, is_1d)

                for k, pred_k in enumerate(preds):
                    gt_k = gt[start:end, :, k + 1, :].to(device)
                    nums[k] += ((pred_k - gt_k) ** 2).sum().item()
                    dens[k] += (gt_k ** 2).sum().item()

        rel_l2s = [(n / max(d, 1e-8)) ** 0.5 for n, d in zip(nums, dens)]
        row = f"{name:<35}" + "".join(f"  {v:.6f}" for v in rel_l2s)
        print(row)

    print(sep)


if __name__ == "__main__":
    main()
