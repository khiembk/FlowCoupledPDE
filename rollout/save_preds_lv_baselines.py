"""
Save FNO1d / CMWNO1d / COMPOL1d-RNN rollout predictions for LV qualitative figures.

Run with pde_baselines conda env.

Usage:
    python rollout/save_preds_lv_baselines.py \
        --gt_path  /scratch/user/u.kt348068/PDE_data/lv_rollout_gt.pt \
        --ckpt_dir /scratch/user/u.kt348068/ckpt \
        --out_dir  /scratch/user/u.kt348068/qualitative/lv \
        --sample_idxs 0 5 12 \
        --n_steps 10
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "baselines"))

from models import FNO1d, CMWNO1d, COMPOL1d  # noqa: E402


def _build_registry():
    n = 2

    def fno():
        return FNO1d(modes=12, width=64, in_channels=n, out_channels=n, n_layers=4)

    def cmwno():
        return CMWNO1d(in_channels=1, out_channels=1, n_proc=n,
                       width=32, n_layers=4, k=2)

    def compol_rnn():
        return COMPOL1d(in_channels=1, out_channels=1, n_proc=n,
                        modes=12, width=32, n_layers=4, aggr_type="rnn")

    return [
        ("lv512_fno1d",        "fno",    fno),
        ("lv512_cmwno1d",      "cmwno",  cmwno),
        ("lv512_compol1d_rnn", "compol", compol_rnn),
    ]


def _remap_cmwno(state_dict):
    mapping = {"net1.": "nets.0.", "net2.": "nets.1."}
    new = {}
    for k, v in state_dict.items():
        for old, rep in mapping.items():
            if k.startswith(old):
                k = rep + k[len(old):]
                break
        new[k] = v
    return new


@torch.no_grad()
def _rollout(model: nn.Module, z0: torch.Tensor, n_steps: int):
    src = z0
    preds = []
    for _ in range(n_steps):
        preds.append(model(src))
        src = preds[-1]
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path",      required=True)
    p.add_argument("--ckpt_dir",     required=True)
    p.add_argument("--out_dir",      required=True)
    p.add_argument("--ckpt_name",    default="checkpoint-best.pth")
    p.add_argument("--sample_idxs",  type=int, nargs="+", default=[0, 5, 12])
    p.add_argument("--n_steps",      type=int, default=10)
    p.add_argument("--device",       default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gt_all = torch.load(args.gt_path, map_location="cpu")
    idxs = args.sample_idxs
    z0_sel = gt_all[idxs, :, 0, :].to(device)   # [N_sel, 2, 256]

    for ckpt_name, tag, factory in _build_registry():
        ckpt_path = Path(args.ckpt_dir) / ckpt_name / args.ckpt_name
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt_path} not found")
            continue
        model = factory().to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        if "cmwno" in ckpt_name:
            state = _remap_cmwno(state)
        model.load_state_dict(state, strict=True)
        model.eval()

        preds = _rollout(model, z0_sel, args.n_steps)
        pred_arr = torch.stack(preds, dim=2).cpu()  # [N_sel, 2, n_steps, 256]
        out_path = out_dir / f"pred_{tag}.pt"
        torch.save(pred_arr, str(out_path))
        print(f"{tag} predictions saved: {pred_arr.shape}  →  {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
