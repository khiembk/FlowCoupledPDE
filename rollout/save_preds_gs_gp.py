"""
Save MeanFlow-GP rollout predictions for GS qualitative figures.

Run with pdemeanflow conda env.

Usage:
    python rollout/save_preds_gs_gp.py \
        --gt_path  /scratch/user/u.kt348068/PDE_data/gs_rollout_gt_10.pt \
        --ckpt_dir /scratch/user/u.kt348068/ckpt \
        --out_dir  /scratch/user/u.kt348068/qualitative/gs \
        --sample_idxs 0 5 12 \
        --n_steps 10
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "meanflow"))

from models.model_configs import instantiate_coupled_model  # noqa: E402


def _load_gp_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["args"]
    if isinstance(raw, dict):
        raw = argparse.Namespace(**raw)
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


def _disable_checkpoint(model):
    """Disable gradient checkpointing for inference on older PyTorch."""
    for m in model.modules():
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = False


@torch.no_grad()
def _rollout_gs(model, z0, n_steps, net1, net2):
    # z0: [B, 2, H, W]
    src1, src2 = z0[:, 0:1], z0[:, 1:2]
    preds = []
    for _ in range(n_steps):
        p1, p2 = model.sample(src1, src2, net1=net1, net2=net2)
        preds.append(torch.cat([p1, p2], dim=1))  # [B, 2, H, W]
        src1, src2 = p1, p2
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path",     required=True)
    p.add_argument("--ckpt_dir",    required=True)
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--ckpt_name",   default="checkpoint-last.pth")
    p.add_argument("--sample_idxs", type=int, nargs="+", default=[0, 5, 12])
    p.add_argument("--n_steps",     type=int, default=10)
    p.add_argument("--device",      default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gt_all = torch.load(args.gt_path, map_location="cpu")
    # gt_all: [N_test, 2, n_t+1, H, W]
    idxs = args.sample_idxs
    gt_sel = gt_all[idxs]   # [N_sel, 2, n_t+1, H, W]
    torch.save(gt_sel, str(out_dir / "gt.pt"))
    print(f"GT saved: {gt_sel.shape}")

    ckpt_path = Path(args.ckpt_dir) / "gs512_small_gp" / args.ckpt_name
    print(f"Loading gs512_small_gp from {ckpt_path}")
    model = _load_gp_model(str(ckpt_path), device)
    _disable_checkpoint(model)

    z0_sel = gt_all[idxs, :, 0, :, :].to(device)   # [N_sel, 2, H, W]
    preds = _rollout_gs(model, z0_sel, args.n_steps, model.net1, model.net2)
    pred_arr = torch.stack(preds, dim=2).cpu()  # [N_sel, 2, n_steps, H, W]
    torch.save(pred_arr, str(out_dir / "pred_gp.pt"))
    print(f"GP (noema) predictions saved: {pred_arr.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
