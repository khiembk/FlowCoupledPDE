"""
Publication-quality qualitative comparison for Lotka-Volterra rollout.

Layout: 2 rows (u species, v species) × 4 columns (1τ, 2τ, 5τ, 10τ).
Each panel shows line plots for GT and all methods over the spatial domain.

Usage:
    python rollout/plot_qualitative_lv.py \
        --pred_dir  /scratch/user/u.kt348068/qualitative/lv \
        --out_dir   ./figs \
        --sample_idx 0

Prerequisites (run first in correct envs):
    python rollout/save_preds_lv_gp.py        --out_dir <pred_dir> ...
    python rollout/save_preds_lv_baselines.py  --out_dir <pred_dir> ...
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ── colour/style constants ────────────────────────────────────────────────────
STEPS       = [1, 2, 5, 10]
STEP_LABELS = [r"$1\tau$", r"$2\tau$", r"$5\tau$", r"$10\tau$"]
CHANNEL_NAMES = ["u (prey)", "v (predator)"]

METHODS = [
    # (tag, label, color, linestyle, zorder, linewidth)
    ("gt",     "GT",            "black",   "-",  5,  2.0),
    ("fno",    "FNO",           "#E07B54",  "--", 3, 1.3),
    ("cmwno",  "CMWNO",         "#5C85D6",  "-.", 3, 1.3),
    ("compol", "COMPOL-RNN",    "#77B77D",  ":",  3, 1.3),
    ("gp",     "Ours",          "#B83232",  "-",  4, 1.8),
]


def _load_preds(pred_dir: Path, sample_idx: int):
    gt_all = torch.load(pred_dir / "gt.pt", map_location="cpu").numpy()
    gt     = gt_all[sample_idx]            # [2, n_t+1, 256]

    preds = {"gt": gt}
    for tag, *_ in METHODS[1:]:
        fpath = pred_dir / f"pred_{tag}.pt"
        if fpath.exists():
            arr = torch.load(str(fpath), map_location="cpu").numpy()
            preds[tag] = arr[sample_idx]   # [2, n_steps, 256]
        else:
            print(f"[WARN] {fpath} not found — skipping {tag}")
    return preds


def _get_frame(preds, tag, step):
    """Return channel-split array at a given τ step."""
    if tag == "gt":
        return preds["gt"][:, step, :]     # [2, 256]
    arr = preds.get(tag)
    if arr is None:
        return None
    return arr[:, step - 1, :]             # [2, 256]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir",   required=True)
    p.add_argument("--out_dir",    default="./figs")
    p.add_argument("--sample_idx", type=int, default=0)
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = _load_preds(pred_dir, args.sample_idx)

    n_chan = 2
    n_cols = len(STEPS)
    x = np.linspace(0, 1, 256)

    fig, axes = plt.subplots(
        n_chan, n_cols,
        figsize=(n_cols * 2.8, n_chan * 2.2),
        dpi=150,
        sharex=True,
        constrained_layout=True,
    )

    for ci in range(n_chan):           # channel (row)
        for si, step in enumerate(STEPS):   # timestep (column)
            ax = axes[ci][si]

            for tag, label, color, ls, zo, lw in METHODS:
                frame = _get_frame(preds, tag, step)
                if frame is None:
                    continue
                y = frame[ci]    # [256]
                ax.plot(x, y, color=color, ls=ls, lw=lw, zorder=zo,
                        label=label if (ci == 0 and si == 0) else None)

            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

            if ci == 0:
                ax.set_title(STEP_LABELS[si], fontsize=9)
            if si == 0:
                ax.set_ylabel(CHANNEL_NAMES[ci], fontsize=8)
            if ci == n_chan - 1:
                ax.set_xlabel("spatial coord.", fontsize=7)

    # single legend in the top-left panel
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.04),
               ncol=len(METHODS),
               fontsize=7.5,
               frameon=False)

    fig.suptitle("Lotka-Volterra — rollout predictions", fontsize=10, y=1.06)

    tag = f"s{args.sample_idx}"
    for ext in ("pdf", "png"):
        out_path = out_dir / f"lv_pred_{tag}.{ext}"
        fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
