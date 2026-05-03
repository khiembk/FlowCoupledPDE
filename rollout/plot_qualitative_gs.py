"""
Publication-quality qualitative comparison for Gray-Scott rollout.

Layout: rows = methods (GT | FNO | CMWNO | COMPOL | Ours)
        columns = time steps (1τ | 2τ | 5τ | 10τ)
Each cell shows the u channel (channel 0) as a heatmap.
A second figure optionally shows the absolute error |pred - GT|.

Usage:
    python rollout/plot_qualitative_gs.py \
        --pred_dir  /scratch/user/u.kt348068/qualitative/gs \
        --out_dir   ./figs \
        --sample_idx 0 \
        --show_error

Prerequisites (run first in correct envs):
    python rollout/save_preds_gs_gp.py       --out_dir <pred_dir> ...
    python rollout/save_preds_gs_baselines.py --out_dir <pred_dir> ...
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch


# ── colour/style constants ────────────────────────────────────────────────────
CMAP       = "RdBu_r"
ERR_CMAP   = "Reds"
STEPS      = [1, 2, 5, 10]    # τ indices to display
STEP_LABELS = [r"$1\tau$", r"$2\tau$", r"$5\tau$", r"$10\tau$"]

METHOD_ORDER = [
    ("gt",     "Ground Truth"),
    ("fno",    "FNO"),
    ("cmwno",  "CMWNO"),
    ("compol", "COMPOL-RNN"),
    ("gp",     "Ours (MeanFlow-GP)"),
]

# ─────────────────────────────────────────────────────────────────────────────

def _load_preds(pred_dir: Path, sample_idx: int):
    """Load GT and all model predictions for one sample."""
    gt_all  = torch.load(pred_dir / "gt.pt", map_location="cpu").numpy()
    gt      = gt_all[sample_idx]                  # [2, n_t+1, H, W]

    data = {"gt": gt}
    for tag, _ in METHOD_ORDER[1:]:
        fpath = pred_dir / f"pred_{tag}.pt"
        if fpath.exists():
            arr = torch.load(str(fpath), map_location="cpu").numpy()
            data[tag] = arr[sample_idx]           # [2, n_steps, H, W]
        else:
            print(f"[WARN] {fpath} not found — skipping {tag}")
    return data, gt


def _get_frame(data, tag, step_idx):
    """Return the u-channel frame for method tag at step_idx (1-based)."""
    if tag == "gt":
        return data["gt"][0, step_idx]    # GT step_idx == τ index
    else:
        arr = data.get(tag)
        if arr is None:
            return None
        return arr[0, step_idx - 1]       # pred step_idx-1 == 0-indexed


def _make_prediction_figure(data, out_path, vmin=None, vmax=None):
    """Main prediction grid: rows = methods, columns = timesteps."""
    n_methods = len(METHOD_ORDER)
    n_cols    = len(STEPS)

    # Compute global vmin/vmax from GT across all displayed timesteps
    if vmin is None or vmax is None:
        frames = [data["gt"][0, s] for s in STEPS]
        vmin = min(f.min() for f in frames)
        vmax = max(f.max() for f in frames)
        # Symmetric around 0 if data spans both signs
        if vmin < 0 < vmax:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

    col_w, row_h = 1.5, 1.5
    cbar_w       = 0.18
    fig_w = n_cols * col_w + cbar_w + 0.6
    fig_h = n_methods * row_h + 0.55

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs  = gridspec.GridSpec(
        n_methods, n_cols + 1,
        width_ratios=[1] * n_cols + [cbar_w / col_w],
        left=0.12, right=0.97, top=0.93, bottom=0.03,
        wspace=0.04, hspace=0.04,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
            for r in range(n_methods)]
    cax  = fig.add_subplot(gs[:, n_cols])

    im_ref = None
    for r, (tag, label) in enumerate(METHOD_ORDER):
        for c, step in enumerate(STEPS):
            ax  = axes[r][c]
            frame = _get_frame(data, tag, step)
            if frame is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(frame, cmap=CMAP, vmin=vmin, vmax=vmax,
                           origin="upper", interpolation="nearest",
                           aspect="equal")
            if im_ref is None:
                im_ref = im
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            # Column header (top row only)
            if r == 0:
                ax.set_title(STEP_LABELS[c], fontsize=9, pad=3)
            # Row label (first column only)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, labelpad=4, rotation=90, va="center")

    # Colorbar
    cb = fig.colorbar(im_ref, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)

    fig.suptitle("Gray-Scott — u field predictions", fontsize=10, y=0.97)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _make_error_figure(data, out_path):
    """Error grid: rows = methods (skip GT), columns = timesteps."""
    methods   = [(t, l) for t, l in METHOD_ORDER if t != "gt"]
    n_methods = len(methods)
    n_cols    = len(STEPS)

    # Global error max for shared scale
    errs = []
    for tag, _ in methods:
        if tag in data:
            for s in STEPS:
                pred_frame = _get_frame(data, tag, s)
                gt_frame   = data["gt"][0, s]
                if pred_frame is not None:
                    errs.append(np.abs(pred_frame - gt_frame).max())
    err_max = max(errs) if errs else 1.0

    col_w, row_h = 1.5, 1.5
    cbar_w       = 0.18
    fig_w = n_cols * col_w + cbar_w + 0.6
    fig_h = n_methods * row_h + 0.55

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs  = gridspec.GridSpec(
        n_methods, n_cols + 1,
        width_ratios=[1] * n_cols + [cbar_w / col_w],
        left=0.15, right=0.97, top=0.93, bottom=0.03,
        wspace=0.04, hspace=0.04,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
            for r in range(n_methods)]
    cax  = fig.add_subplot(gs[:, n_cols])

    im_ref = None
    for r, (tag, label) in enumerate(methods):
        for c, step in enumerate(STEPS):
            ax = axes[r][c]
            pred_frame = _get_frame(data, tag, step)
            gt_frame   = data["gt"][0, step]
            if pred_frame is None:
                ax.set_visible(False)
                continue
            err = np.abs(pred_frame - gt_frame)
            im  = ax.imshow(err, cmap=ERR_CMAP, vmin=0, vmax=err_max,
                            origin="upper", interpolation="nearest",
                            aspect="equal")
            if im_ref is None:
                im_ref = im
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            if r == 0:
                ax.set_title(STEP_LABELS[c], fontsize=9, pad=3)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, labelpad=4, rotation=90, va="center")

    cb = fig.colorbar(im_ref, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    cb.set_label("|error|", fontsize=7)

    fig.suptitle("Gray-Scott — absolute error |pred − GT| (u field)", fontsize=10, y=0.97)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _make_combined_figure(data, out_path, vmin=None, vmax=None):
    """
    Combined figure: top block = predictions, bottom block = errors.
    Rows grouped by method; within each group: prediction row, then error row.
    """
    methods_with_gt = METHOD_ORDER                # GT + 4 models
    methods_no_gt   = [(t, l) for t, l in METHOD_ORDER if t != "gt"]
    n_cols = len(STEPS)

    # vmin/vmax for predictions
    if vmin is None or vmax is None:
        frames = [data["gt"][0, s] for s in STEPS]
        vmin = min(f.min() for f in frames)
        vmax = max(f.max() for f in frames)
        if vmin < 0 < vmax:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

    # error max
    errs = []
    for tag, _ in methods_no_gt:
        if tag in data:
            for s in STEPS:
                pf = _get_frame(data, tag, s)
                gf = data["gt"][0, s]
                if pf is not None:
                    errs.append(np.abs(pf - gf).max())
    err_max = max(errs) if errs else 1.0

    # Total rows: GT row + (prediction+error) for each model
    n_rows = 1 + 2 * len(methods_no_gt)
    col_w, row_h = 1.5, 1.4
    cbar_w       = 0.18
    fig_w = n_cols * col_w + 2 * (cbar_w + 0.05) + 0.7
    fig_h = n_rows * row_h + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # Use two colorbars: one for predictions, one for errors
    gs = gridspec.GridSpec(
        n_rows, n_cols + 2,
        width_ratios=[1] * n_cols + [cbar_w / col_w, cbar_w / col_w],
        left=0.13, right=0.97, top=0.95, bottom=0.02,
        wspace=0.04, hspace=0.04,
    )

    pred_cax = fig.add_subplot(gs[:, n_cols])
    err_cax  = fig.add_subplot(gs[:, n_cols + 1])
    pred_im_ref = None
    err_im_ref  = None

    row = 0
    # ── GT row ──────────────────────────────────────────────────────────────
    for c, step in enumerate(STEPS):
        ax = fig.add_subplot(gs[row, c])
        frame = data["gt"][0, step]
        im = ax.imshow(frame, cmap=CMAP, vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="nearest", aspect="equal")
        if pred_im_ref is None:
            pred_im_ref = im
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        if c == 0:
            ax.set_ylabel("GT", fontsize=8, labelpad=4, rotation=90, va="center",
                          fontweight="bold")
        if row == 0:
            ax.set_title(STEP_LABELS[c], fontsize=9, pad=3)
    row += 1

    # ── per-model pairs ──────────────────────────────────────────────────────
    for tag, label in methods_no_gt:
        # prediction row
        for c, step in enumerate(STEPS):
            ax = fig.add_subplot(gs[row, c])
            pred_frame = _get_frame(data, tag, step)
            if pred_frame is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(pred_frame, cmap=CMAP, vmin=vmin, vmax=vmax,
                           origin="upper", interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if c == 0:
                ax.set_ylabel(label, fontsize=7, labelpad=4, rotation=90, va="center")
        row += 1

        # error row
        for c, step in enumerate(STEPS):
            ax = fig.add_subplot(gs[row, c])
            pred_frame = _get_frame(data, tag, step)
            gt_frame   = data["gt"][0, step]
            if pred_frame is None:
                ax.set_visible(False)
                continue
            err = np.abs(pred_frame - gt_frame)
            im = ax.imshow(err, cmap=ERR_CMAP, vmin=0, vmax=err_max,
                           origin="upper", interpolation="nearest", aspect="equal")
            if err_im_ref is None:
                err_im_ref = im
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if c == 0:
                ax.set_ylabel(f"{label}\n|err|", fontsize=6, labelpad=4,
                              rotation=90, va="center", color="gray")
        row += 1

    # colorbars
    if pred_im_ref is not None:
        cb1 = fig.colorbar(pred_im_ref, cax=pred_cax)
        cb1.ax.tick_params(labelsize=6)
        cb1.outline.set_linewidth(0.5)
    if err_im_ref is not None:
        cb2 = fig.colorbar(err_im_ref, cax=err_cax)
        cb2.ax.tick_params(labelsize=6)
        cb2.outline.set_linewidth(0.5)
        cb2.set_label("|err|", fontsize=6)

    fig.suptitle("Gray-Scott — u field: predictions and errors", fontsize=10, y=0.97)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir",   required=True,
                   help="Directory with gt.npy, pred_gp.npy, pred_fno.npy, ...")
    p.add_argument("--out_dir",    default="./figs")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Which sample in pred_dir arrays to visualise (0-indexed "
                        "into the N_sel samples you chose when saving)")
    p.add_argument("--show_error", action="store_true",
                   help="Also produce a separate error-magnitude figure")
    p.add_argument("--combined",   action="store_true",
                   help="Produce a combined pred+error figure (overrides others)")
    p.add_argument("--channel",    type=int, default=0,
                   help="Which process channel to display (0=u, 1=v)")
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, gt = _load_preds(pred_dir, args.sample_idx)

    # channel selection
    if args.channel != 0:
        for k in data:
            data[k] = data[k][[args.channel]]     # keep only requested channel
        gt = data["gt"]

    tag = f"s{args.sample_idx}_ch{args.channel}"

    if args.combined:
        _make_combined_figure(data, out_dir / f"gs_combined_{tag}.pdf")
        _make_combined_figure(data, out_dir / f"gs_combined_{tag}.png")
    else:
        _make_prediction_figure(data, out_dir / f"gs_pred_{tag}.pdf")
        _make_prediction_figure(data, out_dir / f"gs_pred_{tag}.png")
        if args.show_error:
            _make_error_figure(data, out_dir / f"gs_error_{tag}.pdf")
            _make_error_figure(data, out_dir / f"gs_error_{tag}.png")


if __name__ == "__main__":
    main()
