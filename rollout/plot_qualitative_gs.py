"""
Publication-quality qualitative comparison for Gray-Scott rollout.

Main figure layout (--paper, default):
  Row 0       : GT fields at 1τ / 2τ / 5τ / 10τ
  Rows 1-N    : |pred - GT| error maps, one row per method,
                each row independently colour-normalised so differences
                between methods are clearly visible.

Additional figures:
  --pred_only : raw prediction grid (all on shared GT colorscale)
  --combined  : prediction rows + error rows interleaved

Usage:
    python rollout/plot_qualitative_gs.py \
        --pred_dir  /scratch/user/u.kt348068/qualitative/gs \
        --out_dir   ./figs \
        --sample_idx 0
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ── constants ─────────────────────────────────────────────────────────────────
PRED_CMAP  = "RdBu_r"
ERR_CMAP   = "Reds"
STEPS       = [1, 2, 5, 10]
STEP_LABELS = [r"$1\tau$", r"$2\tau$", r"$5\tau$", r"$10\tau$"]

METHOD_ORDER = [
    ("gt",     "Ground Truth"),
    ("fno",    "FNO"),
    ("cmwno",  "CMWNO"),
    ("compol", "COMPOL"),
    ("gp",     "Ours"),
]


# ── data loading ──────────────────────────────────────────────────────────────

def _load_preds(pred_dir: Path, sample_idx: int):
    gt_all = torch.load(pred_dir / "gt.pt", map_location="cpu").numpy()
    gt     = gt_all[sample_idx]   # [2, n_t+1, H, W]

    data = {"gt": gt}
    for tag, _ in METHOD_ORDER[1:]:
        fpath = pred_dir / f"pred_{tag}.pt"
        if fpath.exists():
            arr = torch.load(str(fpath), map_location="cpu").numpy()
            data[tag] = arr[sample_idx]   # [2, n_steps, H, W]
        else:
            print(f"[WARN] {fpath} not found — skipping {tag}")
    return data


def _get_frame(data, tag, step, channel=0):
    """Return one spatial frame. step is 1-based τ index."""
    if tag == "gt":
        return data["gt"][channel, step]
    arr = data.get(tag)
    if arr is None:
        return None
    return arr[channel, step - 1]


# ── figure: GT row + per-row-normalised error maps (paper figure) ─────────────

def _make_paper_figure(data, out_path, channel=0):
    """
    Top row: GT fields.
    One row per model: |pred - GT|, independently normalised per row.
    This makes inter-method differences clearly visible.
    """
    methods = [(t, l) for t, l in METHOD_ORDER if t != "gt"]
    n_rows  = 1 + len(methods)
    n_cols  = len(STEPS)

    col_w, row_h = 1.6, 1.5
    cbar_w = 0.15
    # one shared pred colorbar + one colorbar per method error row
    n_cbars = 1 + len(methods)
    fig_w = n_cols * col_w + n_cbars * (cbar_w + 0.04) + 0.7
    fig_h = n_rows * row_h + 0.55

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # gridspec: data cols + one cbar col per row group
    gs = gridspec.GridSpec(
        n_rows, n_cols + n_cbars,
        width_ratios=[1] * n_cols + [cbar_w / col_w] * n_cbars,
        left=0.10, right=0.98, top=0.93, bottom=0.03,
        wspace=0.04, hspace=0.06,
    )

    # global GT vmin/vmax for top row
    gt_frames = [_get_frame(data, "gt", s, channel) for s in STEPS]
    vmin = min(f.min() for f in gt_frames)
    vmax = max(f.max() for f in gt_frames)
    if vmin < 0 < vmax:
        a = max(abs(vmin), abs(vmax))
        vmin, vmax = -a, a

    # ── GT row ────────────────────────────────────────────────────────────────
    gt_im = None
    for c, (step, slabel) in enumerate(zip(STEPS, STEP_LABELS)):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(_get_frame(data, "gt", step, channel),
                       cmap=PRED_CMAP, vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="nearest", aspect="equal")
        if gt_im is None:
            gt_im = im
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.5)
        ax.set_title(slabel, fontsize=9, pad=3)
        if c == 0:
            ax.set_ylabel("GT", fontsize=8, labelpad=4, fontweight="bold")

    cb0 = fig.colorbar(gt_im, cax=fig.add_subplot(gs[0, n_cols]), orientation="vertical")
    cb0.ax.tick_params(labelsize=6); cb0.outline.set_linewidth(0.5)

    # ── per-method error rows ─────────────────────────────────────────────────
    for r, (tag, label) in enumerate(methods):
        row = r + 1
        if tag not in data:
            continue

        # per-row error max (independent normalisation)
        err_frames = []
        for step in STEPS:
            pf = _get_frame(data, tag, step, channel)
            gf = _get_frame(data, "gt", step, channel)
            if pf is not None:
                err_frames.append(np.abs(pf - gf))
        if not err_frames:
            continue
        row_emax = max(f.max() for f in err_frames)

        err_im = None
        for c, (step, err_f) in enumerate(zip(STEPS, err_frames)):
            ax = fig.add_subplot(gs[row, c])
            im = ax.imshow(err_f, cmap=ERR_CMAP, vmin=0, vmax=row_emax,
                           origin="upper", interpolation="nearest", aspect="equal")
            if err_im is None:
                err_im = im
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_linewidth(0.5)
            if c == 0:
                ax.set_ylabel(f"{label}\n|err|", fontsize=7.5, labelpad=4)

        cax_col = n_cols + 1 + r
        cb = fig.colorbar(err_im, cax=fig.add_subplot(gs[row, cax_col]),
                          orientation="vertical")
        cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.5)
        cb.formatter.set_powerlimits((-2, 2)); cb.update_ticks()

    ch_name = "u" if channel == 0 else "v"
    fig.suptitle(f"Gray-Scott — {ch_name} field  |  GT (top) and per-method error (rows)",
                 fontsize=9, y=0.97)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── figure: raw predictions on shared colorscale ─────────────────────────────

def _make_pred_figure(data, out_path, channel=0):
    n_methods = len(METHOD_ORDER)
    n_cols    = len(STEPS)

    gt_frames = [_get_frame(data, "gt", s, channel) for s in STEPS]
    vmin = min(f.min() for f in gt_frames)
    vmax = max(f.max() for f in gt_frames)
    if vmin < 0 < vmax:
        a = max(abs(vmin), abs(vmax)); vmin, vmax = -a, a

    col_w, row_h = 1.5, 1.5
    fig_w = n_cols * col_w + 0.35 + 0.6
    fig_h = n_methods * row_h + 0.55

    fig, axes = plt.subplots(n_methods, n_cols,
                             figsize=(fig_w, fig_h), dpi=150)
    im_ref = None
    for r, (tag, label) in enumerate(METHOD_ORDER):
        for c, (step, slabel) in enumerate(zip(STEPS, STEP_LABELS)):
            ax = axes[r][c]
            frame = _get_frame(data, tag, step, channel)
            if frame is None:
                ax.set_visible(False); continue
            im = ax.imshow(frame, cmap=PRED_CMAP, vmin=vmin, vmax=vmax,
                           origin="upper", interpolation="nearest", aspect="equal")
            if im_ref is None: im_ref = im
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_linewidth(0.5)
            if r == 0: ax.set_title(slabel, fontsize=9, pad=3)
            if c == 0: ax.set_ylabel(label, fontsize=8, labelpad=4)

    fig.subplots_adjust(right=0.88, wspace=0.04, hspace=0.04)
    cax = fig.add_axes([0.90, 0.05, 0.02, 0.88])
    cb = fig.colorbar(im_ref, cax=cax); cb.ax.tick_params(labelsize=6)

    ch_name = "u" if channel == 0 else "v"
    fig.suptitle(f"Gray-Scott — {ch_name} field predictions", fontsize=10)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir",   required=True)
    p.add_argument("--out_dir",    default="./figs")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--channel",    type=int, default=0, help="0=u, 1=v")
    p.add_argument("--pred_only",  action="store_true",
                   help="Raw prediction grid on shared colorscale")
    # kept for backward compat but paper figure is always generated
    p.add_argument("--combined",   action="store_true")
    p.add_argument("--show_error", action="store_true")
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_preds(pred_dir, args.sample_idx)
    tag  = f"s{args.sample_idx}_ch{args.channel}"

    # always produce the paper figure (GT + per-row error)
    for ext in ("pdf", "png"):
        _make_paper_figure(data, out_dir / f"gs_paper_{tag}.{ext}", args.channel)

    if args.pred_only or args.combined:
        for ext in ("pdf", "png"):
            _make_pred_figure(data, out_dir / f"gs_pred_{tag}.{ext}", args.channel)


if __name__ == "__main__":
    main()
