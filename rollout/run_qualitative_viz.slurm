#!/usr/bin/env bash
#SBATCH --job-name=qualitative_viz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
unset SLURM_PROCID

GT_GS=/scratch/user/u.kt348068/PDE_data/gs_rollout_gt_10.pt
GT_LV=/scratch/user/u.kt348068/PDE_data/lv_rollout_gt.pt
CKPT=/scratch/user/u.kt348068/ckpt
PRED_GS=/scratch/user/u.kt348068/qualitative/gs
PRED_LV=/scratch/user/u.kt348068/qualitative/lv
FIGS=/scratch/user/u.kt348068/qualitative/figs
SAMPLE_IDXS="0 5 12 42 88"

cd /home/u.kt348068/khiemtt/FlowCoupledPDE

# ── Step 1: GP predictions — flow env (same as existing rollout scripts) ──────
echo "========================================================"
echo "  Saving GS GP predictions"
echo "========================================================"
conda run -n flow python rollout/save_preds_gs_gp.py \
    --gt_path    $GT_GS \
    --ckpt_dir   $CKPT \
    --out_dir    $PRED_GS \
    --sample_idxs $SAMPLE_IDXS \
    --n_steps    10 \
    --device     cuda

echo ""
echo "========================================================"
echo "  Saving LV GP predictions"
echo "========================================================"
conda run -n flow python rollout/save_preds_lv_gp.py \
    --gt_path    $GT_LV \
    --ckpt_dir   $CKPT \
    --out_dir    $PRED_LV \
    --sample_idxs $SAMPLE_IDXS \
    --n_steps    10 \
    --device     cuda

# ── Step 2: Baseline predictions — pde_baselines env ─────────────────────────
echo ""
echo "========================================================"
echo "  Saving GS baseline predictions (FNO / CMWNO / COMPOL)"
echo "========================================================"
conda run -n pde_baselines python rollout/save_preds_gs_baselines.py \
    --gt_path    $GT_GS \
    --ckpt_dir   $CKPT \
    --out_dir    $PRED_GS \
    --sample_idxs $SAMPLE_IDXS \
    --n_steps    10 \
    --device     cuda

echo ""
echo "========================================================"
echo "  Saving LV baseline predictions (FNO / CMWNO / COMPOL)"
echo "========================================================"
conda run -n pde_baselines python rollout/save_preds_lv_baselines.py \
    --gt_path    $GT_LV \
    --ckpt_dir   $CKPT \
    --out_dir    $PRED_LV \
    --sample_idxs $SAMPLE_IDXS \
    --n_steps    10 \
    --device     cuda

# ── Step 3: Plotting — flow env (has matplotlib after install) ────────────────
mkdir -p $FIGS

echo ""
echo "========================================================"
echo "  Plotting GS qualitative comparison"
echo "========================================================"
for SIDX in 0 1 2 3 4; do
    conda run -n flow python rollout/plot_qualitative_gs.py \
        --pred_dir  $PRED_GS \
        --out_dir   $FIGS \
        --sample_idx $SIDX \
        --combined

    conda run -n flow python rollout/plot_qualitative_gs.py \
        --pred_dir  $PRED_GS \
        --out_dir   $FIGS \
        --sample_idx $SIDX \
        --show_error
done

echo ""
echo "========================================================"
echo "  Plotting LV qualitative comparison"
echo "========================================================"
for SIDX in 0 1 2 3 4; do
    conda run -n flow python rollout/plot_qualitative_lv.py \
        --pred_dir  $PRED_LV \
        --out_dir   $FIGS \
        --sample_idx $SIDX
done

echo ""
echo "All figures saved to: $FIGS"
