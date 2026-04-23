#!/usr/bin/env bash
#SBATCH --job-name=gs_lv_rollout10
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh

GT_GS=/scratch/user/u.kt348068/PDE_data/gs_rollout_gt_10.pt
GT_LV=/scratch/user/u.kt348068/PDE_data/lv_rollout_gt.pt
DATA_GS=/scratch/user/u.kt348068/PDE_data/gs_512.pt
CKPT=/scratch/user/u.kt348068/ckpt

cd /home/u.kt348068/khiemtt/FlowCoupledPDE

# ── Step 1: generate 10-step GS rollout GT (pdemeanflow env) ──────────────────
conda activate pdemeanflow

echo "========================================================"
echo "  Generating GS rollout GT (10 steps)"
echo "========================================================"
python data_generator/gen_gs_rollout_gt.py \
    --data_path   $DATA_GS \
    --out_path    $GT_GS \
    --n_steps     10 \
    --train_ratio 0.6305 \
    --val_ratio   0.1232 \
    --workers     8

# ── Step 2: GS MeanFlow-GP (pdemeanflow env) ──────────────────────────────────
echo ""
echo "========================================================"
echo "  GS MeanFlow-GP rollout (10 steps)"
echo "========================================================"
python rollout/eval_rollout_meanflow.py \
    --gt_path    $GT_GS \
    --ckpt_dir   $CKPT \
    --models     gs_gp_512 \
    --max_steps  10 \
    --batch_size 32 \
    --device     cuda

# ── Step 3: GS NaiveMeanFlow (pdemeanflow env) ────────────────────────────────
echo ""
echo "========================================================"
echo "  GS NaiveMeanFlow rollout (10 steps)"
echo "========================================================"
python rollout/eval_rollout_naive.py \
    --gt_path    $GT_GS \
    --ckpt_dir   $CKPT \
    --models     naive_gs512 \
    --max_steps  10 \
    --batch_size 32 \
    --device     cuda

# ── Step 4: GS baselines + LV baselines (pde_baselines env) ──────────────────
conda activate pde_baselines

echo ""
echo "========================================================"
echo "  GS baselines rollout (10 steps, no DiffusionPDE)"
echo "========================================================"
python rollout/eval_rollout_baselines.py \
    --gt_path    $GT_GS \
    --ckpt_dir   $CKPT \
    --dataset    gs512_nodiff \
    --ckpt_name  checkpoint-best.pth \
    --max_steps  10 \
    --batch_size 32 \
    --device     cuda

echo ""
echo "========================================================"
echo "  LV baselines rollout (10 steps, with CMWNO1d)"
echo "========================================================"
python rollout/eval_rollout_lv_baselines.py \
    --gt_path    $GT_LV \
    --ckpt_dir   $CKPT \
    --dataset    lv512 \
    --ckpt_name  checkpoint-best.pth \
    --max_steps  10 \
    --batch_size 32 \
    --device     cuda
