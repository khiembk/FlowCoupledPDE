#!/usr/bin/env bash
#SBATCH --job-name=gs_rollout_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh

GT=/scratch/user/u.kt348068/PDE_data/gs_rollout_gt.pt
CKPT=/scratch/user/u.kt348068/ckpt

cd /home/u.kt348068/khiemtt/FlowCoupledPDE

# ── MeanFlow (GP + naive) ──────────────────────────────────────────────────
conda activate pdemeanflow

echo "========================================================"
echo "  MeanFlow rollout evaluation"
echo "========================================================"
python rollout/eval_rollout_meanflow.py \
    --gt_path   $GT \
    --ckpt_dir  $CKPT \
    --models    gs_gp_512 naive_gs512 \
    --max_steps 5 \
    --batch_size 32 \
    --device    cuda

# ── Baselines ─────────────────────────────────────────────────────────────
conda activate pde_baselines

echo ""
echo "========================================================"
echo "  Baseline rollout evaluation"
echo "========================================================"
python rollout/eval_rollout_baselines.py \
    --gt_path   $GT \
    --ckpt_dir  $CKPT \
    --dataset   gs512 \
    --max_steps 5 \
    --batch_size 32 \
    --device    cuda
