#!/usr/bin/env bash
#SBATCH --job-name=lv_rollout
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh

GT=/scratch/user/u.kt348068/PDE_data/lv_rollout_gt.pt
DATA=/scratch/user/u.kt348068/PDE_data/lv_512.pt
CKPT=/scratch/user/u.kt348068/ckpt

cd /home/u.kt348068/khiemtt/FlowCoupledPDE

# ── Step 1: MeanFlow-GP + MeanFlow-wo_gp (pdemeanflow env) ───────────────────
conda activate pdemeanflow
echo ""
echo "========================================================"
echo "  MeanFlow-GP + wo_gp rollout evaluation"
echo "========================================================"
python rollout/eval_rollout_lv_meanflow.py \
    --gt_path    $GT \
    --ckpt_dir   $CKPT \
    --models     lv512_gp lv512_wo_gp \
    --max_steps  10 \
    --batch_size 32 \
    --device     cuda

# ── Step 3: NaiveMeanFlow (pdemeanflow env) ───────────────────────────────────
echo ""
echo "========================================================"
echo "  NaiveMeanFlow rollout evaluation"
echo "========================================================"
python rollout/eval_rollout_lv_naive.py \
    --gt_path    $GT \
    --ckpt_dir   $CKPT \
    --models     naive_lv512 \
    --max_steps  10 \
    --batch_size 32 \
    --device     cuda

# ── Step 4: DiffusionPDE + FNO1d + UFNO1d + COMPOL1d (pde_baselines env) ─────
conda activate pde_baselines

echo ""
echo "========================================================"
echo "  DiffusionPDE + baseline rollout evaluation"
echo "========================================================"
python rollout/eval_rollout_lv_baselines.py \
    --gt_path    $GT \
    --ckpt_dir   $CKPT \
    --dataset    lv512 \
    --ckpt_name  checkpoint-best.pth \
    --max_steps  10 \
    --batch_size 32 \
    --device     cuda
