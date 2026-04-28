#!/usr/bin/env bash
#SBATCH --job-name=ns2d_cmwno2d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pde_baselines

DATA=/scratch/user/u.kt348068/PDE_data/ns2d.pt
OUT=/scratch/user/u.kt348068/ckpt/ns2d_cmwno2d

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/baselines

unset SLURM_PROCID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_baseline.py \
    --dataset ns2d \
    --data_path $DATA \
    --output_dir $OUT \
    --model cmwno2d \
    --n_proc 3 \
    --width 64 --n_layers 4 \
    --batch_size 32 \
    --epochs 500 \
    --warmup_epochs 50 \
    --lr 1e-3 \
    --eval_frequency 10 \
    --log_per_step 100 \
    --auto_resume
