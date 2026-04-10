#!/usr/bin/env bash
#SBATCH --job-name=mpf1024_deeponet2d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pde_baselines

DATA=/scratch/user/u.kt348068/PDE_data/mpf_1024.pt
cd /home/u.kt348068/khiemtt/FlowCoupledPDE/baselines

python train_baseline.py \
    --model deeponet2d \
    --dataset multiphase \
    --data_path $DATA \
    --output_dir /scratch/user/u.kt348068/ckpt/mpf1024_deeponet2d \
    --n_proc 2 \
    --width 64 \
    --deeponet_p 128 \
    --train_ratio 0.6305 \
    --val_ratio 0.1232 \
    --batch_size 32 \
    --epochs 500 \
    --warmup_epochs 50 \
    --lr 1e-3 \
    --eval_frequency 20 \
    --log_per_step 50 \
    --auto_resume
