#!/usr/bin/env bash
#SBATCH --job-name=gs_transolver2d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pde_baselines

DATA=/home/u.kt348068/khiemtt/FlowCoupledPDE/data_generator/gray-scott/tiny_set/gs.pt
cd /home/u.kt348068/khiemtt/FlowCoupledPDE/baselines

python train_baseline.py \
    --model transolver2d \
    --dataset grayscott \
    --data_path $DATA \
    --output_dir ./output/grayscott_transolver2d \
    --n_proc 2 \
    --width 64 \
    --n_layers 4 \
    --n_slices 32 \
    --n_heads 4 \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --batch_size 32 \
    --epochs 400 \
    --warmup_epochs 50 \
    --lr 1e-3 \
    --eval_frequency 20 \
    --log_per_step 10 \
    --auto_resume
