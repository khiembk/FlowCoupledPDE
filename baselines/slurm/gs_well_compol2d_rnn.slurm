#!/usr/bin/env bash
#SBATCH --job-name=gs_well_compol2d_rnn
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

DATA=/scratch/user/u.kt348068/PDE_data/Well/gray_scott_reaction_diffusion
cd /home/u.kt348068/khiemtt/FlowCoupledPDE/baselines

python train_baseline.py \
    --model compol2d \
    --dataset gs_well \
    --data_path $DATA \
    --output_dir /scratch/user/u.kt348068/ckpt/gs_well_compol2d_rnn \
    --n_proc 2 \
    --modes 16 \
    --width 32 \
    --n_layers 4 \
    --n_heads 4 \
    --padding 9 \
    --aggr_type rnn \
    --batch_size 32 \
    --epochs 500 \
    --warmup_epochs 50 \
    --lr 1e-3 \
    --eval_frequency 20 \
    --log_per_step 50 \
    --auto_resume
