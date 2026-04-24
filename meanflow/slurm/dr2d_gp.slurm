#!/usr/bin/env bash
#SBATCH --job-name=dr2d_gp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pdemeanflow

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/meanflow

DATA=/scratch/user/u.kt348068/PDE_data/PDEBench/2D_diff_react.h5
OUT=/scratch/user/u.kt348068/ckpt/dr2d_gp

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SLURM_PROCID

python train_coupled.py \
    --dataset=dr2d \
    --data_path=$DATA \
    --output_dir=$OUT \
    --arch=unet_lite \
    --batch_size=32 \
    --lr=0.0006 \
    --epochs=500 \
    --warmup_epochs=100 \
    --eval_frequency=20 \
    --tr_sampler=v1 \
    --P_mean_t=-0.6 \
    --P_std_t=1.6 \
    --P_mean_r=-4.0 \
    --P_std_r=1.6 \
    --ratio=0.75 \
    --norm_p=0.75 \
    --norm_eps=1e-3 \
    --dropout=0.2 \
    --ema_decay=0.9999 \
    --ema_decays 0.99995 0.9996 \
    --log_per_step=100 \
    --use_gp \
    --dice_prob=0.5 \
    --gp_log_length_scale=0.0 \
    --num_workers=8 \
    --pin_mem \
    --seq_loss \
    --auto_resume
