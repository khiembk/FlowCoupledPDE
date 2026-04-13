#!/usr/bin/env bash
#SBATCH --job-name=bezier_gs_tiny
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pdemeanflow

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/meanflow

DATA=/home/u.kt348068/khiemtt/FlowCoupledPDE/data_generator/gray-scott/tiny_set/gs.pt
OUT=/scratch/user/u.kt348068/ckpt/bezier_gs_tiny

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SLURM_PROCID

python bezier_coupled_training.py \
    --dataset=grayscott \
    --data_path=$DATA \
    --output_dir=$OUT \
    --train_ratio=0.7 \
    --val_ratio=0.15 \
    --arch=unet32 \
    --batch_size=8 \
    --lr=0.0006 \
    --phi_lr=0.0002 \
    --epochs=100 \
    --warmup_epochs=20 \
    --eval_frequency=10 \
    --tr_sampler=v1 \
    --P_mean_t=-0.6 \
    --P_std_t=1.6 \
    --P_mean_r=-4.0 \
    --P_std_r=1.6 \
    --ratio=0.75 \
    --norm_p=0.75 \
    --norm_eps=1e-3 \
    --dropout=0.1 \
    --ema_decay=0.9999 \
    --ema_decays 0.99995 0.9996 \
    --log_per_step=50 \
    --bezier_hidden=32 \
    --num_workers=4 \
    --pin_mem \
    --auto_resume
