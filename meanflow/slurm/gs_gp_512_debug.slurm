#!/usr/bin/env bash
#SBATCH --job-name=gs_gp_512_debug
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pdemeanflow

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/meanflow

DATA=/scratch/user/u.kt348068/PDE_data/gs_512.pt
OUT=/scratch/user/u.kt348068/ckpt/gs_gp_512_debug

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SLURM_PROCID

python train_coupled.py \
    --dataset=grayscott \
    --data_path=$DATA \
    --output_dir=$OUT \
    --train_ratio=0.6305 \
    --val_ratio=0.1232 \
    --arch=unet \
    --batch_size=32 \
    --lr=0.0006 \
    --epochs=2 \
    --warmup_epochs=1 \
    --eval_frequency=1 \
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
    --log_per_step=1 \
    --use_gp \
    --dice_prob=0.5 \
    --gp_log_length_scale=0.0 \
    --num_workers=2 \
    --pin_mem \
    --test_run
