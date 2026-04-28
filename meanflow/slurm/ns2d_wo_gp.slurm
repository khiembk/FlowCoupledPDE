#!/usr/bin/env bash
#SBATCH --job-name=ns2d_wo_gp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate flow

DATA=/scratch/user/u.kt348068/PDE_data/ns2d.pt
OUT=/scratch/user/u.kt348068/ckpt/ns2d_wo_gp

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/meanflow

unset SLURM_PROCID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_ns2d.py \
    --data_path $DATA \
    --output_dir $OUT \
    --train_ratio 0.6667 --val_ratio 0.1667 \
    --batch_size 32 \
    --epochs 500 \
    --warmup_epochs 100 \
    --lr 6e-4 \
    --dropout 0.1 \
    --ema_decay 0.9999 \
    --ema_decays 0.99995 0.9996 \
    --seq_loss \
    --eval_frequency 3 \
    --log_per_step 100 \
    --num_workers 4 \
    --auto_resume
