#!/usr/bin/env bash
#SBATCH --job-name=test_naive_dr2d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate flow

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/baselines

DATA=/scratch/user/u.kt348068/PDE_data/dr2d.pt
OUT=/scratch/user/u.kt348068/ckpt/test_naive_dr2d

unset SLURM_PROCID

python train_naive_flow.py \
    --dataset dr2d \
    --data_path $DATA \
    --output_dir $OUT \
    --batch_size 32 \
    --lr 6e-4 \
    --epochs 2 \
    --warmup_epochs 1 \
    --eval_frequency 1 \
    --dropout 0.2 \
    --ema_decay 0.9999 \
    --ema_decays 0.99995 0.9996 \
    --tr_sampler v1 \
    --P_mean_t -0.6 \
    --P_std_t 1.6 \
    --P_mean_r -4.0 \
    --P_std_r 1.6 \
    --ratio 0.75 \
    --norm_p 0.75 \
    --norm_eps 1e-3 \
    --num_workers 4 \
    --log_per_step 10 \
    --test_run
