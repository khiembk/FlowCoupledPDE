#!/usr/bin/env bash
#SBATCH --job-name=test_ns2d_gp_naive
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate flow

DATA=/scratch/user/u.kt348068/PDE_data/ns2d.pt
BASE=/scratch/user/u.kt348068/ckpt

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/meanflow

unset SLURM_PROCID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== GP ====="
python train_ns2d.py \
    --data_path $DATA \
    --output_dir $BASE/ns2d_gp \
    --train_ratio 0.6667 --val_ratio 0.1667 \
    --batch_size 16 \
    --epochs 1 \
    --warmup_epochs 10 \
    --lr 6e-4 \
    --dropout 0.1 \
    --ema_decay 0.9999 \
    --ema_decays 0.99995 0.9996 \
    --use_gp --gp_log_length_scale 0.0 \
    --seq_loss \
    --num_workers 4 \
    --test_run

echo "===== WO_GP ====="
python train_ns2d.py \
    --data_path $DATA \
    --output_dir $BASE/ns2d_wo_gp \
    --train_ratio 0.6667 --val_ratio 0.1667 \
    --batch_size 16 \
    --epochs 1 \
    --warmup_epochs 10 \
    --lr 6e-4 \
    --dropout 0.1 \
    --ema_decay 0.9999 \
    --ema_decays 0.99995 0.9996 \
    --seq_loss \
    --num_workers 4 \
    --test_run

echo "===== NAIVE ====="
cd /home/u.kt348068/khiemtt/FlowCoupledPDE/baselines

python train_naive_flow.py \
    --dataset ns2d \
    --data_path $DATA \
    --output_dir $BASE/ns2d_naive \
    --batch_size 16 \
    --lr 6e-4 \
    --epochs 1 \
    --warmup_epochs 10 \
    --dropout 0.1 \
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
    --log_per_step 50 \
    --test_run

echo "===== ALL TESTS DONE ====="
