#!/usr/bin/env bash
#SBATCH --job-name=test_diffusion_pde
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
conda activate pde_baselines

DATA=/scratch/user/u.kt348068/PDE_data/gs_512.pt
CKPT=/scratch/user/u.kt348068/ckpt/test_diffusion_pde

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/baselines

unset SLURM_PROCID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GS512 2D test ==="
python train_diffusion_pde.py \
    --dataset grayscott \
    --data_path $DATA \
    --output_dir ${CKPT}_gs \
    --n_proc 2 \
    --img_resolution 64 \
    --model_channels 64 \
    --num_blocks 4 \
    --dropout 0.10 \
    --train_ratio 0.6305 \
    --val_ratio 0.1232 \
    --batch_size 32 \
    --epochs 2 \
    --warmup_epochs 1 \
    --lr 2e-4 \
    --eval_frequency 1 \
    --eval_num_steps 5 \
    --num_steps 5 \
    --log_per_step 1 \
    --test_run

echo ""
echo "=== LV512 1D test ==="
python train_diffusion_pde.py \
    --dataset lv \
    --data_path /scratch/user/u.kt348068/PDE_data/lv_512.pt \
    --output_dir ${CKPT}_lv \
    --n_proc 2 \
    --img_resolution 16 \
    --model_channels 64 \
    --num_blocks 4 \
    --dropout 0.10 \
    --train_ratio 0.6305 \
    --val_ratio 0.1232 \
    --batch_size 32 \
    --epochs 2 \
    --warmup_epochs 1 \
    --lr 2e-4 \
    --eval_frequency 1 \
    --eval_num_steps 5 \
    --num_steps 5 \
    --log_per_step 1 \
    --test_run

echo ""
echo "=== ALL TESTS PASSED ==="
