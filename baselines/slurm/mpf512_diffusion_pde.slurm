#!/usr/bin/env bash
#SBATCH --job-name=mpf512_diffusion_pde
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pde_baselines

DATA=/scratch/user/u.kt348068/PDE_data/mpf_512.pt
CKPT=/scratch/user/u.kt348068/ckpt/mpf512_diffusion_pde

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/baselines

unset SLURM_PROCID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_diffusion_pde.py \
    --dataset multiphase \
    --data_path $DATA \
    --output_dir $CKPT \
    --n_proc 2 \
    --img_resolution 64 \
    --model_channels 64 \
    --num_blocks 4 \
    --dropout 0.10 \
    --train_ratio 0.6305 \
    --val_ratio 0.1232 \
    --batch_size 32 \
    --epochs 500 \
    --warmup_epochs 50 \
    --lr 2e-4 \
    --eval_frequency 20 \
    --eval_num_steps 10 \
    --num_steps 20 \
    --log_per_step 50 \
    --auto_resume
