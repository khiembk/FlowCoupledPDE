#!/usr/bin/env bash
#SBATCH --job-name=test_ns2d
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
conda activate pde_baselines

DATA=/scratch/user/u.kt348068/PDE_data/ns2d.pt
BASE=/scratch/user/u.kt348068/ckpt

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/baselines

unset SLURM_PROCID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== FNO2D ====="
python train_baseline.py \
    --dataset ns2d --data_path $DATA \
    --output_dir $BASE/ns2d_fno2d \
    --model fno2d --n_proc 3 \
    --modes 16 --width 64 --n_layers 4 --padding 9 \
    --batch_size 8 --epochs 1 --test_run

echo "===== UFNO2D ====="
python train_baseline.py \
    --dataset ns2d --data_path $DATA \
    --output_dir $BASE/ns2d_ufno2d \
    --model ufno2d --n_proc 3 \
    --modes 16 --width 64 --n_layers 4 --padding 9 \
    --batch_size 8 --epochs 1 --test_run

echo "===== CMWNO2D ====="
python train_baseline.py \
    --dataset ns2d --data_path $DATA \
    --output_dir $BASE/ns2d_cmwno2d \
    --model cmwno2d --n_proc 3 \
    --width 64 --n_layers 4 \
    --batch_size 8 --epochs 1 --test_run

echo "===== COMPOL2D ATN ====="
python train_baseline.py \
    --dataset ns2d --data_path $DATA \
    --output_dir $BASE/ns2d_compol2d_atn \
    --model compol2d --n_proc 3 \
    --modes 16 --width 64 --n_layers 4 --padding 9 \
    --aggr_type atn --n_heads 4 \
    --batch_size 8 --epochs 1 --test_run

echo "===== COMPOL2D RNN ====="
python train_baseline.py \
    --dataset ns2d --data_path $DATA \
    --output_dir $BASE/ns2d_compol2d_rnn \
    --model compol2d --n_proc 3 \
    --modes 16 --width 64 --n_layers 4 --padding 9 \
    --aggr_type rnn --n_heads 4 \
    --batch_size 8 --epochs 1 --test_run

echo "===== DiffusionPDE ====="
python train_diffusion_pde.py \
    --dataset ns2d --data_path $DATA \
    --output_dir $BASE/ns2d_diffusion_pde \
    --n_proc 3 \
    --img_resolution 64 \
    --model_channels 64 \
    --num_blocks 4 \
    --dropout 0.10 \
    --batch_size 8 --epochs 1 \
    --eval_num_steps 10 --num_steps 20 \
    --test_run

echo "===== ALL TESTS DONE ====="
