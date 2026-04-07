#!/usr/bin/env bash
#SBATCH --job-name=gs_test_all
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

DATA=/home/u.kt348068/khiemtt/FlowCoupledPDE/data_generator/gray-scott/tiny_set/gs.pt
BASEDIR=/home/u.kt348068/khiemtt/FlowCoupledPDE/baselines
cd $BASEDIR

COMMON="--dataset grayscott --data_path $DATA --n_proc 2
        --train_ratio 0.8 --val_ratio 0.2
        --batch_size 32 --epochs 5 --warmup_epochs 1
        --eval_frequency 2 --log_per_step 5"

echo "===== FNO2d =====" && python train_baseline.py $COMMON \
    --model fno2d --modes 12 --width 64 --n_layers 4 --padding 9 \
    --output_dir ./output/test_fno2d

echo "===== UFNO2d =====" && python train_baseline.py $COMMON \
    --model ufno2d --modes 12 --width 64 --n_layers 4 --padding 9 \
    --output_dir ./output/test_ufno2d

echo "===== DeepONet2d =====" && python train_baseline.py $COMMON \
    --model deeponet2d --width 64 --deeponet_p 128 \
    --output_dir ./output/test_deeponet2d

echo "===== Transolver2d =====" && python train_baseline.py $COMMON \
    --model transolver2d --width 64 --n_layers 4 --n_slices 32 --n_heads 4 \
    --output_dir ./output/test_transolver2d

echo "===== CMWNO2d =====" && python train_baseline.py $COMMON \
    --model cmwno2d --width 32 --n_layers 4 --wavelet_k 2 \
    --output_dir ./output/test_cmwno2d

echo "===== COMPOL2d =====" && python train_baseline.py $COMMON \
    --model compol2d --modes 12 --width 32 --n_layers 4 --n_heads 4 --padding 9 \
    --output_dir ./output/test_compol2d

echo "===== ALL DONE ====="
