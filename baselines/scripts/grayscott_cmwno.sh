#!/usr/bin/env bash
# CMWNO2d on Gray-Scott (2-D, 2 processes)
# Xiao et al. (ICLR 2023)  --  natively coupled

cd "$(dirname "$0")/.."

python train_baseline.py \
    --model cmwno2d \
    --dataset grayscott \
    --data_path ../data_generator/gray-scott/tiny_set/gs.pt \
    --output_dir ./output/grayscott_cmwno2d \
    --n_proc 2 \
    --width 32 \
    --n_layers 4 \
    --wavelet_k 2 \
    --batch_size 32 \
    --epochs 400 \
    --warmup_epochs 50 \
    --lr 1e-3 \
    --eval_frequency 20 \
    --log_per_step 10 \
    "$@"
