#!/usr/bin/env bash
# Transolver2d on Gray-Scott (2-D, 2 processes)
# Wu et al. (2024)  --  adapted: concat all process channels

cd "$(dirname "$0")/.."

python train_baseline.py \
    --model transolver2d \
    --dataset grayscott \
    --data_path ../data_generator/gray-scott/tiny_set/gs.pt \
    --output_dir ./output/grayscott_transolver2d \
    --n_proc 2 \
    --width 128 \
    --n_slices 32 \
    --n_heads 8 \
    --n_layers 6 \
    --batch_size 16 \
    --epochs 400 \
    --warmup_epochs 50 \
    --lr 1e-3 \
    --eval_frequency 20 \
    --log_per_step 10 \
    "$@"
