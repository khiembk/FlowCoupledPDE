#!/usr/bin/env bash
# U-FNO2d on Gray-Scott (2-D, 2 processes)
# Wen et al. (2022)  --  adapted: concat all process channels

cd "$(dirname "$0")/.."

python train_baseline.py \
    --model ufno2d \
    --dataset grayscott \
    --data_path ../data_generator/gray-scott/tiny_set/gs.pt \
    --output_dir ./output/grayscott_ufno2d \
    --n_proc 2 \
    --modes 12 \
    --width 32 \
    --n_layers 4 \
    --padding 9 \
    --batch_size 32 \
    --epochs 400 \
    --warmup_epochs 50 \
    --lr 1e-3 \
    --eval_frequency 20 \
    --log_per_step 10 \
    "$@"
