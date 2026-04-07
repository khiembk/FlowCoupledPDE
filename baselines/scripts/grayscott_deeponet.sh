#!/usr/bin/env bash
# DeepONet2d on Gray-Scott (2-D, 2 processes)
# Lu et al. (2021)  --  adapted: concat all process channels

cd "$(dirname "$0")/.."

python train_baseline.py \
    --model deeponet2d \
    --dataset grayscott \
    --data_path ../data_generator/gray-scott/tiny_set/gs.pt \
    --output_dir ./output/grayscott_deeponet2d \
    --n_proc 2 \
    --width 64 \
    --deeponet_p 128 \
    --batch_size 32 \
    --epochs 400 \
    --warmup_epochs 50 \
    --lr 1e-3 \
    --eval_frequency 20 \
    --log_per_step 10 \
    "$@"
