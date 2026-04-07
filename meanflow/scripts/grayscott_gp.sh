#!/bin/bash
# Training on Gray-Scott with vector-valued GP + DICE strategy
# Run from the meanflow/ directory: bash scripts/grayscott_gp.sh

torchrun --standalone --nproc_per_node=1 --master_port=12345 \
    train_coupled.py \
    --dataset=grayscott \
    --data_path=../data_generator/gray-scott/tiny_set/gs.pt \
    --output_dir=./output_grayscott_gp \
    --arch=unet \
    --batch_size=32 \
    --lr=0.0006 \
    --epochs=400 \
    --warmup_epochs=100 \
    --eval_frequency=10 \
    --tr_sampler=v1 \
    --P_mean_t=-0.6 \
    --P_std_t=1.6 \
    --P_mean_r=-4.0 \
    --P_std_r=1.6 \
    --ratio=0.75 \
    --norm_p=0.75 \
    --norm_eps=1e-3 \
    --dropout=0.2 \
    --ema_decay=0.9999 \
    --ema_decays 0.99995 0.9996 \
    --log_per_step=100 \
    --use_gp \
    --dice_prob=0.5 \
    --gp_log_length_scale=0.0 \
    --num_workers=4 \
    --pin_mem
