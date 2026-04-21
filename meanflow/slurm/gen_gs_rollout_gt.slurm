#!/usr/bin/env bash
#SBATCH --job-name=gs_rollout_gt
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pdemeanflow

cd /home/u.kt348068/khiemtt/FlowCoupledPDE

DATA=/scratch/user/u.kt348068/PDE_data/gs_512.pt
OUT=/scratch/user/u.kt348068/PDE_data/gs_rollout_gt.pt

python data_generator/gen_gs_rollout_gt.py \
    --data_path $DATA \
    --out_path  $OUT \
    --train_ratio 0.6305 \
    --val_ratio 0.1232 \
    --workers 8
