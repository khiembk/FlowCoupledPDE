#!/usr/bin/env bash
#SBATCH --job-name=lipschitz_bound
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pdemeanflow

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/meanflow

DATA=/scratch/user/u.kt348068/PDE_data
OUT=/scratch/user/u.kt348068/lipschitz_results

mkdir -p $OUT

python compute_lipschitz_bound.py \
    --data_dir $DATA \
    --datasets gs mpf lv \
    --N_samples 150 \
    --n_spatial 8 \
    --T_quad 20 \
    --l_gp 1.0 \
    --seed 42 \
    --out_dir $OUT
