#!/usr/bin/env bash
#SBATCH --job-name=convert_dr2d
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate flow

cd /home/u.kt348068/khiemtt/FlowCoupledPDE

python data_generator/convert_dr2d.py \
    --input  /scratch/user/u.kt348068/PDE_data/PDEBench/2D_diff_react.h5 \
    --output /scratch/user/u.kt348068/PDE_data/dr2d.pt \
    --res 64
