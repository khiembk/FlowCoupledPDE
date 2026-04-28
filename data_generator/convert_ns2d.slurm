#!/usr/bin/env bash
#SBATCH --job-name=convert_ns2d
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate flow

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/data_generator

python convert_ns2d.py \
    --input_dir /scratch/user/u.kt348068/PDE_data/PDEArena/NavierStokes2D \
    --output /scratch/user/u.kt348068/PDE_data/ns2d.pt \
    --resolution 64
