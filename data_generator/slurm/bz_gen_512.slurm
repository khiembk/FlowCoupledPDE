#!/usr/bin/env bash
#SBATCH --job-name=bz_gen_512
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pdemeanflow

cd /home/u.kt348068/khiemtt/FlowCoupledPDE

python data_generator/gen_data.py \
    --system bz \
    --n_samples 512 \
    --output_dir /scratch/user/u.kt348068/PDE_data \
    --workers 8

echo "Done: bz_512.pt"
