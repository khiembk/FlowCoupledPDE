#!/usr/bin/env bash
#SBATCH --job-name=gen_gs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pdemeanflow

# Avoid BLAS thread over-subscription with multiprocessing workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT=/scratch/user/u.kt348068/PDE_data
WORKERS=6

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/data_generator

echo "===== Gray-Scott 512 ====="
python gen_data.py --system gs --n_samples 512  --output_dir $OUT --workers $WORKERS

echo "===== Gray-Scott 1024 ====="
python gen_data.py --system gs --n_samples 1024 --output_dir $OUT --workers $WORKERS

echo "Done. Output:"
ls -lh $OUT/gs_*.pt
