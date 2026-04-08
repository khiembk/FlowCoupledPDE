#!/usr/bin/env bash
#SBATCH --job-name=gen_lv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pdemeanflow

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT=/scratch/user/u.kt348068/PDE_data
WORKERS=6

cd /home/u.kt348068/khiemtt/FlowCoupledPDE/data_generator

echo "===== Lotka-Volterra 512 ====="
python gen_data.py --system lv --n_samples 512  --output_dir $OUT --workers $WORKERS

echo "===== Lotka-Volterra 1024 ====="
python gen_data.py --system lv --n_samples 1024 --output_dir $OUT --workers $WORKERS

echo "Done. Output:"
ls -lh $OUT/lv_*.pt
