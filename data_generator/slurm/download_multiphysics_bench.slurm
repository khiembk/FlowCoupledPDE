#!/usr/bin/env bash
#SBATCH --job-name=dl_multiphysics
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pdemeanflow

mkdir -p /scratch/user/u.kt348068/PDE_data/Multiphysices_bench

python - << 'PYEOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Indulge-Bai/Multiphysics_Bench",
    repo_type="dataset",
    local_dir="/scratch/user/u.kt348068/PDE_data/Multiphysices_bench",
    local_dir_use_symlinks=False,
)
print("Download complete.")
PYEOF
