#!/usr/bin/env bash
#SBATCH --job-name=download_well
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate pdemeanflow

BASE=/scratch/user/u.kt348068/PDE_data/Well

# Download gray_scott (~158 GB), active_matter (~56 GB), rayleigh_benard (~370 GB)
DATASETS=(
    "gray_scott_reaction_diffusion"
    "active_matter"
    "rayleigh_benard"
)

python3 - <<'EOF'
import os, sys
from huggingface_hub import snapshot_download

BASE = "/scratch/user/u.kt348068/PDE_data/Well"
DATASETS = [
    "gray_scott_reaction_diffusion",
    "active_matter",
    "rayleigh_benard",
]

for ds in DATASETS:
    dest = os.path.join(BASE, ds)
    os.makedirs(dest, exist_ok=True)
    print(f"\n{'='*60}", flush=True)
    print(f"Downloading polymathic-ai/{ds} → {dest}", flush=True)
    print(f"{'='*60}", flush=True)
    try:
        snapshot_download(
            repo_id=f"polymathic-ai/{ds}",
            repo_type="dataset",
            local_dir=dest,
            max_workers=8,
            ignore_patterns=["*.gitattributes"],
        )
        print(f"✓ Done: {ds}", flush=True)
    except Exception as e:
        print(f"✗ Failed: {ds} — {e}", flush=True)
        sys.exit(1)

print("\nAll downloads complete.", flush=True)
EOF
