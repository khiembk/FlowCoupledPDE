#!/usr/bin/env bash
# Submit all Gray-Scott baseline training jobs to SLURM.
# Usage: bash submit_all.sh
# Logs land in baselines/slurm/<jobname>_<jobid>.out/.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for slurm_file in "$SCRIPT_DIR"/*.slurm; do
    job=$(sbatch --chdir="$SCRIPT_DIR" "$slurm_file")
    echo "$job  ← $(basename $slurm_file)"
done
