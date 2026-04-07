#!/usr/bin/env bash
# Submit all Gray-Scott baseline training jobs to SLURM.
# Usage: bash submit_all.sh
# Logs land in baselines/slurm/<jobname>_<jobid>.out/.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for slurm_file in "$SCRIPT_DIR"/fno2d.slurm "$SCRIPT_DIR"/ufno2d.slurm \
                  "$SCRIPT_DIR"/deeponet2d.slurm "$SCRIPT_DIR"/transolver2d.slurm \
                  "$SCRIPT_DIR"/cmwno2d.slurm "$SCRIPT_DIR"/compol2d.slurm; do
    job=$(sbatch --chdir="$SCRIPT_DIR" "$slurm_file")
    echo "$job  ← $(basename $slurm_file)"
done
