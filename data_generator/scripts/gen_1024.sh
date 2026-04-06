#!/bin/bash
# Generate all benchmark datasets — Option B: 1024 training samples
# Run from the repo root:  bash data_generator/scripts/gen_1024.sh
#
# Produces in data_generator/datasets/:
#   gs_1024.pt   [N_traj, 3, 2, T, 64, 64]
#   lv_1024.pt   [N_traj, 10, 2, T]
#   bz_1024.pt   [N_traj, 4, 3, T, 256]
#
# Recommended dataloader split ratios for 1024 training samples:
#   --train_ratio 0.7734   (~1024 train)
#   --val_ratio   0.0756   (~100 val)
#   (remainder    ~0.1510  (~200 test))

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GEN_DIR="$REPO_ROOT/data_generator"
OUT_DIR="$GEN_DIR/datasets"
N=1024
WORKERS=8

echo "===== Generating datasets: n_samples=$N ====="
echo "Output dir: $OUT_DIR"

cd "$GEN_DIR"

echo ""
echo "----- Gray-Scott (2-D 64x64, 2 processes) -----"
python gen_data.py --system gs --n_samples $N --output_dir "$OUT_DIR" --workers $WORKERS

echo ""
echo "----- Lotka-Volterra (1-D 256-pt, 2 processes) -----"
python gen_data.py --system lv --n_samples $N --output_dir "$OUT_DIR" --workers $WORKERS

echo ""
echo "----- Belousov-Zhabotinsky (1-D 256-pt, 3 processes) -----"
python gen_data.py --system bz --n_samples $N --output_dir "$OUT_DIR" --workers $WORKERS

echo ""
echo "===== All done. Files in $OUT_DIR ====="
ls -lh "$OUT_DIR"
