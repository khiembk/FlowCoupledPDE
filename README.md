# FlowCoupledPDE

A flow-matching framework for learning coupled PDE systems, with six neural operator baselines for comparison.

## Setup

```bash
# Main method
conda env create -f environment.yml
conda activate pdemeanflow

# Baselines
conda env create -f baselines/baseline.yaml
conda activate pde_baselines
```

## Data Generation

Run from repo root. Datasets are saved as `.pt` tensors.

```bash
python data_generator/gen_data.py --system gs  --n_samples 512 --output_dir ./datasets
python data_generator/gen_data.py --system mpf --n_samples 512 --output_dir ./datasets
python data_generator/gen_data.py --system thm --n_samples 512 --output_dir ./datasets
python data_generator/gen_data.py --system bz  --n_samples 512 --output_dir ./datasets
```

Supported systems: `gs` (Gray-Scott, 2D), `lv` (Lotka-Volterra, 1D ODE), `bz` (Belousov-Zhabotinsky, 1D), `mpf` (Multiphase Flow, 2D), `thm` (Thermal-Hydro-Mechanical, 2D, 5 processes).

Paper uses `--n_samples 512` (512 train + 100 val + 200 test trajectories).

## Training: Main Method

Run from `meanflow/`:

```bash
# Single GPU (local)
torchrun --standalone --nproc_per_node=1 --master_port=12345 train_coupled.py \
    --dataset=grayscott \
    --data_path=<path>/gs_512.pt \
    --output_dir=<output_dir> \
    --train_ratio=0.6305 --val_ratio=0.1232 \
    --arch=unet \
    --batch_size=32 --lr=0.0006 --epochs=500 --warmup_epochs=100 \
    --eval_frequency=10 \
    --tr_sampler=v1 \
    --P_mean_t=-0.6 --P_std_t=1.6 --P_mean_r=-4.0 --P_std_r=1.6 \
    --ratio=0.75 --norm_p=0.75 --norm_eps=1e-3 \
    --dropout=0.2 --ema_decay=0.9999 --ema_decays 0.99995 0.9996 \
    --use_gp --dice_prob=0.5 --gp_log_length_scale=0.0 \
    --num_workers=4 --pin_mem --auto_resume
```

Or use the convenience script: `bash scripts/grayscott_gp.sh` (uses `tiny_set` data for quick testing).

**SLURM:**
```bash
cd meanflow
sbatch slurm/gs_gp_512.slurm
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and unset `SLURM_PROCID` when running single-GPU on SLURM (already done in the slurm scripts).

## Training: Baselines

Run from `baselines/`. Six models: `fno2d`, `ufno2d`, `deeponet2d`, `transolver2d`, `cmwno2d`, `compol2d`.

```bash
python train_baseline.py \
    --model fno2d \
    --dataset grayscott \
    --data_path <path>/gs_512.pt \
    --output_dir <output_dir> \
    --n_proc 2 \
    --modes 12 --width 64 --n_layers 4 --padding 9 \
    --train_ratio 0.6305 --val_ratio 0.1232 \
    --batch_size 32 --epochs 500 --warmup_epochs 50 \
    --lr 1e-3 --eval_frequency 20 --auto_resume
```

**SLURM (all baselines at once):**
```bash
cd baselines
bash slurm/submit_all.sh          # tiny_set (quick test)

# GrayScott-512 individual jobs:
sbatch slurm/gs512_fno2d.slurm
sbatch slurm/gs512_ufno2d.slurm
sbatch slurm/gs512_deeponet2d.slurm
sbatch slurm/gs512_transolver2d.slurm
sbatch slurm/gs512_cmwno2d.slurm
sbatch slurm/gs512_compol2d.slurm
```

## Evaluation

```bash
# From meanflow/
python eval_coupled.py \
    --dataset=grayscott \
    --data_path=<path>/gs_512.pt \
    --output_dir=<checkpoint_dir> \
    --train_ratio=0.6305 --val_ratio=0.1232
```

Metric: relative L2 error per process and averaged.

## Repository Structure

```
meanflow/          # Main method: CoupledFlow training
  train_coupled.py
  eval_coupled.py
  models/          # CoupledFlow, SongUNet, GP velocity, EMA
  training/        # Training loop, distributed utils, checkpointing
  data_loaders/
  scripts/         # Bash convenience scripts
  slurm/           # SLURM job scripts
baselines/         # Six neural operator baselines
  train_baseline.py
  models/          # fno, ufno, deeponet, transolver, cmwno, compol
  slurm/
data_generator/    # PDE data generation (gs, lv, bz, mpf, thm)
```
