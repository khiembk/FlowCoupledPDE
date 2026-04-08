# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Main coupled PDE framework
conda env create -f environment.yml && conda activate pdemeanflow

# Baseline models (separate env)
conda env create -f baselines/baseline.yaml && conda activate baseline
```

## Common Commands

**Data generation:**
```bash
python data_generator/gen_data.py --system gs --n_samples 512 --output_dir ./datasets
```

**Training (coupled flow):**
```bash
# Single GPU
python meanflow/train_coupled.py \
    --dataset=grayscott --data_path=<path>/gs.pt \
    --output_dir=./output --use_gp --dice_prob=0.5 --auto_resume

# Multi-GPU distributed
torchrun --standalone --nproc_per_node=8 meanflow/train_coupled.py [args]

# Via convenience script
bash meanflow/scripts/grayscott_gp.sh

# Via SLURM
sbatch meanflow/slurm/gs_gp_512.slurm
```

**Training (baselines):**
```bash
python baselines/train_baseline.py \
    --model fno2d --dataset grayscott --data_path gs.pt --output_dir ./output

# Submit all 6 baselines at once
bash baselines/slurm/submit_all.sh
```

**Evaluation:**
```bash
python meanflow/eval_coupled.py --output_dir=./output --data_path=<path>/gs.pt
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for large-batch training to manage memory fragmentation.

## Architecture Overview

### Core Framework: CoupledFlow (`meanflow/models/coupled_flow.py`)

The central model is `CoupledFlow` — two parallel `SongUNet` networks (`net1`, `net2`) that learn to predict velocity fields for each process in a coupled PDE system. Both networks are trained jointly with shared time sampling but independent loss backpropagation (to reduce peak memory).

EMA is managed by a custom `EMAModule` (`meanflow/models/ema.py`): one primary EMA (decay=0.9999) plus configurable extra EMA networks. The DICE coupling strategy uses Vector-valued GP modules (`gp_velocity.py`, `gp_12`, `gp_21`) to model cross-process interactions stochastically.

### Training Loop (`meanflow/training/coupled_training_loop.py`)

Each training step:
1. Samples a batch `(source_1, source_2, target_1, target_2)` from `GrayScottCoupledDataset`
2. Computes `local_loss + global_loss` per process using `torch.func.jvp`
3. Backpropagates each component's loss separately
4. Calls `synchronize_gradients(model)` to handle DDP gradient sync (required because `torch.func.jvp` must operate on `model.module`, bypassing DDP's automatic sync)
5. Updates EMA weights

`train_step` is wrapped in `torch.compile` for JVP memory and speed efficiency.

### Key JVP Caveat

`torch.func.jvp` does **not** support DDP objects directly. This codebase uses `model.module` for JVP, then manually calls `synchronize_gradients(model)` with a `gradient_sanity_check` to ensure correctness. Any new JVP-based loss computation must follow this pattern.

### Data Format

Datasets are `.pt` tensors of shape `[N_traj, N_env, n_chan, T, H, W]`. The `GrayScottCoupledDataset` (`meanflow/data_loaders/grayscott_loader.py`) returns `(z1_1, z1_2, z0_1, z0_2)` where `z1_*` is the source state and `z0_*` is the target state for each coupled process.

### Baseline Models (`baselines/models/`)

Six neural operator baselines for comparison: FNO, UFNO, DeepONet, Transolver, CMWNO, COMPOL. Each is a self-contained file; `train_baseline.py` selects among them via `--model`. Config is in `baselines/baseline.yaml`.

### Supported PDE Systems

- `gs` — Gray-Scott (2D reaction-diffusion, 2 coupled processes)
- `lv` — Lotka-Volterra (1D ODE)
- `bz` — Belousov-Zhabotinsky (1D)
- `multiphase` — Multiphase flow
- `thm` — Thermal-Hydro-Mechanical (5 coupled processes)

### Argument Parsing (`meanflow/train_arg_parser.py`)

All training hyperparameters are CLI flags. Key groups: optimizer (lr, warmup, EMA decay), dataset (path, split ratios), MeanFlow (time sampler, P_mean/std), DICE/GP (use_gp, dice_prob, gp_log_length_scale), distributed (world_size, local_rank), and debug (--compile, --test_run, --eval_only).
