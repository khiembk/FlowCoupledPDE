# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Main coupled PDE framework
conda env create -f environment.yml && conda activate pdemeanflow

# Baseline models (separate env — name is pde_baselines, not baseline)
conda env create -f baselines/baseline.yaml && conda activate pde_baselines
```

## Common Commands

**Data generation** (run from repo root):
```bash
python data_generator/gen_data.py --system gs  --n_samples 512 --output_dir ./datasets --workers 8
python data_generator/gen_data.py --system lv  --n_samples 512 --output_dir ./datasets --workers 8
python data_generator/gen_data.py --system bz  --n_samples 512 --output_dir ./datasets --workers 8
python data_generator/gen_data.py --system mpf --n_samples 512 --output_dir ./datasets
python data_generator/gen_data.py --system thm --n_samples 512 --output_dir ./datasets --workers 8
```
`--n_samples 512` → 512 train + 100 val + 200 test. `--n_samples 1024` → 1024 + 100 + 200.

**Training (coupled flow — GS, LV, MPF)** (run from `meanflow/`):
```bash
python train_coupled.py \
    --dataset=grayscott --data_path=<path>/gs_512.pt \
    --output_dir=./output --arch=unet \
    --batch_size=32 --lr=0.0006 --epochs=500 --warmup_epochs=100 \
    --train_ratio=0.6305 --val_ratio=0.1232 \
    --dropout=0.2 --ema_decay=0.9999 --ema_decays 0.99995 0.9996 \
    --use_gp --dice_prob=0.5 --gp_log_length_scale=0.0 \
    --seq_loss --auto_resume

# Lightweight arch (~7M/net): --arch=unet_lite (64×64) or --arch=unet32_lite (32×32)
# 1D datasets (LV): --arch=unet1d --dataset=lv
# MPF: use --dataset=grayscott (same 2-process 64×64 format)
# Multi-GPU: torchrun --standalone --nproc_per_node=8 train_coupled.py [args]
```

**Training (BZ — 3-process 1D)** (run from `meanflow/`):
```bash
python train_bz.py \
    --data_path=<path>/bz_512.pt --output_dir=./output \
    --batch_size=32 --lr=0.0006 --epochs=500 --warmup_epochs=100 \
    --train_ratio=0.6305 --val_ratio=0.1232 \
    --dropout=0.1 --ema_decay=0.9999 --ema_decays 0.99995 0.9996 \
    --use_gp --dice_prob=0.3333 --gp_log_length_scale=0.0 \
    --seq_loss --auto_resume
# arch is always unet1d_bz (~4.5M/net); no --arch flag needed
```

**Training (THM — 5-process 2D)** (run from `meanflow/`):
```bash
python train_thm.py \
    --data_path=<path>/thm_512.pt --output_dir=./output \
    --batch_size=16 --lr=0.0006 --epochs=500 --warmup_epochs=100 \
    --train_ratio=0.6305 --val_ratio=0.1232 \
    --dropout=0.1 --ema_decay=0.9999 --ema_decays 0.99995 0.9996 \
    --use_gp --gp_log_length_scale=0.0 \
    --seq_loss --auto_resume
# arch is always unet_lite (~7M/net × 5 nets ≈ 35M total); no --arch flag needed
```

**Training (baselines)** (run from `baselines/`):
```bash
python train_baseline.py \
    --model fno2d --dataset grayscott --data_path gs_512.pt \
    --output_dir ./output --n_proc 2 \
    --modes 16 --width 64 --n_layers 4 --padding 9 \
    --train_ratio 0.6305 --val_ratio 0.1232 \
    --batch_size 32 --epochs 500 --lr 1e-3 --auto_resume

# 1D models: --model fno1d --dataset lv
```
Paper-correct hyperparams: `--modes 16 --width 64 --n_layers 4` for all FNO/UFNO/COMPOL; DeepONet `--epochs 10000`; Transolver `--epochs 500`.

**Evaluation** (run from `meanflow/`):
```bash
python eval_coupled.py --dataset=grayscott --data_path=<path>/gs_512.pt \
    --output_dir=<checkpoint_dir> --train_ratio=0.6305 --val_ratio=0.1232
```
Evaluates all EMA variants (noema, ema, ema0.99995, ema0.9996).

**Split ratios** by dataset size:
- `--n_samples 512`:  `--train_ratio 0.6305 --val_ratio 0.1232`  (total 812)
- `--n_samples 1024`: `--train_ratio 0.7734 --val_ratio 0.0755`  (total 1324)

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `unset SLURM_PROCID` for single-GPU SLURM runs (already in all slurm scripts).

## Architecture Overview

### Training Scripts

| Script | Dataset | Model class |
|---|---|---|
| `train_coupled.py` | GS, LV, MPF | `CoupledFlow` (2-process) |
| `train_bz.py` | BZ | `CoupledFlowBZ` (3-process) |
| `train_thm.py` | THM | `CoupledFlowNProc` (N-process, N=5) |
| `bezier_coupled_training.py` | GS, LV | `CoupledFlowBezier` (2-process, Bezier variant) |

All training scripts share `training/load_and_save.py` for checkpoint I/O. `train_coupled.py` uses `train_arg_parser.py`; `train_bz.py` and `train_thm.py` have their own inline arg parsers.

### Core Framework: CoupledFlow (`meanflow/models/coupled_flow.py`)

Two parallel `SongUNet` networks (`net1`, `net2`) each predict the velocity field for one coupled process. Each network receives **both** processes concatenated (`in_channels=2`) and outputs one channel (`out_channels=1`). Network configs are in `COUPLED_CONFIGS` in `meanflow/models/model_configs.py`; `dropout` is injected at instantiation.

`COUPLED_CONFIGS["unet"]` is hardcoded for `img_resolution=64, in_channels=2` (2-process 2D systems). `COUPLED_CONFIGS["unet1d"]` is hardcoded for `seq_len=256, in_channels=2` (2-process 1D systems). Adding new datasets with a different number of processes or spatial resolution requires updating these configs.

`use_checkpoint=True` is enabled in `COUPLED_CONFIGS["unet"]` for memory efficiency. It is safe with `torch.func.jvp` because `coupled_flow.py` temporarily disables it during JVP computation (JVP runs inside `torch.no_grad()` and the result is detached before entering the backward graph).

### N-Process Extension: CoupledFlowNProc (`meanflow/models/coupled_flow_nproc.py`)

Generalizes CoupledFlow to N processes. Each of the N nets receives all N processes concatenated (`in_channels=N`) and outputs one channel. Instantiated via `instantiate_coupled_nproc_model(args, n_proc)` which always uses `COUPLED_CONFIGS["unet_lite"]` with `in_channels` overridden to `n_proc`.

DICE coupling is generalized to N processes: driver is chosen by uniform categorical; remaining N-1 processes are GP-driven via separate `VectorValuedGP` modules (`gp_1`, ..., `gp_N`).

JVP in the N-process case: `torch.func.jvp(u_func, (*zs, t_b, r_b), (*vs, dtdt, drdt))` with dynamic unpacking of all N processes.

### Architecture Configs & Param Counts

All configs in `COUPLED_CONFIGS` (`meanflow/models/model_configs.py`):

| Arch key | Resolution | Params/net | Use case |
|---|---|---|---|
| `unet` | 64×64 2D | ~56M | GS, MPF full-size |
| `unet32` | 32×32 2D | ~56M | Small 2D datasets |
| `unet_lite` | 64×64 2D | ~7M | MPF small, THM (×5 nets) |
| `unet32_lite` | 32×32 2D | ~7M | GS tiny dataset |
| `unet1d` | 1D len=256 | ~4.5M | LV (2-process) |
| `unet1d_bz` | 1D len=256 | ~4.5M | BZ (3-process, in_channels=3) |

`MODEL_ARCHS` in `model_configs.py` maps arch key → class. `train_arg_parser.py` `--arch` choices must stay in sync with `MODEL_ARCHS`.

### Loss Functions (`meanflow/training/coupled_training_loop.py`)

- **`forward_global_loss`**: enforces velocity matching at endpoints (t=1, r=0). Simple L2, no JVP.
- **`forward_local_loss`**: enforces instantaneous velocity at intermediate time `t` (with `t >= r`). Uses `torch.func.jvp` to compute `du/dt` needed for the MeanFlow target.
- **`train_combined_loss_step`**: single backward on `local + global`. More peak memory.
- **`train_combine_loss_squence_step`** (`--seq_loss`): two separate backward passes — local first, then global. Halves peak activation memory. **Recommended for 2D UNet on 80GB GPU.**

N-process equivalents are in `training/nproc_training_loop.py`.

### Key JVP + DDP Caveat

`torch.func.jvp` does not support DDP objects. The codebase uses `model.module` (unwrapped) for JVP, then calls `synchronize_gradients(model)` with `gradient_sanity_check` to handle DDP grad sync manually. Any new JVP-based loss must follow this pattern.

### DICE Coupling & GP (`meanflow/models/gp_velocity.py`)

Per sample, a coin flip with `dice_prob` decides which process is the "linear driver" and which is "GP-driven". The `VectorValuedGP` computes a **deterministic** posterior mean conditioned on both endpoints (t=0, t=1):

```
z_driven(t) = w0(t) * z_driven_0 + w1(t) * z_driven_1
```

where weights come from a squared-exponential kernel with learned `log_length_scale`. Kernel distance is normalized by feature dimension for scale-invariance. Two GP modules (`gp_12`, `gp_21`) handle both coupling directions.

### EMA

`EMAModule` (`meanflow/models/ema.py`) maintains one primary EMA (`--ema_decay`, default 0.9999) plus additional copies specified by `--ema_decays` (e.g., `0.99995 0.9996`). Only `checkpoint-last.pth` is saved (no best-checkpoint logic).

**Known EMA instability**: for multi-process (THM, MPF) or large-dataset runs, EMA copies often diverge after 50–120 epochs while the raw model (`noema`) remains stable. If EMA diverges, lower the lr with `--reset_optimizer` on restart (see below). `noema` is often the best variant when training hasn't fully converged.

### Checkpoint Resume & `--reset_optimizer`

`--auto_resume` loads `checkpoint-last.pth` from `--output_dir`, restoring model weights, optimizer state, and lr schedule. **To change lr on resume** (e.g., to stabilize EMA divergence), add `--reset_optimizer`: this loads model weights only and starts a fresh optimizer at the new `--lr`. Both flags are supported in `train_coupled.py`, `train_bz.py`, and `train_thm.py`.

### Data Format & Loaders

All datasets are `.pt` tensors of shape `[N_traj, N_env, n_chan, T, H, W]` (2D) or `[N_traj, N_env, n_chan, T, L]` (1D). With `flatten_env=True` (default), each (trajectory, environment) pair is an independent sample. Loaders return 4-tuples `(z1_1, z1_2, z0_1, z0_2)` where `z1_*` = source (earlier), `z0_*` = target (later).

Data loaders live in `meanflow/data_loaders/`. The baselines script adds this to `sys.path` automatically so it can reuse them.

### Bezier Coupled Flow (`meanflow/bezier_coupled_training.py`)

An alternative to CoupledFlow that parameterizes trajectories as quadratic Bezier curves:
```
z_t = (1-t)z^0 + t*z^1 + t(t-1)*φ
v_t = (z^1 - z^0) + (2t-1)*φ
```
where `φ` is a learned control point produced by `BezierEncodingNet` (`meanflow/models/bezier_coupled_flow.py`). The model class is `CoupledFlowBezier`; training uses `bezier_coupled_training.py` with the same arg parser as the coupled flow trainer plus `--phi_lr`, `--bezier_hidden`, and `--meta_grad`.

**`--meta_grad`** implements Algorithm 1 eq. 32: φ is updated via meta-gradient `∇_φ L_global(θ̃)` using `create_graph=True` + `functional_call`. Without `--meta_grad`, φ receives gradient directly through `z_t(φ)` in the local loss (simpler and more stable).

**Known issue**: when `forward_global_loss` uses a straight-line target `v = source - target` (no φ), the only gradient path to φ is through the Hessian `∂²L_local/∂θ∂φ`, which is numerically unstable at init. Fix: include φ in the global loss target — at t=1, true Bezier velocity is `v¹ = (z^0 - z^1) + φ`.

### Baseline Models (`baselines/models/`)

Six neural operator baselines, each with 1D and 2D variants:

| `--model` flag | File |
|---|---|
| `fno2d` / `fno1d` | `fno.py` |
| `ufno2d` / `ufno1d` | `ufno.py` |
| `deeponet2d` / `deeponet1d` | `deeponet.py` |
| `transolver2d` / `transolver1d` | `transolver.py` |
| `cmwno2d` / `cmwno1d` | `cmwno.py` |
| `compol2d` / `compol1d` | `compol.py` |

All models receive `in_channels = n_proc * channels_per_proc` (processes concatenated). Use `--n_proc 2` for GS/LV/MPF, `--n_proc 3` for BZ, `--n_proc 5` for THM. `compol2d/1d` additionally takes `--aggr_type atn|rnn` and `--n_heads`. DeepONet requires `--epochs 10000` (paper setting); 500 epochs is insufficient. `cmwno2d` supports arbitrary `n_proc` (fixed to not assert n_proc==2).

### Supported PDE Systems

| System | File | Dim | Channels | N_env | Coupled flow trainer |
|---|---|---|---|---|---|
| `gs` | `gs_{N}.pt` | 2D, 64×64 | 2 (u, v) | 1 | `train_coupled.py --dataset grayscott --arch unet` |
| `lv` | `lv_{N}.pt` | 1D, len=256 | 2 (u, v) | 1 | `train_coupled.py --dataset lv --arch unet1d` |
| `mpf` | `mpf_{N}.pt` | 2D, 64×64 | 2 (Sw, P) | 3 | `train_coupled.py --dataset grayscott --arch unet` |
| `bz` | `bz_{N}.pt` | 1D, len=256 | 3 (u, v, w) | 1 | `train_bz.py` |
| `thm` | `thm_{N}.pt` | 2D, 64×64 | 5 (T, p, ε_xx, ε_yy, ε_xy) | 1 | `train_thm.py` |

GS uses block IC + T_warmup=200 burn-in + T_pred=20. MPF reuses the grayscott loader (same tensor format). For lightweight runs use `--arch unet_lite` (64×64, ~7M/net) or `--arch unet32_lite` (32×32, ~7M/net).

### Approximate Training Times (single A100 80GB GPU, GP method)

| Dataset | sec/epoch | hrs for 500 ep |
|---|---|---|
| LV512 / LV1024 | 6 / 10 | 0.9 / 1.4 |
| GS512 / GS1024 | 9 / 17 | 1.3 / 2.4 |
| MPF512 / MPF1024 | ~30 / ~60 | ~4 / ~8 |
| BZ512 / BZ1024 | ~30 / ~60 | ~4 / ~8 |
| THM512 / THM1024 | ~60 / ~120 | ~8 / ~17 |

Times for MPF/BZ/THM are training-only estimates; wall-clock measurements include eval overhead.

### Key CLI Flags

| Flag | Default | Notes |
|---|---|---|
| `--seq_loss` | False | Sequential backward passes; halves peak memory for 2D UNet |
| `--use_gp` | False | Enable DICE+GP coupling (recommended) |
| `--dice_prob` | 0.5 | Per-sample probability that process 1 is linear driver (use 1/N for N-process) |
| `--gp_log_length_scale` | 0.0 | Learnable GP kernel length-scale init (ell=1.0) |
| `--tr_sampler` | `v1` | Time sampling strategy (`v0`/`v1`) |
| `--ratio` | 0.75 | Probability that auxiliary time r ≠ t |
| `--test_run` | False | Run one batch + one eval cycle for debugging |
| `--eval_only` | False | Skip training, run evaluation only |
| `--auto_resume` | False | Resume from `checkpoint-last.pth` in output_dir |
| `--reset_optimizer` | False | When resuming, load weights only — reset optimizer to current `--lr` |
| `--compile` | False | Wrap train step in `torch.compile` |
