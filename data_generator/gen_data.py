"""
Unified dataset generation script for FlowCoupledPDE benchmarks.

Generates datasets matching the paper's two experimental options:
    Option A  --n_samples 512   →  512 train + 100 val + 200 test trajectories
    Option B  --n_samples 1024  →  1024 train + 100 val + 200 test trajectories

Supported systems:
    gs   Gray-Scott reaction-diffusion  (2 processes, 2-D 64×64)
    lv   Lotka-Volterra predator-prey   (2 processes, 1-D ODE, 256 time-steps)
    bz   Belousov-Zhabotinsky           (3 processes, 1-D 256-pt spatial mesh)

Output format per system:
    gs   →  [N_traj, N_env, 2, T, 64,  64]   saved as  gs_<n_samples>.pt
    lv   →  [N_traj, N_env, 2, T]             saved as  lv_<n_samples>.pt
    bz   →  [N_traj, N_env, 3, T, 256]        saved as  bz_<n_samples>.pt

Usage examples:
    python gen_data.py --system gs  --n_samples 512  --output_dir ./datasets
    python gen_data.py --system bz  --n_samples 1024 --output_dir ./datasets --workers 8
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# LEADS / local imports — adjust sys.path so we can import the existing
# generators whether this script is run from repo root or data_generator/.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_LEADS_DIR = _HERE / "LEADS" / "src"          # cloned LEADS repo
_DS_DIR = _HERE / "gray-scott" / "dynamicalsystems_dataset"  # existing gen code

for _p in [str(_DS_DIR), str(_LEADS_DIR)]:
    if _p not in sys.path and Path(_p).exists():
        sys.path.insert(0, _p)

from bz_dataset import BZDataset, default_bz_params   # local


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sizes(n_samples: int, n_val: int = 100, n_test: int = 200):
    """Return (n_train, n_val, n_test, total) trajectory counts."""
    return n_samples, n_val, n_test, n_samples + n_val + n_test


def _traj_per_env(total_traj: int, n_env: int) -> int:
    """Round up so that n_env * traj_per_env >= total_traj."""
    return int(np.ceil(total_traj / n_env))


# ---------------------------------------------------------------------------
# Gray-Scott (2-D, 64×64, 2 processes)
# ---------------------------------------------------------------------------

def generate_gs(n_samples: int, output_dir: Path, workers: int) -> Path:
    try:
        from dataset_generation.gs import GrayScottReactionDataset
        from dataset_generation.samplers import SubsetRamdomSampler
    except ImportError:
        raise ImportError(
            "Could not import Gray-Scott generator. "
            "Make sure gray-scott/dynamicalsystems_dataset/ is on sys.path."
        )

    n_train, n_val, n_test, total = _split_sizes(n_samples)

    params = [
        {"D_u": 0.2097, "D_v": 0.105, "F": 0.037, "k": 0.060},
        {"D_u": 0.2097, "D_v": 0.105, "F": 0.030, "k": 0.062},
        {"D_u": 0.2097, "D_v": 0.105, "F": 0.039, "k": 0.058},
    ]
    n_env = len(params)
    n_traj = _traj_per_env(total, n_env)

    print(f"[GS] generating {n_traj} traj × {n_env} envs = {n_traj * n_env} total "
          f"(need {total}: {n_train} train / {n_val} val / {n_test} test)")

    dataset = GrayScottReactionDataset(
        num_traj_per_env=n_traj,
        time_horizon=800,
        params=params,
        dt_eval=40,
        method="RK45",
        group="train",
        size=64,          # 64×64 as per paper (was 32 in LEADS)
        dx=1.0,
        n_block=3,
        buffer=dict(),
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=workers, shuffle=False)

    all_states = []   # each item: [2, T, 64, 64]
    for i, sample in enumerate(loader):
        all_states.append(sample["state"].squeeze(0))  # [2, T, H, W]
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(dataset)}")

    # Stack: [n_traj * n_env, 2, T, H, W]
    data = torch.stack(all_states, dim=0)

    # Reshape to [n_traj, n_env, 2, T, H, W]
    T, H, W = data.shape[2], data.shape[3], data.shape[4]
    data = data.view(n_traj, n_env, 2, T, H, W)

    out_path = output_dir / f"gs_{n_samples}.pt"
    torch.save(data, out_path)
    print(f"[GS] saved {tuple(data.shape)} → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Lotka-Volterra (1-D ODE, 2 processes, 256 time-steps)
# ---------------------------------------------------------------------------

def generate_lv(n_samples: int, output_dir: Path, workers: int) -> Path:
    try:
        from dataset_generation.lv import LotkaVolterraDataset
    except ImportError:
        raise ImportError(
            "Could not import Lotka-Volterra generator. "
            "Make sure gray-scott/dynamicalsystems_dataset/ is on sys.path."
        )

    n_train, n_val, n_test, total = _split_sizes(n_samples)

    params = [
        {"alpha": 0.5,  "beta": 0.5, "gamma": 0.5,  "delta": 0.5},
        {"alpha": 0.25, "beta": 0.5, "gamma": 0.5,  "delta": 0.5},
        {"alpha": 0.75, "beta": 0.5, "gamma": 0.5,  "delta": 0.5},
        {"alpha": 1.0,  "beta": 0.5, "gamma": 0.5,  "delta": 0.5},
        {"alpha": 0.5,  "beta": 0.5, "gamma": 0.25, "delta": 0.5},
        {"alpha": 0.5,  "beta": 0.5, "gamma": 0.75, "delta": 0.5},
        {"alpha": 1.0,  "beta": 0.5, "gamma": 1.0,  "delta": 0.5},
        {"alpha": 0.25, "beta": 0.5, "gamma": 0.25, "delta": 0.5},
        {"alpha": 0.75, "beta": 0.5, "gamma": 0.75, "delta": 0.5},
        {"alpha": 0.5,  "beta": 0.5, "gamma": 1.0,  "delta": 0.5},
    ]
    n_env = len(params)
    n_traj = _traj_per_env(total, n_env)

    # 256 output time steps as per paper ("256-point meshes for 1-D cases")
    time_horizon = 128.0   # total physical time
    dt = time_horizon / 256.0

    print(f"[LV] generating {n_traj} traj × {n_env} envs = {n_traj * n_env} total")

    dataset = LotkaVolterraDataset(
        num_traj_per_env=n_traj,
        time_horizon=time_horizon,
        params=params,
        dt=dt,
        method="RK45",
        group="train",
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=workers, shuffle=False)

    all_states = []
    for i, sample in enumerate(loader):
        # sample['state']: [1, 2, T]
        all_states.append(sample["state"].squeeze(0))  # [2, T]
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(dataset)}")

    data = torch.stack(all_states, dim=0)  # [n_traj * n_env, 2, T]
    T = data.shape[2]
    data = data.view(n_traj, n_env, 2, T)

    out_path = output_dir / f"lv_{n_samples}.pt"
    torch.save(data, out_path)
    print(f"[LV] saved {tuple(data.shape)} → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Belousov-Zhabotinsky (1-D, 3 processes, 256 spatial points)
# ---------------------------------------------------------------------------

def generate_bz(n_samples: int, output_dir: Path, workers: int) -> Path:
    n_train, n_val, n_test, total = _split_sizes(n_samples)

    params = default_bz_params()   # 4 environments
    n_env = len(params)
    n_traj = _traj_per_env(total, n_env)

    # 256 spatial points, 20 time snapshots (t ∈ [0, 20], dt_eval=1.0)
    time_horizon = 20.0
    dt_eval = 1.0
    n_points = 256

    print(f"[BZ] generating {n_traj} traj × {n_env} envs = {n_traj * n_env} total "
          f"(spatial: {n_points} pts,  time: {int(time_horizon / dt_eval)} steps)")

    dataset = BZDataset(
        num_traj_per_env=n_traj,
        n_points=n_points,
        time_horizon=time_horizon,
        dt_eval=dt_eval,
        params=params,
        method="RK45",
        group="train",
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=workers, shuffle=False)

    all_states = []
    for i, sample in enumerate(loader):
        # sample['state']: [1, 3, T, n_points]
        all_states.append(sample["state"].squeeze(0))  # [3, T, n_points]
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(dataset)}")

    data = torch.stack(all_states, dim=0)  # [n_traj * n_env, 3, T, n_points]
    n_proc, T, N = data.shape[1], data.shape[2], data.shape[3]
    data = data.view(n_traj, n_env, n_proc, T, N)

    out_path = output_dir / f"bz_{n_samples}.pt"
    torch.save(data, out_path)
    print(f"[BZ] saved {tuple(data.shape)} → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Split info helper — print the ratios to feed to build_grayscott_dataloader
# ---------------------------------------------------------------------------

def print_split_ratios(n_samples: int):
    n_train, n_val, n_test, total = _split_sizes(n_samples)
    train_r = n_train / total
    val_r   = n_val   / total
    print(
        f"\n[Split ratios for n_samples={n_samples}]\n"
        f"  --train_ratio {train_r:.4f}  (≈{n_train} train)\n"
        f"  --val_ratio   {val_r:.4f}  (≈{n_val} val)\n"
        f"  (remainder    {1 - train_r - val_r:.4f}  ≈{n_test} test)\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate FlowCoupledPDE benchmark datasets")
    p.add_argument("--system",     required=True, choices=["gs", "lv", "bz"],
                   help="Which PDE system to generate")
    p.add_argument("--n_samples",  required=True, type=int, choices=[512, 1024],
                   help="Number of TRAINING trajectories (paper options: 512 or 1024)")
    p.add_argument("--output_dir", default="./datasets",
                   help="Directory where the .pt file will be saved")
    p.add_argument("--workers",    default=4, type=int,
                   help="DataLoader workers for parallel trajectory generation")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generators = {"gs": generate_gs, "lv": generate_lv, "bz": generate_bz}
    out_path = generators[args.system](args.n_samples, out_dir, args.workers)

    print_split_ratios(args.n_samples)
    print(f"Done. Dataset saved to: {out_path}")


if __name__ == "__main__":
    main()
