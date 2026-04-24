"""
Dataloader for the gray_scott_reaction_diffusion dataset from The Well.

HDF5 layout (per file):
    t0_fields/A  : (n_traj=160, n_time=1001, H=128, W=128)  float32
    t0_fields/B  : (n_traj=160, n_time=1001, H=128, W=128)  float32
    scalars/F, scalars/k : scalar parameters

Split is given by the Well's pre-defined subdirectories (train/valid/test).
No train_ratio splitting — use the Well's official split.

Return format (same as grayscott_loader):
    z1_1, z1_2, z0_1, z0_2  each  [1, H, W]
    z1_* = state at t_in  (source, earlier)
    z0_* = state at t_in + horizon  (target, later)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class GSWellConfig:
    data_dir: str           # path to the dataset root (contains train/valid/test subdirs)
    split: str = "train"   # train / valid / test
    horizon: int = 1       # predict t -> t + horizon
    time_stride: int = 10  # stride over time axis to reduce dataset size (1 = use all pairs)
    resolution: int = 64   # downsample spatial dims to this size (None = keep original 128)
    normalize: bool = False


class GSWellDataset(Dataset):
    """
    Lazily reads gray_scott Well HDF5 files.

    Index structure:
        samples = [(file_idx, traj_idx, t_in), ...]

    HDF5 files are opened on first access per worker to be fork-safe.
    """

    def __init__(self, cfg: GSWellConfig):
        self.cfg = cfg

        split_dir = Path(cfg.data_dir) / "data" / cfg.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.files: List[Path] = sorted(split_dir.glob("*.hdf5"))
        if not self.files:
            raise FileNotFoundError(f"No HDF5 files found in {split_dir}")

        # Read metadata from first file (all files share same shape)
        with h5py.File(self.files[0], "r") as f:
            A = f["t0_fields"]["A"]
            self.n_traj_per_file: int = A.shape[0]
            self.n_time: int = A.shape[1]
            self.H: int = A.shape[2]
            self.W: int = A.shape[3]

        # Build flat index: (file_idx, traj_idx, t_in)
        n_files = len(self.files)
        valid_t = range(0, self.n_time - cfg.horizon, cfg.time_stride)
        self.samples: List[Tuple[int, int, int]] = [
            (fi, ti, t)
            for fi in range(n_files)
            for ti in range(self.n_traj_per_file)
            for t in valid_t
        ]

        # Per-worker open file handles (populated lazily in __getitem__)
        self._handles: Dict[int, h5py.File] = {}

        # Normalization stats (computed from first file's first trajectory if needed)
        self.normalize = cfg.normalize
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        if cfg.normalize:
            self._compute_norm_stats()

    def _compute_norm_stats(self):
        # Approximate stats from first 10 trajectories of all train files
        A_vals, B_vals = [], []
        for fp in self.files:
            with h5py.File(fp, "r") as f:
                n = min(10, self.n_traj_per_file)
                A_vals.append(f["t0_fields"]["A"][:n].ravel())
                B_vals.append(f["t0_fields"]["B"][:n].ravel())
        A_all = np.concatenate(A_vals)
        B_all = np.concatenate(B_vals)
        mean = torch.tensor([A_all.mean(), B_all.mean()], dtype=torch.float32).view(2, 1, 1)
        std  = torch.tensor([A_all.std(),  B_all.std()],  dtype=torch.float32).clamp_min(1e-6).view(2, 1, 1)
        self.mean = mean
        self.std  = std

    def _get_handle(self, file_idx: int) -> h5py.File:
        pid = os.getpid()
        key = (pid, file_idx)
        if key not in self._handles:
            self._handles[key] = h5py.File(self.files[file_idx], "r")
        return self._handles[key]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fi, ti, t_in = self.samples[idx]
        t_out = t_in + self.cfg.horizon

        h = self._get_handle(fi)
        a_in  = torch.from_numpy(h["t0_fields"]["A"][ti, t_in].copy())   # [H, W]
        b_in  = torch.from_numpy(h["t0_fields"]["B"][ti, t_in].copy())   # [H, W]
        a_out = torch.from_numpy(h["t0_fields"]["A"][ti, t_out].copy())  # [H, W]
        b_out = torch.from_numpy(h["t0_fields"]["B"][ti, t_out].copy())  # [H, W]

        x_t  = torch.stack([a_in,  b_in],  dim=0)  # [2, H, W]
        x_tp = torch.stack([a_out, b_out], dim=0)  # [2, H, W]

        res = self.cfg.resolution
        if res is not None and res != self.H:
            x_t  = F.interpolate(x_t.unsqueeze(0),  size=(res, res), mode="bilinear", align_corners=False).squeeze(0)
            x_tp = F.interpolate(x_tp.unsqueeze(0), size=(res, res), mode="bilinear", align_corners=False).squeeze(0)

        if self.normalize and self.mean is not None:
            x_t  = (x_t  - self.mean) / self.std
            x_tp = (x_tp - self.mean) / self.std

        return {
            "z1_1": x_t[0:1],   # [1, res, res]
            "z1_2": x_t[1:2],   # [1, res, res]
            "z0_1": x_tp[0:1],  # [1, res, res]
            "z0_2": x_tp[1:2],  # [1, res, res]
        }


def gs_well_collate_fn(batch):
    return (
        torch.stack([b["z1_1"] for b in batch]),
        torch.stack([b["z1_2"] for b in batch]),
        torch.stack([b["z0_1"] for b in batch]),
        torch.stack([b["z0_2"] for b in batch]),
    )


def build_gs_well_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    horizon: int = 1,
    time_stride: int = 10,
    resolution: int = 64,
    normalize: bool = False,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[GSWellDataset, DataLoader]:
    cfg = GSWellConfig(
        data_dir=data_dir,
        split=split,
        horizon=horizon,
        time_stride=time_stride,
        resolution=resolution,
        normalize=normalize,
    )
    dataset = GSWellDataset(cfg)

    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=gs_well_collate_fn,
        persistent_workers=(num_workers > 0),
    )
    return dataset, loader
