"""
Data loader for PDEBench 2D Diffusion-Reaction dataset.

HDF5 layout:
    File: 2D_diff_react.h5
    Keys: '0000' ... '0999'  (1000 trajectories)
    Per sample:
        data:  (T=101, H=128, W=128, 2)   channels-last [u, v]
        grid:  (H, W, 2)                  spatial coords (not used)

Split: fixed 800 / 100 / 100  (train / val / test)

Return format (same as grayscott_loader):
    z1_1, z1_2, z0_1, z0_2  each [1, res, res]
    z1_* = state at t_in  (source, earlier)
    z0_* = state at t_in + horizon  (target, later)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Fixed split sizes out of 1000 total trajectories
_N_TRAIN = 800
_N_VAL   = 100
_N_TEST  = 100


@dataclass
class DR2DConfig:
    data_path: str           # path to 2D_diff_react.h5
    split: str = "train"    # train / val / test
    horizon: int = 1        # predict t -> t + horizon
    time_stride: int = 1    # stride over time axis
    resolution: int = 64    # downsample spatial dims (None = keep 128)
    normalize: bool = False


class DR2DDataset(Dataset):
    """
    Lazy HDF5 reader for PDEBench 2D Diffusion-Reaction.
    Fork-safe: file handles opened per worker PID.
    """

    def __init__(self, cfg: DR2DConfig):
        self.cfg = cfg
        self.data_path = cfg.data_path

        if not Path(cfg.data_path).exists():
            raise FileNotFoundError(f"Dataset not found: {cfg.data_path}")

        # Read metadata from file (fast — just header)
        with h5py.File(cfg.data_path, "r") as f:
            all_keys = sorted(f.keys())
            T = f[all_keys[0]]["data"].shape[0]   # 101
            H = f[all_keys[0]]["data"].shape[1]   # 128
            W = f[all_keys[0]]["data"].shape[2]   # 128

        self.T = T
        self.H = H
        self.W = W

        # Assign split keys
        if cfg.split == "train":
            self.keys = all_keys[:_N_TRAIN]
        elif cfg.split == "val":
            self.keys = all_keys[_N_TRAIN:_N_TRAIN + _N_VAL]
        elif cfg.split == "test":
            self.keys = all_keys[_N_TRAIN + _N_VAL:]
        else:
            raise ValueError(f"Unknown split: {cfg.split!r}")

        # Build flat index: (key_idx, t_in)
        valid_t = range(0, T - cfg.horizon, cfg.time_stride)
        self.samples: List[Tuple[int, int]] = [
            (ki, t)
            for ki in range(len(self.keys))
            for t in valid_t
        ]

        # Per-worker file handles
        self._handles: Dict[int, h5py.File] = {}

    def _get_handle(self) -> h5py.File:
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = h5py.File(self.data_path, "r")
        return self._handles[pid]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ki, t_in = self.samples[idx]
        t_out = t_in + self.cfg.horizon

        f = self._get_handle()
        key = self.keys[ki]
        # data shape: (T, H, W, 2) — channels last
        x_t  = torch.from_numpy(f[key]["data"][t_in].copy())   # [H, W, 2]
        x_tp = torch.from_numpy(f[key]["data"][t_out].copy())  # [H, W, 2]

        # channels last -> channels first: [2, H, W]
        x_t  = x_t.permute(2, 0, 1).float()
        x_tp = x_tp.permute(2, 0, 1).float()

        res = self.cfg.resolution
        if res is not None and res != self.H:
            x_t  = F.interpolate(x_t.unsqueeze(0),  size=(res, res), mode="bilinear", align_corners=False).squeeze(0)
            x_tp = F.interpolate(x_tp.unsqueeze(0), size=(res, res), mode="bilinear", align_corners=False).squeeze(0)

        return {
            "z1_1": x_t[0:1],   # [1, res, res]
            "z1_2": x_t[1:2],
            "z0_1": x_tp[0:1],
            "z0_2": x_tp[1:2],
        }


def dr2d_collate_fn(batch):
    return (
        torch.stack([b["z1_1"] for b in batch]),
        torch.stack([b["z1_2"] for b in batch]),
        torch.stack([b["z0_1"] for b in batch]),
        torch.stack([b["z0_2"] for b in batch]),
    )


def build_dr2d_dataloader(
    data_path: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    horizon: int = 1,
    time_stride: int = 1,
    resolution: int = 64,
    normalize: bool = False,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[DR2DDataset, DataLoader]:
    cfg = DR2DConfig(
        data_path=data_path,
        split=split,
        horizon=horizon,
        time_stride=time_stride,
        resolution=resolution,
        normalize=normalize,
    )
    dataset = DR2DDataset(cfg)

    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=dr2d_collate_fn,
        persistent_workers=(num_workers > 0),
    )
    return dataset, loader
