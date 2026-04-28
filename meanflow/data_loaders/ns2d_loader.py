import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class NS2DConfig:
    data_path: str
    split: str = "train"
    train_ratio: float = 0.6667
    val_ratio: float = 0.1667
    horizon: int = 1
    flatten_env: bool = True


class NS2DDataset(Dataset):
    """
    PDEArena NavierStokes2D dataset.
    Expected tensor shape: [N_traj, N_env, 3, T, H, W]
      channels: 0=u (vorticity/scalar), 1=vx, 2=vy

    Returns (z1_1, z1_2, z1_3, z0_1, z0_2, z0_3) each of shape [1, H, W].
    """

    def __init__(self, cfg: NS2DConfig):
        super().__init__()
        self.cfg = cfg

        if not os.path.exists(cfg.data_path):
            raise FileNotFoundError(f"Could not find dataset file: {cfg.data_path}")

        data = torch.load(cfg.data_path, map_location="cpu").float()

        if data.ndim != 6:
            raise ValueError(
                f"Expected NS2D tensor with 6 dims [N_traj, N_env, 3, T, H, W], got {tuple(data.shape)}"
            )

        n_traj, n_env, n_chan, T, H, W = data.shape
        if n_chan != 3:
            raise ValueError(f"Expected 3 channels for NS2D, got {n_chan}")
        if cfg.horizon < 1 or cfg.horizon >= T:
            raise ValueError(f"horizon must be in [1, {T-1}], got {cfg.horizon}")

        n_train = int(n_traj * cfg.train_ratio)
        n_val = int(n_traj * cfg.val_ratio)

        if cfg.split == "train":
            data = data[:n_train]
        elif cfg.split == "val":
            data = data[n_train:n_train + n_val]
        elif cfg.split == "test":
            data = data[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {cfg.split}")

        self.data = data.contiguous()
        self.n_traj, self.n_env, _, self.T, self.H, self.W = self.data.shape
        self.num_time_pairs = self.T - cfg.horizon

        if cfg.flatten_env:
            self.length = self.n_traj * self.n_env * self.num_time_pairs
        else:
            self.length = self.n_traj * self.num_time_pairs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.cfg.flatten_env:
            per_traj = self.n_env * self.num_time_pairs
            traj_id = idx // per_traj
            rem = idx % per_traj
            env_id = rem // self.num_time_pairs
            t_in = rem % self.num_time_pairs
        else:
            traj_id = idx // self.num_time_pairs
            env_id = 0
            t_in = idx % self.num_time_pairs

        t_out = t_in + self.cfg.horizon

        x_t  = self.data[traj_id, env_id, :, t_in]   # [3, H, W]
        x_tp = self.data[traj_id, env_id, :, t_out]  # [3, H, W]

        return (
            x_t[0:1],   # z1_1  [1, H, W]
            x_t[1:2],   # z1_2  [1, H, W]
            x_t[2:3],   # z1_3  [1, H, W]
            x_tp[0:1],  # z0_1  [1, H, W]
            x_tp[1:2],  # z0_2  [1, H, W]
            x_tp[2:3],  # z0_3  [1, H, W]
        )


def ns2d_collate_fn(batch):
    return tuple(torch.stack([b[i] for b in batch], dim=0) for i in range(6))


def build_ns2d_dataloader(
    data_path: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    horizon: int = 1,
    train_ratio: float = 0.6667,
    val_ratio: float = 0.1667,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    cfg = NS2DConfig(
        data_path=data_path,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        horizon=horizon,
        flatten_env=True,
    )
    dataset = NS2DDataset(cfg)

    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=ns2d_collate_fn,
    )
    return dataset, loader
