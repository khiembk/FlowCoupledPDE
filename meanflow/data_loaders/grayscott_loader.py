import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class GrayScottConfig:
    data_path: str                  # path to gs.pt
    split: str = "train"           # train / val / test
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    horizon: int = 1               # predict t -> t+horizon
    flatten_env: bool = True       # treat each environment as a separate sample stream
    normalize: bool = False        # optional per-channel normalization
    return_env: bool = False       # optionally return env_id
    return_time: bool = False      # optionally return t_in, t_out


class GrayScottCoupledDataset(Dataset):
    """
    Expected source tensor from dynamicalsystems_dataset:
        gs.pt with shape [N_traj, N_env, 2, T, H, W]

    Returned item:
        z1_1, z1_2, z0_1, z0_2
    where
        z1_* = state at t_in
        z0_* = state at t_out = t_in + horizon

    Shapes of each returned tensor:
        [1, H, W]
    """

    def __init__(self, cfg: GrayScottConfig):
        super().__init__()
        self.cfg = cfg

        if not os.path.exists(cfg.data_path):
            raise FileNotFoundError(f"Could not find dataset file: {cfg.data_path}")

        data = torch.load(cfg.data_path, map_location="cpu").float()

        if data.ndim != 6:
            raise ValueError(
                f"Expected Gray-Scott tensor with 6 dims [N_traj, N_env, 2, T, H, W], got {tuple(data.shape)}"
            )

        n_traj, n_env, n_chan, T, H, W = data.shape
        if n_chan != 2:
            raise ValueError(f"Expected 2 channels for Gray-Scott, got {n_chan}")
        if cfg.horizon < 1 or cfg.horizon >= T:
            raise ValueError(f"horizon must be in [1, {T-1}], got {cfg.horizon}")

        # Split on trajectory dimension
        n_train = int(n_traj * cfg.train_ratio)
        n_val = int(n_traj * cfg.val_ratio)
        n_test = n_traj - n_train - n_val

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

        # Optional normalization per channel, computed on this split
        self.normalize = cfg.normalize
        if self.normalize:
            # stats over [traj, env, time, h, w], separately per channel
            x = self.data.permute(2, 0, 1, 3, 4, 5).reshape(2, -1)
            self.mean = x.mean(dim=1).view(2, 1, 1)
            self.std = x.std(dim=1).clamp_min(1e-6).view(2, 1, 1)
        else:
            self.mean = None
            self.std = None

        self.num_time_pairs = self.T - cfg.horizon

        if cfg.flatten_env:
            self.length = self.n_traj * self.n_env * self.num_time_pairs
        else:
            self.length = self.n_traj * self.num_time_pairs

    def __len__(self):
        return self.length

    def _normalize_pair(self, x_t: torch.Tensor, x_tp: torch.Tensor):
        if not self.normalize:
            return x_t, x_tp
        x_t = (x_t - self.mean) / self.std
        x_tp = (x_tp - self.mean) / self.std
        return x_t, x_tp

    def __getitem__(self, idx):
        """
        If flatten_env=True:
            idx maps to (traj_id, env_id, t_in)
        else:
            env dimension is kept inside the sample, which is usually not what you want
            for your current coupled-PDE training, so flatten_env=True is recommended.
        """
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

        # x_t, x_tp shapes: [2, H, W]
        x_t = self.data[traj_id, env_id, :, t_in]
        x_tp = self.data[traj_id, env_id, :, t_out]

        x_t, x_tp = self._normalize_pair(x_t, x_tp)

        # Split the two coupled species into two processes
        z1_1 = x_t[0:1]   # [1, H, W]
        z1_2 = x_t[1:2]   # [1, H, W]
        z0_1 = x_tp[0:1]  # [1, H, W]
        z0_2 = x_tp[1:2]  # [1, H, W]

        out = {
            "z1_1": z1_1,
            "z1_2": z1_2,
            "z0_1": z0_1,
            "z0_2": z0_2,
        }

        if self.cfg.return_env:
            out["env_id"] = torch.tensor(env_id, dtype=torch.long)
        if self.cfg.return_time:
            out["t_in"] = torch.tensor(t_in, dtype=torch.long)
            out["t_out"] = torch.tensor(t_out, dtype=torch.long)

        return out


def grayscott_collate_fn(batch):
    return (
        torch.stack([b["z1_1"] for b in batch], dim=0),
        torch.stack([b["z1_2"] for b in batch], dim=0),
        torch.stack([b["z0_1"] for b in batch], dim=0),
        torch.stack([b["z0_2"] for b in batch], dim=0),
    )


def build_grayscott_dataloader(
    data_path: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    horizon: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: Optional[bool] = None,
    normalize: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    cfg = GrayScottConfig(
        data_path=data_path,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        horizon=horizon,
        flatten_env=True,
        normalize=normalize,
    )
    dataset = GrayScottCoupledDataset(cfg)

    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=grayscott_collate_fn,
    )
    return dataset, loader