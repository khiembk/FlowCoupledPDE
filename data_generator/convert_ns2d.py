"""
Convert PDEArena NavierStokes2D HDF5 files to a single .pt tensor.

Source: multiple HDF5 files in a directory, each containing:
  train/  →  u, vx, vy  each [N_traj, 14, 128, 128]
  valid/  →  u, vx, vy  each [N_traj, 14, 128, 128]
  test/   →  u, vx, vy  each [N_traj, 14, 128, 128]

Output: ns2d.pt  shape  [N_total, 1, 3, 14, res, res]
  axis0: trajectories (train first, then valid, then test)
  axis1: N_env = 1
  axis2: channels  (0=u, 1=vx, 2=vy)
  axis3: time  (14 steps)
  axis4/5: spatial (res×res, downsampled from 128×128)

Split ratios for 5200 train + 1300 valid + 1300 test = 7800 total:
  train_ratio = 0.6667   val_ratio = 0.1667

Usage:
    conda activate flow
    python convert_ns2d.py \
        --input_dir /scratch/user/u.kt348068/PDE_data/PDEArena/NavierStokes2D \
        --output /scratch/user/u.kt348068/PDE_data/ns2d.pt \
        --resolution 64
"""
import argparse
import glob
import os

import h5py
import torch
import torch.nn.functional as F


def downsample(tensor, res):
    """Downsample [N, T, H, W] → [N, T, res, res] via bilinear interpolation."""
    N, T, H, W = tensor.shape
    if H == res and W == res:
        return tensor
    x = tensor.reshape(N * T, 1, H, W)
    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return x.reshape(N, T, res, res)


def load_split(files, split_key, resolution):
    """Load and stack all HDF5 files for a given split key ('train'/'valid'/'test')."""
    chunks = []
    for path in sorted(files):
        with h5py.File(path, "r") as f:
            grp = f[split_key]
            u  = torch.tensor(grp["u"][:],  dtype=torch.float32)   # [N, T, H, W]
            vx = torch.tensor(grp["vx"][:], dtype=torch.float32)
            vy = torch.tensor(grp["vy"][:], dtype=torch.float32)

        # Downsample spatial dims
        u  = downsample(u,  resolution)
        vx = downsample(vx, resolution)
        vy = downsample(vy, resolution)

        # Stack channels: [N, 3, T, res, res]
        sample = torch.stack([u, vx, vy], dim=1)
        # Add N_env dim: [N, 1, 3, T, res, res]
        sample = sample.unsqueeze(1)
        # Move time to axis 3: [N, 1, 3, T, res, res] — already correct since
        # stack gave [N, 3, T, res, res] and we unsqueeze at dim 1
        chunks.append(sample)

    return torch.cat(chunks, dim=0)  # [N_total, 1, 3, T, res, res]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args()

    res = args.resolution
    input_dir = args.input_dir

    train_files = sorted(glob.glob(os.path.join(input_dir, "NavierStokes2D_train_*.h5")))
    valid_files = sorted(glob.glob(os.path.join(input_dir, "NavierStokes2D_valid_*.h5")))
    test_files  = sorted(glob.glob(os.path.join(input_dir, "NavierStokes2D_test_*.h5")))

    print(f"Found {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test files")

    print("Loading train split...")
    train_data = load_split(train_files, "train", res)
    print(f"  train shape: {tuple(train_data.shape)}")

    print("Loading valid split...")
    valid_data = load_split(valid_files, "valid", res)
    print(f"  valid shape: {tuple(valid_data.shape)}")

    print("Loading test split...")
    test_data = load_split(test_files, "test", res)
    print(f"  test shape: {tuple(test_data.shape)}")

    # Concatenate: train first so default split ratios work correctly
    data = torch.cat([train_data, valid_data, test_data], dim=0)
    print(f"Combined shape: {tuple(data.shape)}")

    n_total = data.shape[0]
    n_train = train_data.shape[0]
    n_valid = valid_data.shape[0]
    print(f"Suggested split ratios:")
    print(f"  train_ratio = {n_train/n_total:.4f}  ({n_train}/{n_total})")
    print(f"  val_ratio   = {n_valid/n_total:.4f}  ({n_valid}/{n_total})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"Saving to {args.output} ...")
    torch.save(data, args.output)
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"Done. File size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
