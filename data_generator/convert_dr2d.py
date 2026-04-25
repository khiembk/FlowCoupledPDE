"""
Convert PDEBench 2D Diffusion-Reaction HDF5 to .pt format.

Output shape: [N_traj, 1, 2, T, H, W]  (N_env=1, n_chan=2, T=101, H=64, W=64)
Spatial dims downsampled 128->64 with bilinear interpolation.

Usage (run on login node where h5py/numpy work):
    python data_generator/convert_dr2d.py \
        --input  /scratch/user/u.kt348068/PDE_data/PDEBench/2D_diff_react.h5 \
        --output /scratch/user/u.kt348068/PDE_data/dr2d.pt
"""

import argparse
import h5py
import torch
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to 2D_diff_react.h5")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--res",    type=int, default=64, help="Target spatial resolution")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        keys = sorted(f.keys())
        N = len(keys)
        sample = f[keys[0]]["data"]
        T, H, W, C = sample.shape
        print(f"Found {N} trajectories, shape per traj: T={T}, H={H}, W={W}, C={C}")

        out = torch.zeros(N, 1, C, T, args.res, args.res, dtype=torch.float32)

        for i, key in enumerate(keys):
            if i % 100 == 0:
                print(f"  converting {i}/{N}...")
            # [T, H, W, C] -> [T, C, H, W]
            arr = torch.tensor(f[key]["data"][:], dtype=torch.float32).permute(0, 3, 1, 2)
            if H != args.res:
                arr = F.interpolate(arr, size=(args.res, args.res), mode="bilinear", align_corners=False)
            # [T, C, H, W] -> [1, C, T, H, W]
            out[i, 0] = arr.permute(1, 0, 2, 3)

    print(f"Saving tensor shape {tuple(out.shape)} to {args.output} ...")
    torch.save(out, args.output)
    print("Done.")

if __name__ == "__main__":
    main()
