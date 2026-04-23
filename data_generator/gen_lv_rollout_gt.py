"""
Generate multi-step rollout ground truth for the LV test set.

LV: 1-D reaction-diffusion Lotka-Volterra
  du/dt = Du*lap(u) + a*u - b*u*v
  dv/dt = Dv*lap(v) + c*u*v - d*v
  Du=Dv=0.01, a=b=c=d=0.01, dx=1.0, N=256, T=100 per step (tau)

Output: lv_rollout_gt.pt
  shape: [N_test, 2, 11, 256]
  dim 2 (time): [0tau, 1tau, 2tau, ..., 10tau]

Usage (from repo root):
    conda activate pdemeanflow
    python data_generator/gen_lv_rollout_gt.py \
        --data_path /scratch/user/u.kt348068/PDE_data/lv_512.pt \
        --out_path  /scratch/user/u.kt348068/PDE_data/lv_rollout_gt.pt \
        --train_ratio 0.6305 --val_ratio 0.1232 \
        --workers 8
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

_DU      = 0.01
_DV      = 0.01
_A       = 0.01
_B       = 0.01
_C       = 0.01
_D       = 0.01
_N       = 256
_DX      = 1.0
_T_PRED  = 100.0
_N_STEPS = 10


def _lv_rhs(t, uv_flat, n, dx, Du, Dv, a, b, c, d):
    u = uv_flat[:n]
    v = uv_flat[n:]

    def lap(f):
        return (np.roll(f, 1) - 2.0 * f + np.roll(f, -1)) / dx**2

    return np.concatenate([Du * lap(u) + a * u - b * u * v,
                           Dv * lap(v) + c * u * v - d * v])


def _simulate_from_ic(args):
    idx, uv_ic = args   # uv_ic: [2, 256] float32
    uv0    = np.concatenate([uv_ic[0], uv_ic[1]]).astype(np.float64)
    t_end  = _N_STEPS * _T_PRED
    t_eval = [k * _T_PRED for k in range(1, _N_STEPS + 1)]

    res = solve_ivp(
        _lv_rhs, (0.0, t_end), uv0,
        method='RK45',
        t_eval=t_eval,
        args=(_N, _DX, _DU, _DV, _A, _B, _C, _D),
        rtol=1e-5, atol=1e-6,
        dense_output=False,
    )
    if not res.success:
        print(f"  [Warning] traj {idx} failed: {res.message}", flush=True)

    # res.y: [2*N, N_STEPS] → [2, N_STEPS, 256]
    snapshots = np.stack([
        np.stack([res.y[:_N, s], res.y[_N:, s]], axis=0)
        for s in range(_N_STEPS)
    ], axis=1).astype(np.float32)

    return idx, snapshots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   required=True)
    p.add_argument("--out_path",    required=True)
    p.add_argument("--train_ratio", type=float, default=0.6305)
    p.add_argument("--val_ratio",   type=float, default=0.1232)
    p.add_argument("--workers",     type=int,   default=8)
    args = p.parse_args()

    print(f"Loading {args.data_path} …")
    data = torch.load(args.data_path, map_location="cpu").float()
    # shape: [N_traj, 1, 2, 2, 256]
    print(f"  shape: {tuple(data.shape)}")

    n_traj  = data.shape[0]
    n_train = int(n_traj * args.train_ratio)
    n_val   = int(n_traj * args.val_ratio)
    test_data = data[n_train + n_val:]   # [N_test, 1, 2, 2, 256]
    n_test    = test_data.shape[0]
    print(f"  test set: {n_test} trajectories (indices {n_train+n_val}..{n_traj-1})")

    ics     = test_data[:, 0, :, 0, :].numpy()   # [N_test, 2, 256]
    gt_1tau = test_data[:, 0, :, 1, :].numpy()   # [N_test, 2, 256]

    job_args = [(i, ics[i]) for i in range(n_test)]
    print(f"Simulating {n_test} test ICs × {_N_STEPS} steps "
          f"(T_pred={_T_PRED}) with {args.workers} workers …")

    results = [None] * n_test
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for done, (i, snap) in enumerate(
                    pool.imap_unordered(_simulate_from_ic, job_args, chunksize=4)):
                results[i] = snap
                if (done + 1) % 50 == 0:
                    print(f"  … {done+1}/{n_test}", flush=True)
    else:
        for i, a in enumerate(job_args):
            _, snap = _simulate_from_ic(a)
            results[i] = snap
            if (i + 1) % 50 == 0:
                print(f"  … {i+1}/{n_test}", flush=True)

    sim = torch.from_numpy(np.stack(results, axis=0))   # [N_test, 2, N_STEPS, 256]

    sim_1tau = sim[:, :, 0, :]
    err = (sim_1tau - torch.from_numpy(gt_1tau)).abs().mean().item()
    print(f"\nVerification — mean |sim_1τ − existing_1τ| = {err:.6f}  "
          f"(should be ~0 if IC loaded correctly)")

    ic_t  = torch.from_numpy(ics).unsqueeze(2)       # [N_test, 2, 1, 256]
    gt1_t = torch.from_numpy(gt_1tau).unsqueeze(2)   # [N_test, 2, 1, 256]
    # ic(0τ) + existing_1τ + sim[2τ..10τ]
    rollout = torch.cat([ic_t, gt1_t, sim[:, :, 1:, :]], dim=2)
    # shape: [N_test, 2, 11, 256]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rollout, out_path)
    print(f"\nSaved rollout GT → {out_path}")
    print(f"  shape: {tuple(rollout.shape)}")
    print(f"  time axis: [0τ..10τ]  (τ = {_T_PRED} time units)")


if __name__ == "__main__":
    main()
