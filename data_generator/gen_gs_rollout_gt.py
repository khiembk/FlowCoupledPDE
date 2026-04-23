"""
Generate multi-step rollout ground truth for the GS test set.

The existing gs_512.pt has T=2 snapshots per trajectory:
  index 0  — IC  (after T_warmup=200 burn-in)
  index 1  — 1τ  (T_warmup + T_pred=20)

τ = T_pred = 20 time units.  This script re-runs the PDE from each
test IC and records states at 2τ, 3τ, 4τ, 5τ (i.e. at T_pred × 2,3,4,5).

Output:  gs_rollout_gt.pt
  shape: [N_test, 2, 6, 64, 64]
  dim 2 (time axis): [0τ, 1τ, 2τ, 3τ, 4τ, 5τ]
    0τ  = IC  (already in gs_512.pt[:,:,:,0])
    1τ  = existing endpoint (already in gs_512.pt[:,:,:,1])
    2τ–5τ = newly simulated

Usage (from repo root or data_generator/):
    conda activate pdemeanflow
    python data_generator/gen_gs_rollout_gt.py \
        --data_path /scratch/user/u.kt348068/PDE_data/gs_512.pt \
        --out_path  /scratch/user/u.kt348068/PDE_data/gs_rollout_gt.pt \
        --train_ratio 0.6305 --val_ratio 0.1232 \
        --workers 8
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

# ── GS PDE parameters (must match gen_data.py) ─────────────────────────────
_DU   = 0.12
_DV   = 0.06
_F    = 0.054
_K    = 0.063
_SIZE = 64
_DX   = 1.0
_T_PRED = 20.0


def _gs_rhs(t, uv_flat, size, dx, Du, Dv, F, k):
    N = size * size
    u = uv_flat[:N].reshape(size, size)
    v = uv_flat[N:].reshape(size, size)

    def lap(a):
        nz = np.roll(a,  1, 0);  pz = np.roll(a, -1, 0)
        zn = np.roll(a,  1, 1);  zp = np.roll(a, -1, 1)
        nn = np.roll(nz, 1, 1); np_ = np.roll(nz, -1, 1)
        pn = np.roll(pz, 1, 1);  pp = np.roll(pz, -1, 1)
        return (-3*a + 0.5*(nz+pz+zn+zp) + 0.25*(nn+np_+pn+pp)) / dx**2

    uvv  = u * v * v
    dudt = Du * lap(u) - uvv + _F * (1.0 - u)
    dvdt = Dv * lap(v) + uvv - (_F + _K) * v
    return np.concatenate([dudt.ravel(), dvdt.ravel()])


def _simulate_from_ic(args):
    """Run GS from a given IC for n_steps * T_pred time units."""
    idx, uv_ic, n_steps = args   # uv_ic: [2, H, W] numpy float32

    uv0 = np.concatenate([uv_ic[0].ravel(), uv_ic[1].ravel()]).astype(np.float64)
    t_end  = n_steps * _T_PRED
    t_eval = [k * _T_PRED for k in range(1, n_steps + 1)]

    res = solve_ivp(
        _gs_rhs,
        (0.0, t_end),
        uv0,
        method='RK45',
        t_eval=t_eval,
        args=(_SIZE, _DX, _DU, _DV, _F, _K),
        rtol=1e-5, atol=1e-6,
        dense_output=False,
    )
    if not res.success:
        print(f"  [Warning] traj {idx} failed: {res.message}", flush=True)

    N = _SIZE * _SIZE
    snapshots = np.stack([
        np.stack([res.y[:N, s].reshape(_SIZE, _SIZE),
                  res.y[N:, s].reshape(_SIZE, _SIZE)], axis=0)
        for s in range(n_steps)
    ], axis=1).astype(np.float32)   # [2, n_steps, H, W]

    return idx, snapshots


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   required=True)
    p.add_argument("--out_path",    required=True)
    p.add_argument("--train_ratio", type=float, default=0.6305)
    p.add_argument("--val_ratio",   type=float, default=0.1232)
    p.add_argument("--n_steps",     type=int,   default=10)
    p.add_argument("--workers",     type=int,   default=8)
    args = p.parse_args()

    print(f"Loading {args.data_path} …")
    data = torch.load(args.data_path, map_location="cpu").float()
    # data shape: [N_traj, 1, 2, 2, 64, 64]
    print(f"  shape: {tuple(data.shape)}")

    n_traj = data.shape[0]
    n_train = int(n_traj * args.train_ratio)
    n_val   = int(n_traj * args.val_ratio)
    test_data = data[n_train + n_val:]          # [N_test, 1, 2, 2, H, W]
    n_test = test_data.shape[0]
    print(f"  test set: {n_test} trajectories (indices {n_train+n_val}..{n_traj-1})")

    # IC: snapshot index 0, env index 0 → [N_test, 2, H, W]
    ics = test_data[:, 0, :, 0, :, :].numpy()  # [N_test, 2, 64, 64]
    # Existing 1τ endpoint → [N_test, 2, H, W]
    gt_1tau = test_data[:, 0, :, 1, :, :].numpy()

    n_steps  = args.n_steps
    job_args = [(i, ics[i], n_steps) for i in range(n_test)]

    print(f"Simulating {n_test} test ICs × {n_steps} steps "
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

    sim = torch.from_numpy(np.stack(results, axis=0))  # [N_test, 2, n_steps, H, W]

    sim_1tau = sim[:, :, 0, :, :]
    err = (sim_1tau - torch.from_numpy(gt_1tau)).abs().mean().item()
    print(f"\nVerification — mean |sim_1τ − existing_1τ| = {err:.6f}  "
          f"(should be ~0 if IC loaded correctly)")

    ic_t  = torch.from_numpy(ics).unsqueeze(2)       # [N_test, 2, 1, H, W]
    gt1_t = torch.from_numpy(gt_1tau).unsqueeze(2)   # [N_test, 2, 1, H, W]
    rollout = torch.cat([ic_t, gt1_t, sim[:, :, 1:, :, :]], dim=2)
    # shape: [N_test, 2, n_steps+1, H, W]  — [0τ, 1τ, ..., n_steps*τ]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rollout, out_path)
    print(f"\nSaved rollout GT → {out_path}")
    print(f"  shape: {tuple(rollout.shape)}")
    print(f"  time axis: [0τ..{n_steps}τ]  (τ = T_pred = {_T_PRED} time units)")


if __name__ == "__main__":
    main()
