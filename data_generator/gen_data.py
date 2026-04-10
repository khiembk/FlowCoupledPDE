"""
Unified dataset generation script for FlowCoupledPDE benchmarks.

Generates datasets matching the paper's two experimental options:
    Option A  --n_samples 512   →  512 train + 100 val + 200 test trajectories
    Option B  --n_samples 1024  →  1024 train + 100 val + 200 test trajectories

Supported systems:
    gs   Gray-Scott reaction-diffusion  (2 processes, 2-D 64×64)
    lv   Lotka-Volterra predator-prey   (2 processes, 1-D PDE, 256 spatial pts)
    bz   Belousov-Zhabotinsky           (3 processes, 1-D 256-pt spatial mesh)
    mpf  Multiphase Flow (oil-water)    (2 processes, 2-D 64×64)
    thm  Thermal-Hydro-Mechanical       (5 processes, 2-D 64×64)

Output format per system:
    gs   →  [N_traj, 1,     2, 2,   64,  64]  saved as  gs_<n_samples>.pt
             (1 env, 2 proc, 2 snapshots=t0+tT, 64×64 grid)
    lv   →  [N_traj, 1,     2, 2,  256]       saved as  lv_<n_samples>.pt
             (1 env, 2 proc, 2 snapshots=t0+tT, 256-pt spatial mesh)
    bz   →  [N_traj, 1,     3, 20, 256]       saved as  bz_<n_samples>.pt
             (1 env, 3 proc, 20 snapshots, 256-pt mesh)
    mpf  →  [N_traj, N_env, 2, T, 64,  64]   saved as  mpf_<n_samples>.pt
    thm  →  [N_traj, 1,     5, 12, 64,  64]  saved as  thm_<n_samples>.pt
             (1 env, 5 proc: T/p/ε_xx/ε_yy/ε_xy, 12 non-uniform steps, 64×64)

Generation matches COMPOL paper (arXiv:2501.17296) settings:
  GS:  Single env, D_u=0.12, D_v=0.06, F=0.054, k=0.063, T_pred=20
       Block IC → T_warmup=200 burn-in → predict 20 more time units.
       State at T_warmup serves as the "GRF-like IC"; 64×64 grid.
  LV:  Single env, D_u=D_v=0.01, a=b=c=d=0.01, T=100
       1-D GRF IC (spectral, l=0.1, σ=1), 256-pt mesh, periodic BCs.
       IC→final pair per trajectory (matching GS format).
  MPF: IMPES solver, 3 viscosity environments, 15 timesteps, 64×64 grid.

Usage examples:
    python gen_data.py --system gs   --n_samples 512  --output_dir ./datasets
    python gen_data.py --system mpf  --n_samples 512  --output_dir ./datasets
    python gen_data.py --system thm  --n_samples 512  --output_dir ./datasets
    python gen_data.py --system bz   --n_samples 1024 --output_dir ./datasets --workers 8
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from scipy.integrate import solve_ivp
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

from bz_dataset import BZDataset                                     # local
from multiphase_dataset import MultiphaseFlowDataset, default_mpf_params  # local
from thm_dataset import THMDataset                                         # local


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
# Gray-Scott helpers — module-level so multiprocessing can pickle them
# ---------------------------------------------------------------------------

# COMPOL paper parameters (arXiv:2501.17296, Appendix A)
_GS_PARAMS   = {"D_u": 0.12, "D_v": 0.06, "F": 0.054, "k": 0.063}
_GS_SIZE     = 64
_GS_DX       = 1.0
# Warm-up time: simulate from block IC to T_WARMUP to develop stable patterns.
# The state at T_WARMUP is the "GRF-like IC" referenced in the paper.
# After T=100 the pattern is fully developed (trivial-predictor error ≈ 0.035);
# we use T_WARMUP=200 for safety.
_GS_T_WARMUP = 200.0
# Prediction horizon: the paper's "T=20" is 20 more time units from the
# developed-pattern state, giving trivial-predictor error ≈ 0.035 and
# achievable FNO test error ≈ 0.005-0.006 (matching the paper's 0.0056).
_GS_T_PRED   = 20.0
_GS_N_BLOCK  = 3      # number of random seed blocks
_GS_BLOCK_R  = 6      # block side length (≈ size/10)


def _gs_rhs(t, uv_flat, size, dx, Du, Dv, F, k):
    """Gray-Scott RHS with 9-point isotropic Laplacian and periodic BCs."""
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
    dudt = Du * lap(u) - uvv + F * (1.0 - u)
    dvdt = Dv * lap(v) + uvv - (F + k) * v
    return np.concatenate([dudt.ravel(), dvdt.ravel()])


def _gen_one_gs(args):
    """
    Generate one Gray-Scott trajectory.

    Two-phase approach:
      Phase 1 (burn-in):  start from block IC (u=0.95 background + 3 random
                          blocks of u=0, v=1), simulate for T_WARMUP=200 to
                          develop stable soliton/spot patterns.
      Phase 2 (predict):  simulate T_PRED=20 more time units.
      Store (state at T_WARMUP, state at T_WARMUP+T_PRED) as the IC→final pair.

    The developed-pattern state at T_WARMUP is spatially complex (looks like a
    2-D random field), consistent with the paper's "GRF initial condition"
    description. Trivial-predictor rel-L2 error ≈ 0.035; FNO achieves ≈ 0.005.
    """
    idx, size, dx, Du, Dv, F, k, T_warmup, T_pred = args
    rng = np.random.default_rng(idx)

    # Block IC: background near equilibrium + random seed blocks
    u0 = 0.95 * np.ones((size, size))
    v0 = 0.05 * np.ones((size, size))
    r = _GS_BLOCK_R
    for _ in range(_GS_N_BLOCK):
        N2 = rng.integers(0, size - r, size=2)
        u0[N2[0]:N2[0] + r, N2[1]:N2[1] + r] = 0.0
        v0[N2[0]:N2[0] + r, N2[1]:N2[1] + r] = 1.0
    uv0 = np.concatenate([u0.ravel(), v0.ravel()])

    # Simulate burn-in + prediction in one call; evaluate at [T_warmup, T_warmup+T_pred]
    res = solve_ivp(
        _gs_rhs,
        (0.0, T_warmup + T_pred),
        uv0,
        method='RK45',
        t_eval=[T_warmup, T_warmup + T_pred],
        args=(size, dx, Du, Dv, F, k),
        rtol=1e-5, atol=1e-6,
    )
    if not res.success:
        print(f"  [GS] Warning: traj {idx} failed — {res.message}", flush=True)

    N = size * size
    # shape: [n_chan=2, n_time=2, H, W]
    return np.stack([
        np.stack([res.y[:N, 0].reshape(size, size),   # u at T_warmup  (IC)
                  res.y[:N, 1].reshape(size, size)],  # u at T_warmup+T_pred
                 axis=0),
        np.stack([res.y[N:, 0].reshape(size, size),   # v at T_warmup  (IC)
                  res.y[N:, 1].reshape(size, size)],  # v at T_warmup+T_pred
                 axis=0),
    ], axis=0).astype(np.float32)   # [2, 2, H, W]


# ---------------------------------------------------------------------------
# Gray-Scott (2-D, 64×64, 2 processes)
# ---------------------------------------------------------------------------

def generate_gs(n_samples: int, output_dir: Path, workers: int) -> Path:
    """
    Generate Gray-Scott data matching COMPOL paper (arXiv:2501.17296):

    Key differences from the old (wrong) generation:
      OLD: 3 environments with D_u=0.2097/D_v=0.105/different F,k,
           block-based IC (n_block=3 squares), time_horizon=800/dt_eval=40
           → 20 time steps, horizon=1 predicts consecutive steps.
      NEW: single environment D_u=0.12/D_v=0.06/F=0.054/k=0.063 (paper §A),
           block IC → T_WARMUP=200 burn-in → predict T_PRED=20 more time units.
           The developed-pattern state at T_WARMUP is the paper's "GRF IC"
           (spatially complex, looks like a random field).
           Trivial-predictor rel-L2 ≈ 0.035; FNO achieves ≈ 0.005-0.006.

    Output shape: [N_traj, 1, 2, 2, 64, 64]
                   N_traj   — total trajectories (e.g. 812 for n_samples=512)
                   1        — single environment (fixed PDE params, only IC varies)
                   2        — species (u, v)
                   2        — time snapshots (IC at T_warmup, final at T_warmup+T_pred)
                   64, 64   — spatial grid
    """
    n_train, n_val, n_test, total = _split_sizes(n_samples)

    Du, Dv = _GS_PARAMS['D_u'], _GS_PARAMS['D_v']
    F,  k  = _GS_PARAMS['F'],   _GS_PARAMS['k']

    print(f"[GS] generating {total} trajectories "
          f"({n_train} train / {n_val} val / {n_test} test)")
    print(f"     D_u={Du}  D_v={Dv}  F={F}  k={k}  "
          f"T_warmup={_GS_T_WARMUP}  T_pred={_GS_T_PRED}  "
          f"grid={_GS_SIZE}×{_GS_SIZE}  dx={_GS_DX}")
    print(f"     IC: block (u=0.95/v=0.05 background + {_GS_N_BLOCK} random {_GS_BLOCK_R}×{_GS_BLOCK_R} blocks)")

    job_args = [
        (i, _GS_SIZE, _GS_DX, Du, Dv, F, k, _GS_T_WARMUP, _GS_T_PRED)
        for i in range(total)
    ]

    results = []
    if workers > 1:
        with Pool(workers) as pool:
            for i, state in enumerate(pool.imap(_gen_one_gs, job_args, chunksize=8)):
                results.append(state)
                if (i + 1) % 100 == 0:
                    print(f"  ... {i+1}/{total}", flush=True)
    else:
        for i, a in enumerate(job_args):
            results.append(_gen_one_gs(a))
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{total}", flush=True)

    # [total, 2, 2, H, W] → unsqueeze n_env → [total, 1, 2, 2, H, W]
    data = torch.from_numpy(np.stack(results, axis=0)).unsqueeze(1)

    out_path = output_dir / f"gs_{n_samples}.pt"
    torch.save(data, out_path)
    print(f"[GS] saved {tuple(data.shape)} → {out_path}")
    print(f"     split ratios: train={n_train/total:.4f}  val={n_val/total:.4f}  "
          f"test={(n_test/total):.4f}")
    return out_path


# ---------------------------------------------------------------------------
# Lotka-Volterra helpers — module-level so multiprocessing can pickle them
# ---------------------------------------------------------------------------

# COMPOL paper parameters (arXiv:2501.17296, Appendix A)
# "uniform interaction parameters (a=b=c=d=0.01) and uniform diffusion
#  coefficients (D_u=D_v=0.01)"
_LV_PARAMS  = {"Du": 0.01, "Dv": 0.01, "a": 0.01, "b": 0.01, "c": 0.01, "d": 0.01}
_LV_N       = 256          # 256-point spatial mesh (paper: "256-point meshes for 1-D")
_LV_T_FINAL = 100.0        # physical simulation end time
_LV_GRF_LS  = 0.1          # GRF correlation length (paper: "length-scale l=0.1")
_LV_GRF_AMP = 1.0          # GRF amplitude (paper: "amplitude σ=1")


def _grf_1d(n: int, length_scale: float, rng: np.random.Generator) -> np.ndarray:
    """
    1-D periodic Gaussian Random Field via spectral filtering.
    Returns a zero-mean, unit-std field of shape (n,).
    length_scale is a fraction of the Nyquist frequency.
    """
    noise_ft = np.fft.rfft(rng.standard_normal(n))
    k = np.fft.rfftfreq(n)
    spectral_filter = np.exp(-k ** 2 / (2.0 * length_scale ** 2))
    field = np.fft.irfft(noise_ft * spectral_filter, n=n)
    std = field.std()
    return field / std if std > 1e-10 else field


def _lv_rhs(t, uv_flat, n, dx, Du, Dv, a, b, c, d):
    """
    1-D reaction-diffusion Lotka-Volterra RHS with periodic BCs.
        du/dt = Du * ∇²u + a*u - b*u*v
        dv/dt = Dv * ∇²v + c*u*v - d*v
    Laplacian via 2nd-order central differences on a periodic domain.
    """
    u = uv_flat[:n]
    v = uv_flat[n:]

    def lap(f):
        return (np.roll(f, 1) - 2.0 * f + np.roll(f, -1)) / dx ** 2

    dudt = Du * lap(u) + a * u - b * u * v
    dvdt = Dv * lap(v) + c * u * v - d * v
    return np.concatenate([dudt, dvdt])


def _gen_one_lv(args):
    """Generate one LV trajectory (IC at t=0, final state at t=T)."""
    idx, n, Du, Dv, a, b, c, d, T_final, ls, amp = args
    rng = np.random.default_rng(idx)
    dx  = 1.0   # unit grid spacing (same convention as GS: dx=1, domain=[0,n))
    # With D=0.01 and dx=1: D/dx²=0.01 → diffusion is slow, RK45 is not stiff.
    # GRF at l=0.1 (normalised frequency) → spatial scale ~n*l = 25 grid pts →
    # τ_diff = (25)²/0.01 = 62500 >> T=100, so spatial patterns are preserved.

    # GRF initial conditions around the equilibrium (u*, v*) = (d/c, a/b)
    u_eq = d / c    # = 1.0 for a=b=c=d=0.01
    v_eq = a / b    # = 1.0
    u0 = np.clip(u_eq + amp * _grf_1d(n, ls, rng), 1e-6, None)
    v0 = np.clip(v_eq + amp * _grf_1d(n, ls, rng), 1e-6, None)
    uv0 = np.concatenate([u0, v0])

    res = solve_ivp(
        _lv_rhs, (0.0, T_final), uv0,
        method='RK45',           # not stiff with dx=1, D=0.01 → D/dx²=0.01
        t_eval=[0.0, T_final],
        args=(n, dx, Du, Dv, a, b, c, d),
        rtol=1e-5, atol=1e-6,
    )
    if not res.success:
        print(f"  [LV] Warning: traj {idx} failed — {res.message}", flush=True)

    # shape: [n_proc=2, n_time=2, n_spatial=256]
    return np.stack([
        np.stack([res.y[:n, 0], res.y[:n, 1]], axis=0),   # u at t=0, t=T
        np.stack([res.y[n:, 0], res.y[n:, 1]], axis=0),   # v at t=0, t=T
    ], axis=0).astype(np.float32)   # [2, 2, 256]


# ---------------------------------------------------------------------------
# Lotka-Volterra (1-D reaction-diffusion PDE, 2 processes, 256 spatial pts)
# ---------------------------------------------------------------------------

def generate_lv(n_samples: int, output_dir: Path, workers: int) -> Path:
    """
    Generate 1-D reaction-diffusion Lotka-Volterra data matching COMPOL paper.

    Paper specs (arXiv:2501.17296, Appendix A):
      ∂u/∂t = D_u ∇²u + a·u − b·u·v
      ∂v/∂t = D_v ∇²v + c·u·v − d·v
      D_u=D_v=0.01,  a=b=c=d=0.01,  single environment (fixed params)
      1-D GRF IC on 256-point periodic mesh (length-scale=0.1, amplitude=1)
      Task: IC at t=0 → final solution at t=T=100

    Output shape: [N_traj, 1, 2, 2, 256]
      (N_traj trajectories, 1 env, 2 processes, 2 snapshots t=0+tT, 256 pts)
    """
    n_train, n_val, n_test, total = _split_sizes(n_samples)

    Du = _LV_PARAMS['Du'];  Dv = _LV_PARAMS['Dv']
    a  = _LV_PARAMS['a'];   b  = _LV_PARAMS['b']
    c  = _LV_PARAMS['c'];   d  = _LV_PARAMS['d']

    print(f"[LV] generating {total} trajectories "
          f"({n_train} train / {n_val} val / {n_test} test)")
    print(f"     Du={Du}  Dv={Dv}  a=b=c=d={a}  "
          f"T={_LV_T_FINAL}  n_spatial={_LV_N}  dx=1.0")
    print(f"     IC: 1-D GRF (length_scale={_LV_GRF_LS}, amplitude±{_LV_GRF_AMP})")

    job_args = [
        (i, _LV_N, Du, Dv, a, b, c, d, _LV_T_FINAL, _LV_GRF_LS, _LV_GRF_AMP)
        for i in range(total)
    ]

    results = []
    if workers > 1:
        with Pool(workers) as pool:
            for i, state in enumerate(pool.imap(_gen_one_lv, job_args, chunksize=4)):
                results.append(state)
                if (i + 1) % 100 == 0:
                    print(f"  ... {i+1}/{total}", flush=True)
    else:
        for i, a_arg in enumerate(job_args):
            results.append(_gen_one_lv(a_arg))
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{total}", flush=True)

    # [total, 2, 2, 256] → unsqueeze n_env → [total, 1, 2, 2, 256]
    data = torch.from_numpy(np.stack(results, axis=0)).unsqueeze(1)

    out_path = output_dir / f"lv_{n_samples}.pt"
    torch.save(data, out_path)
    print(f"[LV] saved {tuple(data.shape)} → {out_path}")
    print(f"     split ratios: train={n_train/total:.4f}  val={n_val/total:.4f}  "
          f"test={(n_test/total):.4f}")
    return out_path


# ---------------------------------------------------------------------------
# Belousov-Zhabotinsky (1-D, 3 processes, 256 spatial points)
# ---------------------------------------------------------------------------

def generate_bz(n_samples: int, output_dir: Path, workers: int) -> Path:
    """
    Generate Belousov-Zhabotinsky data matching COMPOL paper (arXiv:2501.17296):

    Equations: ∂u/∂t = ε₁∇²u + u + v − uv − u²
               ∂v/∂t = ε₂∇²v + w − v − uv
               ∂w/∂t = ε₃∇²w + u − w
    Params: ε₁=ε₂=0.01, ε₃=0.005, T_end=0.5, domain [0,1] periodic.
    Solver: ETDRK4 pseudo-spectral, dt=5×10⁻⁴.
    IC:     independent 1-D GRFs (Gaussian kernel, l=0.03), scaled by 0.1.
    Grid:   1024 (fine) → subsampled to 256, 20 snapshots.
    Envs:   N_env=1 (fixed parameters, ICs vary).

    Output shape: [N_traj, 1, 3, 20, 256]
    """
    n_train, n_val, n_test, total = _split_sizes(n_samples)

    print(f"[BZ] generating {total} trajectories "
          f"({n_train} train / {n_val} val / {n_test} test)")
    print(f"     ε₁=ε₂=0.01  ε₃=0.005  T_end=0.5  grid:1024→256  20 snapshots")

    # Train split (seeds 0..n_traj-1)
    train_ds = BZDataset(num_traj=total, group="train")
    loader   = DataLoader(train_ds, batch_size=1, num_workers=workers, shuffle=False)

    all_states = []
    for i, sample in enumerate(loader):
        all_states.append(sample["state"])   # [1, 3, 20, 256] — already float32
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{total}", flush=True)

    # Stack: [total, 3, 20, 256] → unsqueeze N_env → [total, 1, 3, 20, 256]
    data = torch.stack([s.squeeze(0) for s in all_states], dim=0).unsqueeze(1)

    out_path = output_dir / f"bz_{n_samples}.pt"
    torch.save(data, out_path)
    print(f"[BZ] saved {tuple(data.shape)} → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Multiphase Flow (2-D, 64×64, 2 processes: Sw and P)
# ---------------------------------------------------------------------------

def generate_mpf(n_samples: int, output_dir: Path, workers: int) -> Path:
    """
    Generate two-phase (oil-water) flow dataset using IMPES scheme.

    Three environments with different viscosity ratios (μo/μw = 5, 10, 20).
    Each trajectory has a different random log-normal initial saturation.

    Output shape: [n_traj, n_env, 2, T, 64, 64]
        channel 0: Sw  (water saturation)
        channel 1: P   (normalised pore pressure)

    T=15 snapshots matches the COMPOL paper ("each simulated for 15 timesteps").
    """
    n_train, n_val, n_test, total = _split_sizes(n_samples)
    params = default_mpf_params()   # 3 environments
    n_env  = len(params)
    n_traj = _traj_per_env(total, n_env)

    # Time: 15 snapshots at dt_eval = 0.1  (dimensionless time horizon = 1.5)
    # Paper: "15 timesteps over 7.5×10^6 seconds"
    time_horizon = 1.5
    dt_eval      = 0.1
    size         = 64

    print(
        f"[MPF] generating {n_traj} traj × {n_env} envs = {n_traj * n_env} total "
        f"(need {total}: {n_train} train / {n_val} val / {n_test} test)  "
        f"grid={size}×{size}, T={int(time_horizon/dt_eval)} snapshots"
    )

    dataset = MultiphaseFlowDataset(
        num_traj_per_env=n_traj,
        size=size,
        time_horizon=time_horizon,
        dt_eval=dt_eval,
        params=params,
        group="train",
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=workers, shuffle=False)

    all_states = []
    for i, sample in enumerate(loader):
        all_states.append(sample["state"].squeeze(0))   # [2, T, H, W]
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(dataset)}")

    data = torch.stack(all_states, dim=0)                # [n_traj*n_env, 2, T, H, W]
    T_steps, H, W = data.shape[2], data.shape[3], data.shape[4]
    data = data.view(n_traj, n_env, 2, T_steps, H, W)   # [n_traj, n_env, 2, T, 64, 64]

    out_path = output_dir / f"mpf_{n_samples}.pt"
    torch.save(data, out_path)
    print(f"[MPF] saved {tuple(data.shape)} → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Thermal-Hydro-Mechanical (2-D, 64×64, 5 processes)
# ---------------------------------------------------------------------------

def generate_thm(n_samples: int, output_dir: Path, workers: int) -> Path:
    """
    Generate Thermo-Hydro-Mechanical dataset matching COMPOL paper:

    Setting: 2-D 64×64 grid, 32m×32m domain, periodic BC.
    Physics:  Biot thermo-poroelasticity (quasi-static), 5 channels.
    Timesteps: 12 non-uniform — 2×500s, 4×1250s, 4×2250s, 2×4000s.
    IC:  random T field in [273, 323] K (GRF), fractal permeability per traj.
    Envs: N_env=1 (single set of material params, IC and k vary).

    Output shape: [N_traj, 1, 5, 12, 64, 64]
        channel 0: T    (temperature, normalised)
        channel 1: p    (pore pressure, normalised)
        channel 2: ε_xx (normal strain x, normalised)
        channel 3: ε_yy (normal strain y, normalised)
        channel 4: ε_xy (shear strain, normalised)
    """
    n_train, n_val, n_test, total = _split_sizes(n_samples)

    print(f"[THM] generating {total} trajectories "
          f"({n_train} train / {n_val} val / {n_test} test)")
    print(f"     64×64 grid, 32m×32m, 12 steps "
          f"[2×500s, 4×1250s, 4×2250s, 2×4000s], N_env=1")

    dataset = THMDataset(num_traj=total, group="train")
    loader  = DataLoader(dataset, batch_size=1, num_workers=workers, shuffle=False)

    all_states = []
    for i, sample in enumerate(loader):
        all_states.append(sample["state"])   # [1, 5, 12, 64, 64]
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{total}", flush=True)

    # Stack: [total, 5, 12, 64, 64] → unsqueeze N_env → [total, 1, 5, 12, 64, 64]
    data = torch.stack([s.squeeze(0) for s in all_states], dim=0).unsqueeze(1)

    out_path = output_dir / f"thm_{n_samples}.pt"
    torch.save(data, out_path)
    print(f"[THM] saved {tuple(data.shape)} → {out_path}")
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
    p.add_argument(
        "--system", required=True,
        choices=["gs", "lv", "bz", "mpf", "thm"],
        help=(
            "PDE system to generate: "
            "gs=Gray-Scott, lv=Lotka-Volterra, bz=Belousov-Zhabotinsky, "
            "mpf=Multiphase Flow, thm=Thermal-Hydro-Mechanical"
        ),
    )
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

    generators = {
        "gs":  generate_gs,
        "lv":  generate_lv,
        "bz":  generate_bz,
        "mpf": generate_mpf,
        "thm": generate_thm,
    }
    out_path = generators[args.system](args.n_samples, out_dir, args.workers)

    print_split_ratios(args.n_samples)
    print(f"Done. Dataset saved to: {out_path}")


if __name__ == "__main__":
    main()
