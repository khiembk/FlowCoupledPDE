"""
Two-phase (oil-water) flow in a 2-D heterogeneous porous medium.

Uses IMPES (IMplicit Pressure, Explicit Saturation) scheme:
  - Pressure solved implicitly at each time step via sparse linear system
  - Saturation advected explicitly with first-order upwind scheme

State variables (2 processes):
    Sw  - water saturation  [0, 1]
    P   - normalized pore pressure [0, 1]

Governing equations:
    Pressure:    -∇·(λ_t K ∇P) = q_inj δ_{inj} - q_prod δ_{prod}
    Saturation:  φ ∂Sw/∂t = -∇·(fw vt)

where:
    krw(Sw) = Sw^nw,   kro(Sw) = (1-Sw)^no       (Corey model)
    λ_w = krw/μw,      λ_o = kro/μo               (phase mobilities)
    λ_t = λ_w + λ_o                               (total mobility)
    fw  = λ_w / λ_t                               (fractional flow)
    vt  = -K λ_t ∇P                               (total Darcy velocity)

Permeability K(x,y) is log-normal and fixed per environment.
Trajectories differ only in the initial water saturation Sw(x,y,0).

Output format: [N_traj, N_env, 2, T, size, size]
    channel 0: Sw (water saturation)
    channel 1: P  (normalized pressure)

References
----------
Bear, J. (2010). Modeling groundwater flow and contaminant transport.
Hashemi, M. et al. (2021). Pore-scale modeling of multiphase flow.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class MultiphaseFlowDataset(Dataset):
    """
    IMPES two-phase flow dataset for 2-D heterogeneous porous media.

    Args:
        num_traj_per_env: number of distinct initial conditions per environment
        size:             spatial grid size (default 64 → 64×64 grid)
        time_horizon:     total dimensionless simulation time
        dt_eval:          time interval between saved snapshots
        params:           list of dicts with keys
                          {phi, mu_w, mu_o, n_w, n_o, q_rate}
        group:            'train' or 'test' (controls random seed offset)
    """

    DEFAULT_PARAMS = [
        # Env 0: moderate viscosity ratio (μo/μw = 5)
        {"phi": 0.20, "mu_w": 1.0, "mu_o": 5.0,  "n_w": 2.0, "n_o": 2.0, "q_rate": 1.0},
        # Env 1: unfavorable mobility ratio (μo/μw = 10)
        {"phi": 0.20, "mu_w": 1.0, "mu_o": 10.0, "n_w": 2.0, "n_o": 3.0, "q_rate": 1.0},
        # Env 2: strongly unfavorable mobility ratio (μo/μw = 20)
        {"phi": 0.25, "mu_w": 1.0, "mu_o": 20.0, "n_w": 3.0, "n_o": 2.0, "q_rate": 1.0},
    ]

    def __init__(
        self,
        num_traj_per_env: int,
        size: int = 64,
        time_horizon: float = 1.0,
        dt_eval: float = 0.1,
        params: list = None,
        group: str = "train",
    ):
        super().__init__()
        self.num_traj_per_env = num_traj_per_env
        self.params_eq = params if params is not None else self.DEFAULT_PARAMS
        self.num_env   = len(self.params_eq)
        self.len       = num_traj_per_env * self.num_env
        self.size      = size
        self.N         = size * size
        self.time_horizon = float(time_horizon)
        self.dt_eval   = dt_eval
        self.n_steps   = int(round(time_horizon / dt_eval))
        self.test      = group == "test"
        self.max_seed  = np.iinfo(np.int32).max
        self.buffer    = {}
        self.indices   = [
            list(range(e * num_traj_per_env, (e + 1) * num_traj_per_env))
            for e in range(self.num_env)
        ]
        # Uniform grid on [0,1]²
        self.dx = 1.0 / (size - 1)

        # Well locations (flat indices): injector at top-left, producer at bottom-right
        self._inj  = 0
        self._prod = self.N - 1

        # Pre-generate one log-normal permeability field per environment
        self._K  = [self._generate_permeability(e) for e in range(self.num_env)]
        # Pre-compute harmonic-mean K at cell faces (fixed per environment)
        self._Kx = []   # x-faces [size, size-1]
        self._Ky = []   # y-faces [size-1, size]
        for K in self._K:
            Kx = 2 * K[:, :-1] * K[:, 1:] / (K[:, :-1] + K[:, 1:] + 1e-30)
            Ky = 2 * K[:-1, :] * K[1:, :] / (K[:-1, :] + K[1:, :] + 1e-30)
            self._Kx.append(Kx)
            self._Ky.append(Ky)

    # ------------------------------------------------------------------
    # Permeability generation (log-normal, isotropic, FFT method)
    # ------------------------------------------------------------------

    def _generate_permeability(self, env_seed: int) -> np.ndarray:
        """Return a 2-D log-normal permeability field of shape [size, size]."""
        rng = np.random.default_rng(env_seed + 9999)
        n   = self.size
        # Random correlation length in [0.10, 0.30] of the domain width
        l = rng.uniform(0.10, 0.30)
        kx = np.fft.fftfreq(n, d=1.0 / n)
        ky = np.fft.fftfreq(n, d=1.0 / n)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        # Squared-exponential power spectrum
        psd   = np.exp(-0.5 * (KX**2 + KY**2) * l**2)
        noise = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        field = np.real(np.fft.ifft2(noise * np.sqrt(psd * n**2)))
        field = (field - field.mean()) / (field.std() + 1e-8)
        K     = np.exp(1.5 * field)   # log-normal, range ≈ [0.05, 20]
        K     = K / K.mean()          # normalize to mean permeability = 1
        return K.astype(np.float64)

    # ------------------------------------------------------------------
    # Pressure solver
    # ------------------------------------------------------------------

    def _build_pressure_matrix(self, mob_t: np.ndarray, env: int) -> csr_matrix:
        """
        Assemble the FV pressure matrix A for -∇·(λ_t K ∇P).

        Uses arithmetic-mean mobility and harmonic-mean permeability at faces.
        Returns a CSR matrix of size [N, N].
        """
        n  = self.size
        N  = self.N
        dx = self.dx
        Kx = self._Kx[env]   # [n, n-1]
        Ky = self._Ky[env]   # [n-1, n]

        # Mobility at x-faces (arithmetic average of neighbouring cells)
        mob_xf = 0.5 * (mob_t[:, :-1] + mob_t[:, 1:])   # [n, n-1]
        mob_yf = 0.5 * (mob_t[:-1, :] + mob_t[1:, :])   # [n-1, n]
        Tx = (Kx * mob_xf).flatten() / dx**2              # [n*(n-1)]
        Ty = (Ky * mob_yf).flatten() / dx**2              # [(n-1)*n]

        # Flat indices of cells on each side of a face
        idx    = np.arange(N).reshape(n, n)
        rows_x = idx[:, :-1].flatten()    # left  cell at x-face
        cols_x = idx[:, 1:].flatten()     # right cell at x-face
        rows_y = idx[:-1, :].flatten()    # top   cell at y-face
        cols_y = idx[1:,  :].flatten()    # bot   cell at y-face

        # Diagonal = sum of outgoing transmissibilities
        diag = np.zeros(N)
        np.add.at(diag, rows_x, Tx);  np.add.at(diag, cols_x, Tx)
        np.add.at(diag, rows_y, Ty);  np.add.at(diag, cols_y, Ty)

        r = np.concatenate([np.arange(N), rows_x, cols_x, rows_y, cols_y])
        c = np.concatenate([np.arange(N), cols_x, rows_x, cols_y, rows_y])
        d = np.concatenate([diag, -Tx, -Tx, -Ty, -Ty])
        return csr_matrix((d, (r, c)), shape=(N, N))

    def _solve_pressure(self, Sw: np.ndarray, env: int) -> np.ndarray:
        """Solve the elliptic pressure equation; return P [size, size]."""
        p   = self.params_eq[env]
        Sw_c = np.clip(Sw, 0.0, 1.0)
        krw  = Sw_c ** p["n_w"]
        kro  = (1.0 - Sw_c) ** p["n_o"]
        mob_t = krw / p["mu_w"] + kro / p["mu_o"]

        A = self._build_pressure_matrix(mob_t, env)
        q = np.zeros(self.N)
        q[self._inj]  =  p["q_rate"]
        q[self._prod] = -p["q_rate"]

        # Dirichlet BC at producer: P_prod = 0 (reference pressure)
        A = A.tolil()
        A[self._prod, :]             = 0.0
        A[self._prod, self._prod]    = 1.0
        q[self._prod]                = 0.0
        A = A.tocsr()

        return spsolve(A, q).reshape(self.size, self.size)

    # ------------------------------------------------------------------
    # Saturation advection (first-order upwind)
    # ------------------------------------------------------------------

    def _advance_saturation(
        self, Sw: np.ndarray, P: np.ndarray, env: int, dt: float
    ) -> np.ndarray:
        """Advance Sw one sub-step: φ ∂Sw/∂t = -∇·(fw vt)."""
        p   = self.params_eq[env]
        Kx  = self._Kx[env]
        Ky  = self._Ky[env]
        dx  = self.dx
        n   = self.size

        Sw_c  = np.clip(Sw, 0.0, 1.0)
        krw   = Sw_c ** p["n_w"]
        kro   = (1.0 - Sw_c) ** p["n_o"]
        mob_t = krw / p["mu_w"] + kro / p["mu_o"] + 1e-30
        fw    = (krw / p["mu_w"]) / mob_t   # fractional flow [n, n]

        # Total velocity at x-faces: vt_x = -K_x * λ_t_face * dP/dx
        mob_xf = 0.5 * (mob_t[:, :-1] + mob_t[:, 1:])
        vt_x   = -Kx * mob_xf * (P[:, 1:] - P[:, :-1]) / dx   # [n, n-1]

        mob_yf = 0.5 * (mob_t[:-1, :] + mob_t[1:, :])
        vt_y   = -Ky * mob_yf * (P[1:, :] - P[:-1, :]) / dx   # [n-1, n]

        # Upwind fractional flow at faces
        fw_x = np.where(vt_x >= 0, fw[:, :-1], fw[:, 1:]) * vt_x
        fw_y = np.where(vt_y >= 0, fw[:-1, :], fw[1:, :]) * vt_y

        # Divergence ∇·(fw vt)
        div = np.zeros((n, n))
        div[:, 1:]  += fw_x / dx;   div[:, :-1] -= fw_x / dx
        div[1:, :]  += fw_y / dx;   div[:-1, :] -= fw_y / dx

        Sw_new = Sw_c - dt * div / p["phi"]
        return np.clip(Sw_new, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def _get_init_cond(self, traj_index: int) -> np.ndarray:
        """Random initial water saturation: oil-dominated with a wet injection patch."""
        seed = traj_index if not self.test else self.max_seed - traj_index
        rng  = np.random.default_rng(seed)
        n    = self.size

        # Oil-dominated background
        Sw0 = rng.uniform(0.05, 0.15) * np.ones((n, n))

        # Wet patch near the injector corner
        patch = rng.integers(n // 8, n // 4)
        pi    = rng.integers(0, max(1, n // 4))
        pj    = rng.integers(0, max(1, n // 4))
        Sw0[pi:pi + patch, pj:pj + patch] = rng.uniform(0.6, 0.9)
        return Sw0

    # ------------------------------------------------------------------
    # IMPES simulation loop
    # ------------------------------------------------------------------

    def _simulate(self, env: int, traj_idx: int) -> np.ndarray:
        """
        Run the IMPES simulation for one trajectory.

        Returns:
            state: float32 array [2, n_steps, size, size]
                   channel 0 = Sw, channel 1 = P (normalised to [0,1])
        """
        Sw  = self._get_init_cond(traj_idx)
        p   = self.params_eq[env]

        # Snapshot at t = 0: solve pressure first, then use actual Darcy
        # velocity to set CFL sub-step (much tighter than K_max/phi estimate).
        P = self._solve_pressure(Sw, env)

        Sw_c   = np.clip(Sw, 0.0, 1.0)
        krw    = Sw_c ** p["n_w"];  kro = (1.0 - Sw_c) ** p["n_o"]
        mob_t  = krw / p["mu_w"] + kro / p["mu_o"] + 1e-30
        mob_xf = 0.5 * (mob_t[:, :-1] + mob_t[:, 1:])
        mob_yf = 0.5 * (mob_t[:-1, :] + mob_t[1:, :])
        vt_x   = np.abs(self._Kx[env] * mob_xf * (P[:, 1:] - P[:, :-1]) / self.dx)
        vt_y   = np.abs(self._Ky[env] * mob_yf * (P[1:, :] - P[:-1, :]) / self.dx)
        v_max  = max(float(vt_x.max()), float(vt_y.max()), 1e-6)

        dt_cfl  = p["phi"] * self.dx / v_max
        n_inner = max(1, int(np.ceil(self.dt_eval / dt_cfl)))
        dt_sim  = self.dt_eval / n_inner

        Sw_snaps, P_snaps = [], []
        Sw_snaps.append(Sw.copy())
        P_snaps.append(P.copy())

        for _step in range(1, self.n_steps):
            for _sub in range(n_inner):
                P  = self._solve_pressure(Sw, env)
                Sw = self._advance_saturation(Sw, P, env, dt_sim)
            Sw_snaps.append(Sw.copy())
            P_snaps.append(P.copy())

        Sw_arr = np.stack(Sw_snaps, axis=0)   # [T, n, n]
        P_arr  = np.stack(P_snaps, axis=0)    # [T, n, n]

        # Normalise pressure to [0, 1] per snapshot
        P_min  = P_arr.min(axis=(1, 2), keepdims=True)
        P_max  = P_arr.max(axis=(1, 2), keepdims=True)
        P_norm = (P_arr - P_min) / (P_max - P_min + 1e-10)

        state = np.stack([Sw_arr, P_norm], axis=0).astype(np.float32)  # [2, T, n, n]
        return state

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> dict:
        env      = index // self.num_traj_per_env
        traj_idx = index % self.num_traj_per_env

        if index not in self.buffer:
            self.buffer[index] = self._simulate(env, traj_idx)

        state = torch.from_numpy(self.buffer[index])   # [2, T, n, n]
        t     = torch.arange(self.n_steps, dtype=torch.float32) * self.dt_eval
        return {"state": state, "t": t, "env": env}


# ------------------------------------------------------------------
# Default parameter sets (exported for use in gen_data.py)
# ------------------------------------------------------------------

def default_mpf_params() -> list:
    """Three multiphase-flow environments with increasing viscosity ratio."""
    return MultiphaseFlowDataset.DEFAULT_PARAMS
