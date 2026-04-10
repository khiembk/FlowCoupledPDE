"""
Thermo-Hydro-Mechanical (THM) dataset — paper-faithful implementation.

Reference: COMPOL paper (arXiv:2501.17296), Appendix A / Section on THM

Setting (adapted from Gao et al. 2020, 2-D plane-strain):
    Domain:    64×64 grid on 32 m × 32 m, periodic BC (pseudo-spectral heat)
    Physics:   Biot thermo-poroelasticity (quasi-static)
    Channels:  3 — pore pressure p, volumetric strain ε_v, temperature T
    Timesteps: 12 non-uniform steps: 2×500s, 4×1250s, 4×2250s, 2×4000s
               (total 2.3×10⁴ s, matching paper)
    IC:        Random temperature field T ∈ [273, 323] K (GRF),  p₀ = 0
    Permeability: fractal field per trajectory, k ∈ [10 D, 200 D], k_base=50 D

Governing equations (quasi-static, periodic):
    Heat:       ρC ∂T/∂t = κ_T ∇²T
    Hydraulic:  S_eff ∂p/∂t = ∇·(k/μ ∇p) + C_T ∂T/∂t
    Strain:     ε_v = (α p + K_d α_T T_rel) / M_pe     [algebraic]

where (derived parameters):
    M_pe  = K_d + 4G/3                    [P-wave modulus, ≈30 GPa]
    S_eff = 1/M_b + α²/M_pe              [effective storage, ≈6.2×10⁻¹¹ Pa⁻¹]
    C_T   = φ₀ α_T_f − α K_d α_T / M_pe  [thermo-hydraulic coupling, ≈3.5×10⁻⁵ /K]

Output per trajectory: [3, 12, 64, 64]
    channel 0: p    (pore pressure, normalised)
    channel 1: ε_v  (volumetric strain, normalised)
    channel 2: T    (temperature, normalised)

Output tensor format: [N_traj, 1, 3, 12, 64, 64]  (N_env = 1)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import eye, coo_matrix
from scipy.sparse.linalg import factorized


class THMDataset(Dataset):
    """
    THM dataset: 3 channels (p, ε_v, T), 12 non-uniform timesteps, N_env=1.

    Each trajectory has:
      - unique random temperature IC drawn uniformly in [273, 323] K via GRF
      - unique fractal permeability field in [K_MIN, K_MAX] Darcy

    Args:
        num_traj:  number of trajectories to generate
        group:     'train' or 'test' (controls RNG seed)
    """

    # Grid / domain
    SIZE    = 64
    DOMAIN  = 32.0     # metres

    # Non-uniform timesteps (seconds) — matches paper
    DT_LIST = [500, 500,
               1250, 1250, 1250, 1250,
               2250, 2250, 2250, 2250,
               4000, 4000]
    N_STEPS = 12

    # Temperature IC range (K)
    T_MIN = 273.0
    T_MAX = 323.0

    # ---- Material parameters (SI) ----
    G_MOD    = 10.0e9      # Pa  shear modulus
    NU       = 0.25        # –   Poisson ratio
    # K_d = 2G(1+ν)/(3(1−2ν))
    K_D      = 2 * 10.0e9 * (1 + 0.25) / (3 * (1 - 2 * 0.25))   # ≈ 16.67 GPa
    # M_pe = K_d + 4G/3
    M_PE     = K_D + 4 * 10.0e9 / 3.0                              # ≈ 30.00 GPa
    ALPHA    = 0.6         # –   Biot coefficient
    M_BIOT   = 20.0e9     # Pa  Biot modulus (1/specific storage at const strain)
    KT       = 2.0         # W/(m·K)  thermal conductivity
    RHO_C    = 2.2e6       # J/(m³·K) volumetric heat capacity
    PHI0     = 0.1         # –   initial porosity
    ALPHA_T  = 1.5e-5      # /K  thermal expansion coefficient (solid)
    ALPHA_TF = 4.0e-4      # /K  thermal expansion coefficient (fluid)
    MU       = 1.0e-3      # Pa·s water viscosity

    # Permeability range (Darcy → m²)
    D_TO_M2  = 9.869e-13   # 1 Darcy in m²
    K_MIN_D  = 10.0        # Darcy
    K_MAX_D  = 200.0       # Darcy
    K_BASE_D = 50.0        # Darcy
    K_HURST  = 0.75        # fractal Hurst exponent

    # ---- Derived coupling coefficients ----
    # S_eff = 1/M_b + α²/M_pe
    S_EFF = 1.0 / M_BIOT + ALPHA ** 2 / M_PE       # ≈ 6.2e-11 Pa^-1
    # C_T = φ₀ α_T_f − α K_d α_T / M_pe
    C_T   = PHI0 * ALPHA_TF - ALPHA * K_D * ALPHA_T / M_PE   # ≈ 3.5e-5 /K
    # ε_v coefficients
    EPS_P = ALPHA / M_PE                             # Pa^-1
    EPS_T = K_D * ALPHA_T / M_PE                    # /K

    def __init__(self, num_traj: int, group: str = "train"):
        super().__init__()
        self.num_traj = num_traj
        self.is_test  = (group == "test")
        self.max_seed = np.iinfo(np.int32).max
        self.buffer   = {}

        n = self.SIZE
        L = self.DOMAIN
        dx = L / n          # periodic grid spacing

        # Pre-compute heat equation Fourier denominators for each dt
        kx = 2 * np.pi * np.fft.fftfreq(n, d=L / n)   # rad/m
        ky = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        self._K2 = KX ** 2 + KY ** 2                    # [n, n]

        self._dx = dx

        # Build heat denominators per timestep (reused for all trajectories)
        # T̂_new = T̂_old * num / den, num = RHO_C/dt, den = RHO_C/dt + KT*K2
        self._heat_num = []   # scalar α_t = RHO_C/dt
        self._heat_den = []   # array [n, n]
        for dt in self.DT_LIST:
            a = self.RHO_C / dt
            self._heat_num.append(a)
            self._heat_den.append(a + self.KT * self._K2)

    # ------------------------------------------------------------------
    # Fractal permeability field
    # ------------------------------------------------------------------

    def _fractal_k(self, rng: np.random.Generator) -> np.ndarray:
        """
        Generate fractal permeability field [SIZE, SIZE] in m².

        Uses power-law spectrum with Hurst exponent H=0.75 (beta = 2*(H+1) = 3.5).
        Maps to [K_MIN_D, K_MAX_D] Darcy via min-max scaling in log space.
        """
        n = self.SIZE
        kx = np.fft.fftfreq(n) * n    # pixel frequencies [0, 1, ..., n/2, ...]
        ky = np.fft.fftfreq(n) * n
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX ** 2 + KY ** 2
        K2[0, 0] = 1.0                 # avoid div-by-zero for DC

        beta = 2.0 * (self.K_HURST + 1.0)
        S = K2 ** (-beta / 2.0)
        S[0, 0] = 0.0                  # zero-mean in log space

        noise = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        log_k_raw = np.real(np.fft.ifft2(np.sqrt(S) * noise))

        # Min-max → [log(K_MIN_D), log(K_MAX_D)]
        lo, hi = log_k_raw.min(), log_k_raw.max()
        t = (log_k_raw - lo) / (hi - lo + 1e-10)
        log_k = np.log(self.K_MIN_D) + t * np.log(self.K_MAX_D / self.K_MIN_D)
        k_darcy = np.exp(log_k)        # [K_MIN_D, K_MAX_D] Darcy

        return k_darcy * self.D_TO_M2  # m²

    # ------------------------------------------------------------------
    # Darcy operator ∇·(k/μ ∇) with periodic BC
    # ------------------------------------------------------------------

    def _build_darcy_op(self, k_m2: np.ndarray):
        """
        Build sparse matrix L_k for ∇·(k/μ ∇) with periodic BC.

        Uses 5-point FD with arithmetic-mean face permeabilities.
        Returns factorised solver for backward Euler system
        (S_eff/dt * I − L_k) which is called once per timestep.
        """
        n   = self.SIZE
        dx  = self._dx
        N   = n * n
        mu  = self.MU
        s   = 1.0 / (dx ** 2 * mu)

        # Arithmetic-mean face permeabilities (periodic wrap)
        kx_face = 0.5 * (k_m2 + np.roll(k_m2, -1, axis=0))  # k at face (i+½, j)
        ky_face = 0.5 * (k_m2 + np.roll(k_m2, -1, axis=1))  # k at face (i, j+½)

        kxf = kx_face.ravel()
        kyf = ky_face.ravel()
        kxm = np.roll(kx_face, 1, axis=0).ravel()   # k at face (i-½, j)
        kym = np.roll(ky_face, 1, axis=1).ravel()   # k at face (i, j-½)

        idx = np.arange(N)
        # Neighbour indices (periodic)
        ip = ((idx // n + 1) % n) * n + (idx % n)   # +x
        im = ((idx // n - 1) % n) * n + (idx % n)   # -x
        jp = (idx // n) * n + (idx % n + 1) % n     # +y
        jm = (idx // n) * n + (idx % n - 1) % n     # -y

        diag = -(kxf + kxm + kyf + kym) * s
        rows = np.tile(idx, 5)
        cols = np.concatenate([idx, ip, im, jp, jm])
        vals = np.concatenate([diag, kxf * s, kxm * s, kyf * s, kym * s])

        L = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
        return L   # caller builds (S_eff/dt * I − L) and factorises

    # ------------------------------------------------------------------
    # Temperature IC: GRF in [T_MIN, T_MAX] K
    # ------------------------------------------------------------------

    def _temperature_ic(self, rng: np.random.Generator) -> np.ndarray:
        """
        Random temperature field in [T_MIN, T_MAX] K via 2-D GRF.
        Returns [SIZE, SIZE] float64.
        """
        n = self.SIZE
        L = self.DOMAIN
        # Random correlation length: 3–12 m (≈ 10–40% of domain)
        l = rng.uniform(3.0, 12.0)

        kx = np.fft.fftfreq(n, d=L / n)
        ky = np.fft.fftfreq(n, d=L / n)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        # Gaussian power spectrum in physical frequency space
        S = np.exp(-0.5 * ((KX ** 2 + KY ** 2) * l ** 2))

        noise = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        field = np.real(np.fft.ifft2(noise * np.sqrt(S * n ** 2)))
        field -= field.mean()
        std = field.std()
        if std > 1e-10:
            field /= std
        # Clip to ±2 sigma then scale to [T_MIN, T_MAX]
        field = np.clip(field, -2.0, 2.0)
        field = (field + 2.0) / 4.0          # → [0, 1]
        T = self.T_MIN + field * (self.T_MAX - self.T_MIN)
        return T

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(self, traj_idx: int) -> np.ndarray:
        """Simulate one THM trajectory. Returns [3, 12, 64, 64]."""
        seed = traj_idx if not self.is_test else self.max_seed - traj_idx
        rng  = np.random.default_rng(seed)

        # Initial conditions
        T = self._temperature_ic(rng)          # [n, n] K
        p = np.zeros((self.SIZE, self.SIZE))   # gauge pressure

        k_m2 = self._fractal_k(rng)            # [n, n] m²

        # Build Darcy operator once per trajectory
        L_k = self._build_darcy_op(k_m2)       # sparse [N, N]
        L_k_dense = None                        # factorised solvers built per dt

        # Build factorised solvers per unique dt value
        # (there are 3 unique dt: 500, 1250, 2250, 4000 — precompute all)
        unique_dts = sorted(set(self.DT_LIST))
        solvers = {}
        for dt in unique_dts:
            a = self.S_EFF / dt
            A = a * eye(self.SIZE ** 2, format='csr') - L_k
            solvers[dt] = factorized(A)

        p_snaps, eps_snaps, T_snaps = [], [], []

        for si, dt in enumerate(self.DT_LIST):
            # ---- (1) Advance T: backward Euler in Fourier space ----
            Th = np.fft.fft2(T)
            Th_new = Th * (self._heat_num[si] / self._heat_den[si])
            T_new = np.real(np.fft.ifft2(Th_new))

            # ---- (2) Advance p: backward Euler with thermal source ----
            dT = T_new - T
            rhs = (self.S_EFF / dt) * p.ravel() + (self.C_T / dt) * dT.ravel()
            p_new = solvers[dt](rhs).reshape(self.SIZE, self.SIZE)
            # Fix mean (should be ~0 but may drift due to round-off)
            p_new -= p_new.mean()

            T = T_new
            p = p_new

            # ---- (3) Volumetric strain (algebraic) ----
            # ε_v = (α·p + K_d·α_T·T_rel) / M_pe
            T_rel = T - 0.5 * (self.T_MIN + self.T_MAX)   # relative to mean
            eps_v = self.EPS_P * p + self.EPS_T * T_rel

            p_snaps.append(p.copy())
            eps_snaps.append(eps_v.copy())
            T_snaps.append(T.copy())

        def _norm(arr_list):
            arr  = np.stack(arr_list, axis=0).astype(np.float32)  # [12, 64, 64]
            vmin = arr.min()
            vmax = arr.max()
            return (arr - vmin) / (vmax - vmin + 1e-10)

        p_arr   = _norm(p_snaps)
        eps_arr = _norm(eps_snaps)
        T_arr   = _norm(T_snaps)

        # Stack: [3, 12, 64, 64] — channels: p, ε_v, T
        return np.stack([p_arr, eps_arr, T_arr], axis=0)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_traj

    def __getitem__(self, idx: int) -> dict:
        if idx not in self.buffer:
            self.buffer[idx] = self._simulate(idx)
        state = torch.from_numpy(self.buffer[idx])   # [3, 12, 64, 64]
        return {"state": state, "env": 0}
