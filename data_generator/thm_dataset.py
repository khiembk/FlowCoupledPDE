"""
Thermo-Hydro-Mechanical (THM) dataset — paper-faithful implementation.

Reference: COMPOL paper (arXiv:2501.17296), Appendix §THM and §Experiments

Setting (adapted from Gao et al. 2020, 2-D plane-strain):
    Domain:    64×64 grid on 32 m × 32 m, periodic BC
    Physics:   Biot thermo-poroelasticity (quasi-static)
    Channels:  5 — T, p, ε_xx, ε_yy, ε_xy  ("5 processes considering strain
               components separately", COMPOL main text)
    Timesteps: 12 non-uniform steps: 2×500s, 4×1250s, 4×2250s, 2×4000s
    IC:        Random temperature field T ∈ [273, 323] K (GRF),  p₀ = 0
    Permeability: fractal field per trajectory, k ∈ [10 D, 200 D], k_base=50 D

Governing equations (quasi-static, plane-strain):
    Heat:       ρC ∂T/∂t = κ_T ∇²T
    Hydraulic:  S_eff ∂p/∂t = ∇·(k/μ ∇p) + C_T ∂T/∂t
    Mechanical: G∇²u + (λ+G)∇(∇·u) = ∇Q,  Q = αp + K_T T_rel  [quasi-static]

Derived parameters (see derivation in code):
    M_pe  = λ + 2G = K_d + 4G/3                   [P-wave modulus, 30 GPa]
    S_eff = 1/M + α²/M_pe                          [effective storage, 6.2e-11 Pa⁻¹]
    C_T   = α α_T_m + φ₀(α_T_f−α_T_m) − α K_T/M_pe  [thermo-hydraulic, 4.25e-5 /K]

Strain components (pseudo-spectral, from quasi-static solution):
    ε̂_αβ(k) = k_α k_β Q̂(k) / (M_pe |k|²),   k ≠ 0
    Q = α p + K_T (T − T_ref),   T_ref = (T_min + T_max)/2

Output per trajectory: [5, 12, 64, 64]  (each channel normalised to [0,1])
    channel 0: T    (temperature)
    channel 1: p    (pore pressure)
    channel 2: ε_xx (normal strain, x-direction)
    channel 3: ε_yy (normal strain, y-direction)
    channel 4: ε_xy (shear strain)

Output tensor format: [N_traj, 1, 5, 12, 64, 64]  (N_env = 1)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import eye, coo_matrix
from scipy.sparse.linalg import factorized


class THMDataset(Dataset):
    """
    THM dataset: 5 channels (T, p, ε_xx, ε_yy, ε_xy), 12 non-uniform timesteps.

    Each trajectory has:
      - unique random temperature IC drawn in [273, 323] K via GRF
      - unique fractal permeability field in [K_MIN_D, K_MAX_D] Darcy

    Args:
        num_traj:  number of trajectories
        group:     'train' or 'test'
    """

    # ---- Grid / domain ----
    SIZE   = 64
    DOMAIN = 32.0      # metres

    # ---- Non-uniform timesteps (seconds) — matches paper ----
    DT_LIST = [500, 500,
               1250, 1250, 1250, 1250,
               2250, 2250, 2250, 2250,
               4000, 4000]
    N_STEPS = 12

    # ---- Temperature IC range ----
    T_MIN = 273.0      # K
    T_MAX = 323.0      # K
    T_REF = 298.0      # K  (midpoint; T_rel = T - T_REF)

    # ---- Material parameters (SI) ----
    G_MOD  = 10.0e9    # Pa  shear modulus
    NU     = 0.25      # –   Poisson's ratio
    # Drained bulk modulus:  K_d = 2G(1+ν)/(3(1−2ν))
    K_D    = 2 * 10.0e9 * (1 + 0.25) / (3 * (1 - 2 * 0.25))   # 16.667 GPa
    # P-wave modulus: M_pe = λ + 2G = K_d + 4G/3
    M_PE   = K_D + 4 * 10.0e9 / 3.0                            # 30.000 GPa
    ALPHA  = 0.6       # –   Biot coefficient
    M_BIOT = 20.0e9    # Pa  Biot modulus (= 1/specific_storage_at_const_strain)
    KT     = 2.0       # W/(m·K)   thermal conductivity
    RHO_C  = 2.2e6     # J/(m³·K)  volumetric heat capacity
    PHI0   = 0.1       # –         initial porosity
    ALPHA_T_M = 1.5e-5 # /K  thermal expansion of porous solid
    ALPHA_T_F = 4.0e-4 # /K  thermal expansion of pore fluid
    MU     = 1.0e-3    # Pa·s water viscosity

    # Thermo-mechanical coupling coefficient (K_T = K_d * alpha_T_m)
    K_T    = K_D * ALPHA_T_M   # Pa/K ≈ 2.5e5

    # ---- Permeability (Darcy → m²) ----
    D_TO_M2  = 9.869e-13   # 1 Darcy in m²
    K_MIN_D  = 10.0
    K_MAX_D  = 200.0
    K_HURST  = 0.75

    # ---- Derived coupling coefficients (from paper governing equations) ----
    #
    # From quasi-static mechanics: ε_kk = Q / M_pe,  Q = α p + K_T T_rel
    # Substituting ∂ε_kk/∂t into fluid mass balance:
    #   S_eff ∂p/∂t = ∇·(k/μ ∇p) + C_T ∂T/∂t
    #
    # S_eff = 1/M_biot + α²/M_pe
    S_EFF = 1.0 / M_BIOT + ALPHA ** 2 / M_PE          # 6.20e-11 Pa⁻¹
    #
    # C_T = β_T − α K_T/M_pe,  where β_T = α α_T_m + φ₀(α_T_f − α_T_m)
    _BETA_T = ALPHA * ALPHA_T_M + PHI0 * (ALPHA_T_F - ALPHA_T_M)
    C_T  = _BETA_T - ALPHA * K_T / M_PE               # 4.25e-5 /K

    def __init__(self, num_traj: int, group: str = "train"):
        super().__init__()
        self.num_traj = num_traj
        self.is_test  = (group == "test")
        self.max_seed = np.iinfo(np.int32).max
        self.buffer   = {}

        n  = self.SIZE
        L  = self.DOMAIN
        dx = L / n          # periodic grid spacing (m)
        self._dx = dx

        # ---- Physical wavenumbers for heat (Fourier) ----
        kx = 2 * np.pi * np.fft.fftfreq(n, d=L / n)   # rad/m
        ky = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX ** 2 + KY ** 2                         # |k|²  [n, n]
        self._K2 = K2

        # ---- Strain multipliers (spectral, quasi-static mechanics) ----
        # ε̂_αβ(k) = k_α k_β Q̂(k) / (M_pe |k|²)  for k ≠ 0
        K2_safe = np.where(K2 == 0, 1.0, K2)           # avoid /0 at DC
        self._M_xx = KX ** 2 / (self.M_PE * K2_safe)
        self._M_yy = KY ** 2 / (self.M_PE * K2_safe)
        self._M_xy = KX * KY / (self.M_PE * K2_safe)
        # Zero DC (rigid body has no strain)
        self._M_xx[0, 0] = 0.0
        self._M_yy[0, 0] = 0.0
        self._M_xy[0, 0] = 0.0

        # ---- Pre-compute heat backward-Euler denominators per dt ----
        # T̂_new(k) = (ρC/dt) / (ρC/dt + κ|k|²) * T̂_old(k)
        self._heat_num = []   # ρC/dt (scalar)
        self._heat_den = []   # ρC/dt + κ|k|²  [n,n]
        for dt in self.DT_LIST:
            a = self.RHO_C / dt
            self._heat_num.append(a)
            self._heat_den.append(a + self.KT * K2)

    # ------------------------------------------------------------------
    # Fractal permeability field
    # ------------------------------------------------------------------

    def _fractal_k(self, rng: np.random.Generator) -> np.ndarray:
        """
        2-D fractal permeability [SIZE, SIZE] in m².
        Power-law spectrum with Hurst H=0.75 (β = 2*(H+1) = 3.5).
        Mapped to [K_MIN_D, K_MAX_D] Darcy via min-max in log space.
        """
        n = self.SIZE
        kx = np.fft.fftfreq(n) * n
        ky = np.fft.fftfreq(n) * n
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX ** 2 + KY ** 2
        K2[0, 0] = 1.0

        beta = 2.0 * (self.K_HURST + 1.0)   # = 3.5
        S = K2 ** (-beta / 2.0)
        S[0, 0] = 0.0

        noise   = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        log_k_r = np.real(np.fft.ifft2(np.sqrt(S) * noise))

        lo, hi  = log_k_r.min(), log_k_r.max()
        t       = (log_k_r - lo) / (hi - lo + 1e-10)
        log_k   = np.log(self.K_MIN_D) + t * np.log(self.K_MAX_D / self.K_MIN_D)
        return np.exp(log_k) * self.D_TO_M2   # m²

    # ------------------------------------------------------------------
    # Darcy operator ∇·(k/μ ∇) with periodic BC (vectorised sparse FD)
    # ------------------------------------------------------------------

    def _build_darcy_op(self, k_m2: np.ndarray):
        """
        Sparse FD matrix for L_k = ∇·(k/μ ∇) with 5-point stencil,
        periodic BC, arithmetic-mean face permeabilities.
        Returns CSR matrix [N, N] where N = SIZE².
        """
        n   = self.SIZE
        dx  = self._dx
        N   = n * n
        s   = 1.0 / (dx ** 2 * self.MU)

        # Face permeabilities (periodic arithmetic mean)
        kx_f = 0.5 * (k_m2 + np.roll(k_m2, -1, axis=0))  # k at face (i+½, j)
        ky_f = 0.5 * (k_m2 + np.roll(k_m2, -1, axis=1))  # k at face (i, j+½)
        kxf  = kx_f.ravel()
        kyf  = ky_f.ravel()
        kxm  = np.roll(kx_f, 1, axis=0).ravel()           # k at face (i-½, j)
        kym  = np.roll(ky_f, 1, axis=1).ravel()           # k at face (i, j-½)

        idx  = np.arange(N)
        ip   = ((idx // n + 1) % n) * n + (idx % n)       # +x neighbour
        im_  = ((idx // n - 1) % n) * n + (idx % n)       # -x neighbour
        jp   = (idx // n) * n + (idx % n + 1) % n         # +y neighbour
        jm_  = (idx // n) * n + (idx % n - 1) % n         # -y neighbour

        diag = -(kxf + kxm + kyf + kym) * s
        rows = np.tile(idx, 5)
        cols = np.concatenate([idx, ip, im_, jp, jm_])
        vals = np.concatenate([diag, kxf*s, kxm*s, kyf*s, kym*s])

        L = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
        return L

    # ------------------------------------------------------------------
    # Temperature IC
    # ------------------------------------------------------------------

    def _temperature_ic(self, rng: np.random.Generator) -> np.ndarray:
        """Random GRF temperature field in [T_MIN, T_MAX] K."""
        n = self.SIZE
        L = self.DOMAIN
        l = rng.uniform(3.0, 12.0)                        # correlation length (m)

        kx = np.fft.fftfreq(n, d=L / n)
        ky = np.fft.fftfreq(n, d=L / n)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        S = np.exp(-0.5 * (KX ** 2 + KY ** 2) * l ** 2)

        noise = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        field = np.real(np.fft.ifft2(noise * np.sqrt(S * n ** 2)))
        field -= field.mean()
        std = field.std()
        if std > 1e-10:
            field /= std
        field = np.clip(field, -2.0, 2.0)
        field = (field + 2.0) / 4.0                       # → [0, 1]
        return self.T_MIN + field * (self.T_MAX - self.T_MIN)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(self, traj_idx: int) -> np.ndarray:
        """Simulate one THM trajectory. Returns float32 [5, 12, 64, 64]."""
        seed = traj_idx if not self.is_test else self.max_seed - traj_idx
        rng  = np.random.default_rng(seed)

        T   = self._temperature_ic(rng)                   # [n, n] K
        p   = np.zeros((self.SIZE, self.SIZE))             # gauge pressure
        k_m2 = self._fractal_k(rng)                       # [n, n] m²

        # Build Darcy operator and pre-factor solvers for each unique dt
        L_k = self._build_darcy_op(k_m2)
        solvers = {}
        for dt in sorted(set(self.DT_LIST)):
            A = (self.S_EFF / dt) * eye(self.SIZE ** 2, format='csr') - L_k
            solvers[dt] = factorized(A)

        T_snaps, p_snaps, exx_snaps, eyy_snaps, exy_snaps = [], [], [], [], []

        for si, dt in enumerate(self.DT_LIST):

            # ---- (1) Advance T: backward Euler in Fourier space ----
            Th    = np.fft.fft2(T)
            T_new = np.real(np.fft.ifft2(
                Th * (self._heat_num[si] / self._heat_den[si])
            ))

            # ---- (2) Advance p: backward Euler, heterogeneous Darcy ----
            dT    = T_new - T
            rhs   = (self.S_EFF / dt) * p.ravel() + (self.C_T / dt) * dT.ravel()
            p_new = solvers[dt](rhs).reshape(self.SIZE, self.SIZE)
            p_new -= p_new.mean()                         # fix null-space (mean p=0)

            T = T_new
            p = p_new

            # ---- (3) Strain components: pseudo-spectral quasi-static ----
            # Q = α p + K_T (T − T_ref)  [Pa]
            Q    = self.ALPHA * p + self.K_T * (T - self.T_REF)
            Q_h  = np.fft.fft2(Q)
            eps_xx = np.real(np.fft.ifft2(self._M_xx * Q_h))
            eps_yy = np.real(np.fft.ifft2(self._M_yy * Q_h))
            eps_xy = np.real(np.fft.ifft2(self._M_xy * Q_h))

            T_snaps.append(T.copy())
            p_snaps.append(p.copy())
            exx_snaps.append(eps_xx)
            eyy_snaps.append(eps_yy)
            exy_snaps.append(eps_xy)

        def _norm(arr_list):
            arr  = np.stack(arr_list, axis=0).astype(np.float32)  # [12, 64, 64]
            vmin = arr.min()
            vmax = arr.max()
            return (arr - vmin) / (vmax - vmin + 1e-10)

        # channels: T, p, ε_xx, ε_yy, ε_xy
        return np.stack([
            _norm(T_snaps),
            _norm(p_snaps),
            _norm(exx_snaps),
            _norm(eyy_snaps),
            _norm(exy_snaps),
        ], axis=0)                                        # [5, 12, 64, 64]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_traj

    def __getitem__(self, idx: int) -> dict:
        if idx not in self.buffer:
            self.buffer[idx] = self._simulate(idx)
        state = torch.from_numpy(self.buffer[idx])        # [5, 12, 64, 64]
        return {"state": state, "env": 0}
