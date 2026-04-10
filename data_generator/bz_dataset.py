"""
Belousov-Zhabotinsky (BZ) dataset — paper-faithful implementation.

Reference: COMPOL paper (arXiv:2501.17296), Appendix A

Equations:
    ∂u/∂t = ε₁ ∇²u  +  u + v - uv - u²
    ∂v/∂t = ε₂ ∇²v  +  w - v - uv
    ∂w/∂t = ε₃ ∇²w  +  u - w

Parameters: ε₁ = ε₂ = 0.01,  ε₃ = 0.005,  t ∈ [0, 0.5]
Domain:     [0, 1], periodic BC
Solver:     ETDRK4 (pseudo-spectral, Cox & Matthews 2002)
Resolution: 1024 points (fine grid) → subsampled to 256 (dataset)
IC:         Independent 1D GRFs per channel, Gaussian kernel, l = 0.03

Output per trajectory: [3, T, 256]
    channel 0: u
    channel 1: v
    channel 2: w

Output tensor format: [N_traj, 1, 3, T, 256]   (N_env = 1)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class BZDataset(Dataset):
    """
    Generates BZ trajectories via ETDRK4 pseudo-spectral method.

    Simulates on N_FINE=1024 points, subsamples to N_COARSE=256.
    IC: independent 1D GRF per channel with Gaussian kernel (l=0.03).

    Args:
        num_traj:    number of trajectories
        dt:          ETDRK4 internal step size (default 5e-4 is stable)
        n_snapshots: number of saved output frames in (0, T_END]
        group:       'train' or 'test' (controls RNG seed offset)
    """

    EPS1 = 0.01
    EPS2 = 0.01
    EPS3 = 0.005
    T_END = 0.5
    N_FINE = 1024
    N_COARSE = 256
    GRF_L = 0.03          # GRF correlation length

    def __init__(
        self,
        num_traj: int,
        dt: float = 5e-4,
        n_snapshots: int = 20,
        group: str = "train",
    ):
        super().__init__()
        self.num_traj    = num_traj
        self.dt          = dt
        self.n_snapshots = n_snapshots
        self.is_test     = (group == "test")
        self.max_seed    = np.iinfo(np.int32).max
        self.buffer      = {}

        n = self.N_FINE
        # Wavenumbers for [0,1] periodic domain: k = 2π·n·freq
        freq = np.fft.rfftfreq(n, d=1.0 / n)   # [0, 1, 2, ..., n/2]
        k    = 2 * np.pi * freq                  # physical wavenumbers
        self._k2 = k ** 2                        # k² for Laplacian

        # Subsampling stride
        self._stride = self.N_FINE // self.N_COARSE

        # Pre-compute ETDRK4 coefficients (reused every step)
        self._cu = self._precompute(self.EPS1, dt)
        self._cv = self._precompute(self.EPS2, dt)
        self._cw = self._precompute(self.EPS3, dt)

        # Steps between consecutive saved snapshots
        dt_snap = self.T_END / self.n_snapshots
        self._steps_per_snap = max(1, round(dt_snap / dt))

    # ------------------------------------------------------------------
    # ETDRK4 pre-computation
    # ------------------------------------------------------------------

    def _precompute(self, eps: float, h: float) -> dict:
        """
        Compute ETDRK4 coefficients for linear operator c = -eps * k².
        Handles c = 0 (k = 0 mode) via Taylor series.
        Returns dict with arrays of shape [N_FINE//2 + 1].
        """
        c  = -eps * self._k2           # ≤ 0
        ch = c * h
        ch2 = c * (h / 2.0)

        E  = np.exp(ch)
        E2 = np.exp(ch2)

        # phi(x) = (exp(x) - 1) / x,  lim_{x→0} = 1
        # phi_h  = h * phi(ch)  → h as ch→0
        # phi_h2 = (h/2) * phi(ch2) → h/2 as ch→0
        def phi(x, scale):
            small = np.abs(x) < 1e-8
            safe_x = np.where(small, 1e-8, x)
            val = scale * (np.exp(x) - 1.0) / safe_x
            return np.where(small, scale * np.ones_like(x), val)

        phi_h  = phi(ch,  h)
        phi_h2 = phi(ch2, h / 2.0)

        # Cox-Matthews final-step weights (all approach h/6, h/3, h/6 as ch→0)
        small = np.abs(ch) < 1e-6
        ch2_s = np.where(small, 1e-6, ch) ** 2   # avoid /0

        f1 = np.where(small, h / 6.0,
                      h * (-4 - ch + E * (4 - 3*ch + ch**2)) / ch2_s)
        f2 = np.where(small, h / 3.0,
                      h * 2 * (2 + ch + E * (-2 + ch)) / ch2_s)
        f3 = np.where(small, h / 6.0,
                      h * (-4 - 3*ch - ch**2 + E * (4 - ch)) / ch2_s)

        return dict(E=E, E2=E2, phi_h=phi_h, phi_h2=phi_h2, f1=f1, f2=f2, f3=f3)

    # ------------------------------------------------------------------
    # Reaction (nonlinear) terms
    # ------------------------------------------------------------------

    @staticmethod
    def _Nu(u, v, w): return  u + v - u * v - u ** 2
    @staticmethod
    def _Nv(u, v, w): return  w - v - u * v
    @staticmethod
    def _Nw(u, v, w): return  u - w

    # ------------------------------------------------------------------
    # One ETDRK4 step (pseudo-spectral)
    # ------------------------------------------------------------------

    def _step(self, u, v, w):
        n   = self.N_FINE
        cu, cv, cw = self._cu, self._cv, self._cw
        rfft  = np.fft.rfft
        irfft = np.fft.irfft

        def to_k(x):    return rfft(x)
        def to_x(x, c): return irfft(x, n=n)  # noqa – c unused but documents intent

        # Stage 1 — evaluate N at current state
        Nu1 = rfft(self._Nu(u, v, w))
        Nv1 = rfft(self._Nv(u, v, w))
        Nw1 = rfft(self._Nw(u, v, w))

        uh = rfft(u); vh = rfft(v); wh = rfft(w)

        # Stage 2 — half-step using N1
        au = irfft(cu["E2"] * uh + cu["phi_h2"] * Nu1, n=n)
        av = irfft(cv["E2"] * vh + cv["phi_h2"] * Nv1, n=n)
        aw = irfft(cw["E2"] * wh + cw["phi_h2"] * Nw1, n=n)

        Nu2 = rfft(self._Nu(au, av, aw))
        Nv2 = rfft(self._Nv(au, av, aw))
        Nw2 = rfft(self._Nw(au, av, aw))

        # Stage 3 — half-step using N2
        bu = irfft(cu["E2"] * uh + cu["phi_h2"] * Nu2, n=n)
        bv = irfft(cv["E2"] * vh + cv["phi_h2"] * Nv2, n=n)
        bw = irfft(cw["E2"] * wh + cw["phi_h2"] * Nw2, n=n)

        Nu3 = rfft(self._Nu(bu, bv, bw))
        Nv3 = rfft(self._Nv(bu, bv, bw))
        Nw3 = rfft(self._Nw(bu, bv, bw))

        # Stage 4 — full-step using 2·N3 - N1 (starting from half-step a)
        auh = rfft(au); avh = rfft(av); awh = rfft(aw)
        cu_ = irfft(cu["E2"] * auh + cu["phi_h2"] * (2 * Nu3 - Nu1), n=n)
        cv_ = irfft(cv["E2"] * avh + cv["phi_h2"] * (2 * Nv3 - Nv1), n=n)
        cw_ = irfft(cw["E2"] * awh + cw["phi_h2"] * (2 * Nw3 - Nw1), n=n)

        Nu4 = rfft(self._Nu(cu_, cv_, cw_))
        Nv4 = rfft(self._Nv(cu_, cv_, cw_))
        Nw4 = rfft(self._Nw(cu_, cv_, cw_))

        # Final Cox-Matthews combination
        u_new = irfft(cu["E"] * uh + cu["f1"] * Nu1 + cu["f2"] * (Nu2 + Nu3) + cu["f3"] * Nu4, n=n)
        v_new = irfft(cv["E"] * vh + cv["f1"] * Nv1 + cv["f2"] * (Nv2 + Nv3) + cv["f3"] * Nv4, n=n)
        w_new = irfft(cw["E"] * wh + cw["f1"] * Nw1 + cw["f2"] * (Nw2 + Nw3) + cw["f3"] * Nw4, n=n)

        return u_new, v_new, w_new

    # ------------------------------------------------------------------
    # GRF initial condition
    # ------------------------------------------------------------------

    def _grf(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample one 1D GRF on [0,1] periodic with Gaussian kernel, l=0.03.
        Returns array of shape [N_FINE], zero-mean.
        """
        n = self.N_FINE
        freq = np.fft.rfftfreq(n, d=1.0 / n)
        k    = 2 * np.pi * freq
        S    = np.exp(-0.5 * (k * self.GRF_L) ** 2)

        # Complex white noise (Hermitian symmetry enforced by rfft convention)
        noise = (rng.standard_normal(len(k)) + 1j * rng.standard_normal(len(k)))
        noise[0] = rng.standard_normal()                # DC must be real
        if n % 2 == 0:
            noise[-1] = rng.standard_normal()           # Nyquist must be real

        fld = np.fft.irfft(np.sqrt(S) * noise, n=n)
        fld -= fld.mean()
        std = fld.std()
        if std > 1e-10:
            fld /= std
        return fld

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(self, traj_idx: int) -> np.ndarray:
        """Simulate one trajectory. Returns [3, T, N_COARSE]."""
        seed = traj_idx if not self.is_test else self.max_seed - traj_idx
        rng  = np.random.default_rng(seed)

        # Independent GRF for each channel; scale to small amplitude
        u = 0.1 * self._grf(rng)
        v = 0.1 * self._grf(rng)
        w = 0.1 * self._grf(rng)

        s = self._stride
        snaps = []
        for _ in range(self.n_snapshots):
            for _ in range(self._steps_per_snap):
                u, v, w = self._step(u, v, w)
            snaps.append(np.stack([u[::s], v[::s], w[::s]], axis=0))  # [3, N_COARSE]

        return np.stack(snaps, axis=1)   # [3, T, N_COARSE]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_traj

    def __getitem__(self, idx: int) -> dict:
        if idx not in self.buffer:
            self.buffer[idx] = self._simulate(idx)
        state = torch.from_numpy(self.buffer[idx]).float()   # [3, T, N_COARSE]
        return {"state": state, "env": 0}


def default_bz_params() -> list:
    """Single environment — paper uses fixed ε₁=ε₂=0.01, ε₃=0.005."""
    return [{"eps1": BZDataset.EPS1, "eps2": BZDataset.EPS2, "eps3": BZDataset.EPS3}]
