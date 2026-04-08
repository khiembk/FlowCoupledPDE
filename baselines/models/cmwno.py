"""
Coupled Multiwavelet Neural Operator (CMWNO) — Xiao et al. (ICLR 2023)
https://openreview.net/forum?id=kIo_C6QmMOM

Faithful re-implementation based on the official code:
  https://github.com/joshuaxiao98/CMWNO

Architecture (1-D, official):
  - Two separate MWT networks, one per coupled process (rho / phi).
  - Each MWT contains nCZ stacked MWT_CZ layers.
  - MWT_CZ uses Legendre polynomial filter banks (H0/H1/G0/G1) for
    analysis (even/odd decomposition) and synthesis (even/odd reconstruction).
  - Inside each MWT_CZ, three sparse Fourier kernels A, B, C are used:
      Ud[level] = A(detail) + B(scaling)   # detail contribution
      Us[level] = C(detail)                # scaling contribution
  - Coupling: network-1 runs first with no_grad to produce (Ud1, Us1).
    Network-2 receives these and adds Us1[level] to its own Us[level]
    during reconstruction — this is the cross-process information injection.
    The same is done symmetrically (net2 → net1).

2-D extension: separable row-then-column 1-D MWT transforms.

Input / output convention (matches train_baseline.py):
  x:   [B, n_proc * in_channels, ...]   processes concatenated on channel dim
  out: [B, n_proc * out_channels, ...]
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from sympy import Poly, legendre, Symbol


# ---------------------------------------------------------------------------
# Legendre filter bank — direct port of utils.py from official repo
# ---------------------------------------------------------------------------

def _eval_legendre(n: int, x: np.ndarray) -> np.ndarray:
    """Evaluate Legendre polynomial P_n(x) via three-term recurrence (no scipy)."""
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    p_prev, p_curr = np.ones_like(x), x.copy()
    for k in range(2, n + 1):
        p_next = ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
        p_prev, p_curr = p_curr, p_next
    return p_curr


def _legendre_der(k, x):
    out = 0
    for i in np.arange(k - 1, -1, -2):
        out += (2 * i + 1) * _eval_legendre(i, x)
    return out


def _get_phi_psi(k: int):
    x = Symbol('x')
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))
    for ki in range(k):
        c = Poly(legendre(ki, 2 * x - 1), x).all_coeffs()
        phi_coeff[ki, :ki + 1] = np.flip(np.sqrt(2 * ki + 1) * np.array(c, dtype=np.float64))
        c = Poly(legendre(ki, 4 * x - 1), x).all_coeffs()
        phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(c, dtype=np.float64))

    psi1 = np.zeros((k, k))
    psi2 = np.zeros((k, k))
    for ki in range(k):
        psi1[ki] = phi_2x_coeff[ki]
        for i in range(k):
            a, b = phi_2x_coeff[ki, :ki + 1], phi_coeff[i, :i + 1]
            prod = np.convolve(a, b)
            prod[np.abs(prod) < 1e-8] = 0
            proj = (prod / (np.arange(len(prod)) + 1) * np.power(0.5, 1 + np.arange(len(prod)))).sum()
            psi1[ki] -= proj * phi_coeff[i]
            psi2[ki] -= proj * phi_coeff[i]
        for j in range(ki):
            a, b = phi_2x_coeff[ki, :ki + 1], psi1[j]
            prod = np.convolve(a, b)
            prod[np.abs(prod) < 1e-8] = 0
            proj = (prod / (np.arange(len(prod)) + 1) * np.power(0.5, 1 + np.arange(len(prod)))).sum()
            psi1[ki] -= proj * psi1[j]
            psi2[ki] -= proj * psi2[j]

        a = psi1[ki]
        prod = np.convolve(a, a)
        prod[np.abs(prod) < 1e-8] = 0
        n1 = (prod / (np.arange(len(prod)) + 1) * np.power(0.5, 1 + np.arange(len(prod)))).sum()
        a = psi2[ki]
        prod = np.convolve(a, a)
        prod[np.abs(prod) < 1e-8] = 0
        n2 = (prod / (np.arange(len(prod)) + 1) * (1 - np.power(0.5, 1 + np.arange(len(prod))))).sum()
        norm = np.sqrt(n1 + n2)
        psi1[ki] /= norm
        psi2[ki] /= norm
        psi1[np.abs(psi1) < 1e-8] = 0
        psi2[np.abs(psi2) < 1e-8] = 0

    phi_poly  = [np.poly1d(np.flip(phi_coeff[i]))  for i in range(k)]
    psi1_poly = [np.poly1d(np.flip(psi1[i]))        for i in range(k)]
    psi2_poly = [np.poly1d(np.flip(psi2[i]))        for i in range(k)]
    return phi_poly, psi1_poly, psi2_poly


def get_filter(k: int):
    """Return Legendre wavelet filter matrices H0, H1, G0, G1 each of shape (k, k)."""
    def _psi(psi1_poly, psi2_poly, i, pts):
        return [psi1_poly[i](v) if v <= 0.5 else psi2_poly[i](v) for v in pts]

    x = Symbol('x')
    phi_poly, psi1_poly, psi2_poly = _get_phi_psi(k)
    roots = Poly(legendre(k, 2 * x - 1)).all_roots()
    x_m = np.array([r.evalf(20) for r in roots], dtype=np.float64)
    wm = 1.0 / k / _legendre_der(k, 2 * x_m - 1) / _eval_legendre(k - 1, 2 * x_m - 1)

    H0, H1, G0, G1 = [np.zeros((k, k)) for _ in range(4)]
    for ki in range(k):
        for kpi in range(k):
            H0[ki, kpi] = (1 / np.sqrt(2)) * (wm * phi_poly[ki](x_m / 2)       * phi_poly[kpi](x_m)).sum()
            H1[ki, kpi] = (1 / np.sqrt(2)) * (wm * phi_poly[ki]((x_m + 1) / 2) * phi_poly[kpi](x_m)).sum()
            G0[ki, kpi] = (1 / np.sqrt(2)) * (wm * np.array(_psi(psi1_poly, psi2_poly, ki, x_m / 2))       * phi_poly[kpi](x_m)).sum()
            G1[ki, kpi] = (1 / np.sqrt(2)) * (wm * np.array(_psi(psi1_poly, psi2_poly, ki, (x_m + 1) / 2)) * phi_poly[kpi](x_m)).sum()

    for M in (H0, H1, G0, G1):
        M[np.abs(M) < 1e-8] = 0
    return H0, H1, G0, G1


# ---------------------------------------------------------------------------
# Sparse Fourier kernel — direct port of sparseKernelFT from official repo
# ---------------------------------------------------------------------------

class SparseKernelFT1d(nn.Module):
    def __init__(self, k: int, alpha: int, c: int):
        super().__init__()
        self.modes = alpha
        scale = 1.0 / (c * k) ** 2
        self.weights = nn.Parameter(scale * torch.rand(c * k, c * k, alpha, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, c, k]
        B, N, c, k = x.shape
        x = x.view(B, N, -1).permute(0, 2, 1)          # [B, c*k, N]
        x_ft = torch.fft.rfft(x)
        l = min(self.modes, N // 2 + 1)
        out_ft = torch.zeros(B, x.shape[1], N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = torch.einsum("bix,iox->box", x_ft[:, :, :l], self.weights[:, :, :l])
        x = torch.fft.irfft(out_ft, n=N)
        return x.permute(0, 2, 1).view(B, N, c, k)


# ---------------------------------------------------------------------------
# MWT_CZ: one coupled-wavelet layer — direct port from official repo
# ---------------------------------------------------------------------------

class MWT_CZ(nn.Module):
    """
    Legendre-wavelet decomposition + sparse-Fourier operator + reconstruction.

    When u_s is not None (coupling mode), the other network's scaling
    coefficients are injected additively during reconstruction:
        x = x + Us[level] + u_s[level]
    """

    def __init__(self, k: int, alpha: int, c: int, L: int = 0):
        super().__init__()
        self.k = k
        self.L = L

        H0, H1, G0, G1 = get_filter(k)

        self.A = SparseKernelFT1d(k, alpha, c)
        self.B = SparseKernelFT1d(k, alpha, c)
        self.C = SparseKernelFT1d(k, alpha, c)
        self.T0 = nn.Linear(k, k)

        # Analysis filters: shape [2k, k]
        self.register_buffer('ec_s', torch.tensor(np.concatenate([H0.T, H1.T], axis=0), dtype=torch.float32))
        self.register_buffer('ec_d', torch.tensor(np.concatenate([G0.T, G1.T], axis=0), dtype=torch.float32))
        # Synthesis filters: shape [2k, k]
        self.register_buffer('rc_e', torch.tensor(np.concatenate([H0, G0], axis=0), dtype=torch.float32))
        self.register_buffer('rc_o', torch.tensor(np.concatenate([H1, G1], axis=0), dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        u_d: Optional[List[torch.Tensor]] = None,
        u_s: Optional[List[torch.Tensor]] = None,
    ):
        """
        x:   [B, N, c, k]   (N must be a power-of-two)
        Returns (x_out, Ud, Us) where Ud/Us are per-level coefficient lists.
        """
        B, N, c, k = x.shape
        ns = math.floor(math.log2(N))

        Ud: List[torch.Tensor] = []
        Us: List[torch.Tensor] = []

        # Decompose
        for _ in range(ns - self.L):
            d, x = self._wavelet_transform(x)
            Ud.append(self.A(d) + self.B(x))
            Us.append(self.C(d))

        x = self.T0(x)   # coarsest-scale transform

        # Reconstruct
        if u_s is None:
            for i in range(ns - 1 - self.L, -1, -1):
                x = x + Us[i]
                x = torch.cat([x, Ud[i]], dim=-1)
                x = self._even_odd(x)
        else:
            for i in range(ns - 1 - self.L, -1, -1):
                x = x + Us[i] + u_s[i]          # <-- cross-process injection
                x = torch.cat([x, Ud[i]], dim=-1)
                x = self._even_odd(x)

        return x, Ud, Us

    def _wavelet_transform(self, x: torch.Tensor):
        # x: [B, N, c, k]  →  d, s: [B, N//2, c, k]
        xa = torch.cat([x[:, ::2], x[:, 1::2]], dim=-1)   # [B, N//2, c, 2k]
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def _even_odd(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, c, 2k]  →  [B, 2N, c, k]
        B, N, c, _ = x.shape
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
        out = torch.zeros(B, N * 2, c, self.k, device=x.device, dtype=x.dtype)
        out[:, ::2]  = x_e
        out[:, 1::2] = x_o
        return out


# ---------------------------------------------------------------------------
# MWT: full multi-scale network for a single process
# ---------------------------------------------------------------------------

class MWT(nn.Module):
    """
    Multiwavelet neural operator for one process.
    Direct port of the official MWT class.
    """

    def __init__(self, ich: int, k: int, alpha: int, c: int, nCZ: int, L: int = 0):
        super().__init__()
        self.k = k
        self.c = c
        self.nCZ = nCZ

        self.Lk  = nn.Linear(ich, c * k)
        self.czs = nn.ModuleList([MWT_CZ(k, alpha, c, L) for _ in range(nCZ)])
        self.Lc0 = nn.Linear(c * k, 128)
        self.Lc1 = nn.Linear(128, 1)

    def forward(
        self,
        x: torch.Tensor,
        u_d: Optional[List[List[torch.Tensor]]] = None,
        u_s: Optional[List[List[torch.Tensor]]] = None,
    ):
        """
        x:   [B, N, ich]
        u_d: nCZ-length list of per-level detail lists (from partner network)
        u_s: nCZ-length list of per-level scaling lists (from partner network)
        Returns (out [B, N, 1], Ud_all, Us_all).
        """
        B, N, _ = x.shape
        x = self.Lk(x).view(B, N, self.c, self.k)

        Ud_all: List[List[torch.Tensor]] = []
        Us_all: List[List[torch.Tensor]] = []

        for i, cz in enumerate(self.czs):
            ud_in = u_d[i] if u_d is not None else None
            us_in = u_s[i] if u_s is not None else None
            x, ud, us = cz(x, ud_in, us_in)
            x = torch.tanh(x)
            Ud_all.append(ud)
            Us_all.append(us)

        x = x.view(B, N, -1)
        x = F.relu(self.Lc0(x))
        x = self.Lc1(x)           # [B, N, 1]
        return x, Ud_all, Us_all


# ---------------------------------------------------------------------------
# Coupled forward helper
# ---------------------------------------------------------------------------

def _coupled_forward_1d(x1, x2, net1, net2):
    """
    Bidirectional coupling:
      1. net1 runs no_grad  → produces Ud1, Us1
         net2 receives Ud1/Us1 and produces out2
      2. net2 runs no_grad  → produces Ud2, Us2
         net1 receives Ud2/Us2 and produces out1
    Both outputs [B, N, 1] are returned.
    """
    with torch.no_grad():
        _, Ud1, Us1 = net1(x1)
    out2, _, _ = net2(x2, Ud1, Us1)

    with torch.no_grad():
        _, Ud2, Us2 = net2(x2)
    out1, _, _ = net1(x1, Ud2, Us2)

    return out1, out2   # each [B, N, 1]


# ---------------------------------------------------------------------------
# CMWNO1d
# ---------------------------------------------------------------------------

class CMWNO1d(nn.Module):
    """
    1-D Coupled Multiwavelet Neural Operator.

    Args:
        in_channels:  channels per process in input  (typically 1).
        out_channels: channels per process in output (typically 1, hardcoded
                      by the official architecture's final Linear(128,1)).
        n_proc:       number of coupled processes (must be 2).
        width:        controls hidden size as c = width // k.
        n_layers:     number of MWT_CZ stacked layers (nCZ).
        k:            Legendre wavelet order (e.g. 4).
        alpha:        number of Fourier modes in sparse kernel.
        L:            coarsest-level offset (0 = full decomposition).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_proc: int = 2,
        width: int = 64,
        n_layers: int = 2,
        k: int = 4,
        alpha: int = 10,
        L: int = 0,
    ):
        super().__init__()
        assert n_proc == 2, "CMWNO requires exactly 2 coupled processes"
        self.in_channels = in_channels
        c = max(1, width // k)
        self.net1 = MWT(ich=in_channels, k=k, alpha=alpha, c=c, nCZ=n_layers, L=L)
        self.net2 = MWT(ich=in_channels, k=k, alpha=alpha, c=c, nCZ=n_layers, L=L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2*C, L]
        C = self.in_channels
        x1 = x[:, :C].permute(0, 2, 1)    # [B, L, C]
        x2 = x[:, C:2*C].permute(0, 2, 1) # [B, L, C]

        out1, out2 = _coupled_forward_1d(x1, x2, self.net1, self.net2)
        # out: [B, L, 1] → [B, 1, L]
        out1 = out1.permute(0, 2, 1)
        out2 = out2.permute(0, 2, 1)
        return torch.cat([out1, out2], dim=1)   # [B, 2, L]


# ---------------------------------------------------------------------------
# CMWNO2d — separable 2-D extension
# ---------------------------------------------------------------------------

class CMWNO2d(nn.Module):
    """
    2-D Coupled Multiwavelet Neural Operator.
    Applies coupled 1-D MWT separably: first along the W (column) axis,
    then along the H (row) axis — mirroring the standard separable wavelet
    approach for 2-D signals.

    Spatial dims (H, W) must each be a power of two.

    Args: same as CMWNO1d; padding pads H and W before the wavelet layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_proc: int = 2,
        width: int = 64,
        n_layers: int = 2,
        k: int = 4,
        alpha: int = 10,
        L: int = 0,
        padding: int = 0,
    ):
        super().__init__()
        assert n_proc == 2, "CMWNO requires exactly 2 coupled processes"
        self.in_channels = in_channels
        self.padding = padding
        c = max(1, width // k)

        # Row-wise coupled pair (operates along W)
        self.net1_row = MWT(ich=in_channels, k=k, alpha=alpha, c=c, nCZ=n_layers, L=L)
        self.net2_row = MWT(ich=in_channels, k=k, alpha=alpha, c=c, nCZ=n_layers, L=L)
        # Column-wise coupled pair (operates along H; input is 1-channel from row stage)
        self.net1_col = MWT(ich=1, k=k, alpha=alpha, c=c, nCZ=n_layers, L=L)
        self.net2_col = MWT(ich=1, k=k, alpha=alpha, c=c, nCZ=n_layers, L=L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2*C, H, W]
        B, _, H, W = x.shape
        C = self.in_channels
        x1 = x[:, :C]      # [B, C, H, W]
        x2 = x[:, C:2*C]

        if self.padding > 0:
            x1 = F.pad(x1, [0, self.padding, 0, self.padding])
            x2 = F.pad(x2, [0, self.padding, 0, self.padding])
        Hp, Wp = x1.shape[2], x1.shape[3]

        # ── Row transform (along W) ──────────────────────────────────────────
        # reshape to [B*H, W, C] for MWT
        x1r = x1.permute(0, 2, 1, 3).reshape(B * Hp, C, Wp).permute(0, 2, 1)
        x2r = x2.permute(0, 2, 1, 3).reshape(B * Hp, C, Wp).permute(0, 2, 1)

        o1r, o2r = _coupled_forward_1d(x1r, x2r, self.net1_row, self.net2_row)
        # o: [B*H, W, 1] → [B, 1, H, W]
        o1r = o1r.permute(0, 2, 1).reshape(B, Hp, 1, Wp).permute(0, 2, 1, 3)
        o2r = o2r.permute(0, 2, 1).reshape(B, Hp, 1, Wp).permute(0, 2, 1, 3)

        # ── Column transform (along H) ───────────────────────────────────────
        # reshape to [B*W, H, 1] for MWT
        x1c = o1r.permute(0, 3, 1, 2).reshape(B * Wp, 1, Hp).permute(0, 2, 1)
        x2c = o2r.permute(0, 3, 1, 2).reshape(B * Wp, 1, Hp).permute(0, 2, 1)

        o1c, o2c = _coupled_forward_1d(x1c, x2c, self.net1_col, self.net2_col)
        # o: [B*W, H, 1] → [B, 1, H, W]
        o1 = o1c.permute(0, 2, 1).reshape(B, Wp, 1, Hp).permute(0, 2, 3, 1)
        o2 = o2c.permute(0, 2, 1).reshape(B, Wp, 1, Hp).permute(0, 2, 3, 1)

        if self.padding > 0:
            o1 = o1[:, :, :H, :W]
            o2 = o2[:, :, :H, :W]

        return torch.cat([o1, o2], dim=1)   # [B, 2, H, W]
