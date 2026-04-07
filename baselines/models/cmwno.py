"""
Coupled Multiwavelet Neural Operator (CMWNO) — Xiao et al. (ICLR 2023)
https://openreview.net/forum?id=kIo_C6QmMOM

Extends the Multiwavelet Neural Operator (MWT) to coupled PDE systems.
Key ideas:
  1. Multiwavelet decomposition: hierarchically splits the input into
     scaling (low-freq) and wavelet (high-freq) coefficients.
  2. Coupled linear operators: at each decomposition level, the operator
     mixes representations across all coupled processes.
  3. Reconstruction via inverse multiwavelet transform.

Implementation notes:
  - The wavelet filters (analysis / synthesis) are *learned* linear maps
    initialised from Haar wavelets for stability.
  - Coupling is realised by a per-level dense mixing matrix over processes.
  - Supports both 1-D and 2-D spatial domains.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers: Haar-initialised learnable filter bank
# ---------------------------------------------------------------------------

def _haar_matrix(size: int) -> torch.Tensor:
    """Return the Haar wavelet matrix of given size (power-of-two)."""
    assert size > 0 and (size & (size - 1)) == 0, "size must be a power of 2"
    H = torch.zeros(size, size)
    if size == 1:
        H[0, 0] = 1.0
        return H
    half = size // 2
    # scaling coefficients (low-pass)
    for i in range(half):
        H[i, 2 * i] = 1 / math.sqrt(2)
        H[i, 2 * i + 1] = 1 / math.sqrt(2)
    # wavelet coefficients (high-pass)
    for i in range(half):
        H[half + i, 2 * i] = 1 / math.sqrt(2)
        H[half + i, 2 * i + 1] = -1 / math.sqrt(2)
    return H


# ---------------------------------------------------------------------------
# 1-D multiwavelet decomposition / reconstruction layer
# ---------------------------------------------------------------------------

class MWTLayer1d(nn.Module):
    """
    Single-level learnable 1-D wavelet decomposition + coupled linear operator.

    The analysis filters split x -> (scaling, detail).
    A coupled linear operator mixes the scaling coefficients across processes.
    The synthesis filters reconstruct the output.
    """

    def __init__(self, channels: int, n_proc: int, k: int = 2):
        """
        Args:
            channels: hidden channel width per process.
            n_proc:   number of coupled processes.
            k:        wavelet filter size (should be power of 2, e.g. 2 or 4).
        """
        super().__init__()
        self.k = k
        self.channels = channels
        self.n_proc = n_proc

        # Learnable analysis / synthesis matrices (k x k)
        H = _haar_matrix(k) if k <= 4 else torch.eye(k)
        self.register_buffer("H_init", H)

        self.A = nn.Parameter(H.clone())   # analysis
        self.S = nn.Parameter(H.t().clone())  # synthesis

        # Coupled linear operator on scaling coefficients [n_proc * c_s, n_proc * c_s]
        # c_s = channels (scaling half has same width)
        self.W_coup = nn.Parameter(
            torch.eye(n_proc * channels) + 0.01 * torch.randn(n_proc * channels, n_proc * channels)
        )
        self.bn = nn.BatchNorm1d(n_proc * channels)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            xs: list of [B, channels, L] per process
        Returns:
            list of [B, channels, L] per process (transformed)
        """
        B, C, L = xs[0].shape
        k = self.k

        # Pad L to be divisible by k
        pad = (k - L % k) % k
        xs_padded = [F.pad(x, (0, pad)) for x in xs]
        Lp = L + pad

        # Analysis: reshape and apply filter matrix A
        # [B, C, Lp] -> [B, C, Lp//k, k] -> apply A -> split scaling / detail
        scalings, details = [], []
        for x in xs_padded:
            x_blocks = x.reshape(B, C, Lp // k, k)          # [B, C, Lp//k, k]
            x_wt = torch.einsum("bcnk,mk->bcnm", x_blocks, self.A)  # [B, C, Lp//k, k]
            half = k // 2
            scalings.append(x_wt[..., :half].reshape(B, C, -1))
            details.append(x_wt[..., half:].reshape(B, C, -1))

        # Coupled operator on scaling: mix across processes
        # Stack: [B, n_proc*C, Lp//k//... ] -> dense mix -> unstack
        Ls = scalings[0].shape[-1]
        sc_stack = torch.stack(scalings, dim=1)              # [B, n_proc, C, Ls]
        sc_flat = sc_stack.reshape(B, self.n_proc * C, Ls)   # [B, n_proc*C, Ls]

        sc_mixed = torch.einsum("ij,bjl->bil", self.W_coup, sc_flat)  # [B, n_proc*C, Ls]
        sc_mixed = F.gelu(self.bn(sc_mixed))
        sc_mixed = sc_mixed.reshape(B, self.n_proc, C, Ls)  # [B, n_proc, C, Ls]

        # Synthesis: reconstruct per-process output
        outs = []
        for i in range(self.n_proc):
            sc = sc_mixed[:, i]                                       # [B, C, Ls]
            det = details[i]                                          # [B, C, Ls]
            x_wt = torch.cat([sc, det], dim=-1)                      # [B, C, 2*Ls]
            x_blocks = x_wt.reshape(B, C, Lp // k, k)                # [B, C, Lp//k, k]
            x_rec = torch.einsum("bcnk,mk->bcnm", x_blocks, self.S)  # [B, C, Lp//k, k]
            x_rec = x_rec.reshape(B, C, Lp)                          # [B, C, Lp]
            outs.append(x_rec[..., :L])                               # remove padding
        return outs


# ---------------------------------------------------------------------------
# 2-D multiwavelet decomposition layer
# ---------------------------------------------------------------------------

class MWTLayer2d(nn.Module):
    """
    Single-level 2-D coupled multiwavelet layer.
    Applies 1-D MWT separably (rows then columns) and couples processes.
    """

    def __init__(self, channels: int, n_proc: int, k: int = 2):
        super().__init__()
        self.row_mwt = MWTLayer1d(channels, n_proc, k)
        self.col_mwt = MWTLayer1d(channels, n_proc, k)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        # xs: list of [B, C, H, W]
        B, C, H, W = xs[0].shape

        # Apply along rows (last dim = W)
        xs_row = [x.reshape(B * H, C, W) for x in xs]
        xs_row = self.row_mwt(xs_row)
        xs_row = [x.reshape(B, C, H, W) for x in xs_row]

        # Apply along columns (reshape so cols are last)
        xs_col = [x.permute(0, 1, 3, 2).reshape(B * W, C, H) for x in xs_row]
        xs_col = self.col_mwt(xs_col)
        xs_out = [x.reshape(B, C, W, H).permute(0, 1, 3, 2) for x in xs_col]
        return xs_out


# ---------------------------------------------------------------------------
# Full CMWNO models
# ---------------------------------------------------------------------------

class CMWNO2d(nn.Module):
    """
    2-D Coupled Multiwavelet Neural Operator.

    Args:
        in_channels:   channels per process in the input (usually 1).
        out_channels:  channels per process in the output (usually 1).
        n_proc:        number of coupled processes.
        width:         hidden channel width.
        n_layers:      number of coupled multiwavelet layers.
        k:             wavelet filter size.
        padding:       spatial padding before wavelet layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_proc: int = 2,
        width: int = 32,
        n_layers: int = 4,
        k: int = 2,
        padding: int = 0,
    ):
        super().__init__()
        self.n_proc = n_proc
        self.padding = padding

        # Per-process lifting + projection
        self.lifts = nn.ModuleList(
            [nn.Conv2d(in_channels, width, 1) for _ in range(n_proc)]
        )
        self.mwt_layers = nn.ModuleList(
            [MWTLayer2d(width, n_proc, k) for _ in range(n_layers)]
        )
        # Residual pointwise conv per layer per process
        self.res_convs = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_proc)])
            for _ in range(n_layers)
        ])
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, out_channels, 1))
            for _ in range(n_proc)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_proc * in_channels, H, W]
               (processes concatenated along channel dim)
        Returns:
            [B, n_proc * out_channels, H, W]
        """
        B, total_C, H, W = x.shape
        C_per = total_C // self.n_proc

        # Split into per-process tensors
        xs = [x[:, i * C_per:(i + 1) * C_per] for i in range(self.n_proc)]

        # Lift
        xs = [self.lifts[i](xs[i]) for i in range(self.n_proc)]

        if self.padding > 0:
            xs = [F.pad(xi, [0, self.padding, 0, self.padding]) for xi in xs]

        # Coupled multiwavelet layers
        for layer_idx, mwt in enumerate(self.mwt_layers):
            res = xs
            xs = mwt(xs)
            # Residual + GELU
            xs = [
                F.gelu(xs[i] + self.res_convs[layer_idx][i](res[i]))
                for i in range(self.n_proc)
            ]

        if self.padding > 0:
            xs = [xi[..., :-self.padding, :-self.padding] for xi in xs]

        # Project to output
        outs = [self.projs[i](xs[i]) for i in range(self.n_proc)]
        return torch.cat(outs, dim=1)


class CMWNO1d(nn.Module):
    """
    1-D Coupled Multiwavelet Neural Operator.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_proc: int = 2,
        width: int = 64,
        n_layers: int = 4,
        k: int = 2,
    ):
        super().__init__()
        self.n_proc = n_proc

        self.lifts = nn.ModuleList(
            [nn.Conv1d(in_channels, width, 1) for _ in range(n_proc)]
        )
        self.mwt_layers = nn.ModuleList(
            [MWTLayer1d(width, n_proc, k) for _ in range(n_layers)]
        )
        self.res_convs = nn.ModuleList([
            nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_proc)])
            for _ in range(n_layers)
        ])
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(width, 64, 1), nn.GELU(), nn.Conv1d(64, out_channels, 1))
            for _ in range(n_proc)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, total_C, L = x.shape
        C_per = total_C // self.n_proc
        xs = [x[:, i * C_per:(i + 1) * C_per] for i in range(self.n_proc)]
        xs = [self.lifts[i](xs[i]) for i in range(self.n_proc)]

        for layer_idx, mwt in enumerate(self.mwt_layers):
            res = xs
            xs = mwt(xs)
            xs = [
                F.gelu(xs[i] + self.res_convs[layer_idx][i](res[i]))
                for i in range(self.n_proc)
            ]

        outs = [self.projs[i](xs[i]) for i in range(self.n_proc)]
        return torch.cat(outs, dim=1)
