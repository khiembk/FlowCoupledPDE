"""
COMPOL — Sun et al. (2025)
arXiv:2501.17296  "COMPOL: Coupled Multi-Physics Operator Learning"

Two variants as described in the paper (§3.2, Figure 1):
  COMPOL-ATN  (attention-based aggregation)
  COMPOL-RNN  (GRU-based recurrent aggregation)

Per-layer computation (for both variants):
  1. Aggregate all M process features → single shared latent z
       ATN:  z_l(x) = attention_over_M_processes(v_{l-1}^1(x), ..., v_{l-1}^M(x))
             per-spatial-location; z_l has full spatial structure [B, width, H, W]
       RNN:  z_l  = GRU_step(stack(v_{l-1}^m), z_{l-1})
  2. Per-process FNO block (intra-process dynamics):
       v̂_l^m = SpectralConv(v_{l-1}^m) + W(v_{l-1}^m)
  3. Inject shared z (broadcast-add, same z for every process):
       v_l^m  = GELU( v̂_l^m + W_z · z_l )

This matches paper §3.2:
  "v_l^m = h_l^m(v_{l-1}^m, z_{l-1})"
  "z_{l-1} = A(v_{l-1}^1, ..., v_{l-1}^M)"
  "b_{l-1} = simple addition" (same z broadcast to all processes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import SpectralConv2d, SpectralConv1d


# ---------------------------------------------------------------------------
# Aggregation modules (produce ONE shared z from M process features)
# ---------------------------------------------------------------------------

class ATNAggregation2d(nn.Module):
    """
    Attention-based aggregation (COMPOL-ATN) for 2-D fields.

    Computes z at each spatial location by attending over M process features:
        z(x) = sum_j alpha_j(x) * A_j(x),  alpha = softmax(Q K^T / sqrt(d_k))
    where Q is derived from the mean of all process features at x, K/A are
    per-process projections.  Maintains full spatial structure [B, width, H, W].

    c_{l-1} (paper §A.5): per-process linear transform applied before attention.
    a_{l-1} (paper §A.5): the scaled dot-product attention over M processes.
    No output projection — paper formula is z = sum_j alpha_j A_j.
    """

    def __init__(self, width: int, n_proc: int):
        super().__init__()
        self.n_proc = n_proc
        self.scale = width ** 0.5
        # Paper's c_{l-1}: per-process linear transform before attention
        self.c = nn.Linear(width, width)
        # Q from mean of all process features, K/A per-process
        self.Wq = nn.Linear(width, width)
        self.Wk = nn.Linear(width, width)
        self.Wa = nn.Linear(width, width)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            xs: list of M tensors, each [B, width, H, W]
        Returns:
            z: [B, width, H, W]  — per-spatial-location aggregate
        """
        B, width, H, W = xs[0].shape
        # Stack processes and reshape for per-location attention
        V = torch.stack(xs, dim=1)              # [B, M, width, H, W]
        V = V.permute(0, 3, 4, 1, 2)           # [B, H, W, M, width]
        V = self.c(V)                           # [B, H, W, M, width]  (c_{l-1})

        Q = self.Wq(V.mean(-2, keepdim=True))  # [B, H, W, 1, width]
        K = self.Wk(V)                          # [B, H, W, M, width]
        A = self.Wa(V)                          # [B, H, W, M, width]

        scores = (Q @ K.transpose(-2, -1)) / self.scale  # [B, H, W, 1, M]
        alpha  = F.softmax(scores, dim=-1)                 # [B, H, W, 1, M]
        z = (alpha @ A).squeeze(-2)                        # [B, H, W, width]

        return z.permute(0, 3, 1, 2).contiguous()         # [B, width, H, W]


class GRUAggregation2d(nn.Module):
    """
    GRU-based recurrent aggregation (COMPOL-RNN) for 2-D fields.

    Maintains a spatial hidden state z across layers.  At each layer:
        V  = concat(v^1, ..., v^M)   [B, M*width, H, W]
        q  = sigmoid(W_q(V) + U_q(z))
        r  = sigmoid(W_r(V) + U_r(z))
        z̃  = tanh(W_z(V) + U_z(r⊙z))
        z  = q⊙z + (1−q)⊙z̃
    All W_* / U_* are 1×1 Conv2d (pointwise, same spatial structure as features).
    Returns updated z with same shape as each process feature [B, width, H, W].
    """

    def __init__(self, width: int, n_proc: int):
        super().__init__()
        in_dim = width * n_proc
        self.Wq = nn.Conv2d(in_dim, width, 1)
        self.Uq = nn.Conv2d(width,  width, 1)
        self.Wr = nn.Conv2d(in_dim, width, 1)
        self.Ur = nn.Conv2d(width,  width, 1)
        self.Wz = nn.Conv2d(in_dim, width, 1)
        self.Uz = nn.Conv2d(width,  width, 1)

    def forward(self, xs: list[torch.Tensor], z_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xs:     list of M tensors [B, width, H, W]
            z_prev: [B, width, H, W]  — previous recurrent state
        Returns:
            z_new:  [B, width, H, W]
        """
        V = torch.cat(xs, dim=1)                                # [B, M*width, H, W]
        q = torch.sigmoid(self.Wq(V) + self.Uq(z_prev))
        r = torch.sigmoid(self.Wr(V) + self.Ur(z_prev))
        z_tilde = torch.tanh(self.Wz(V) + self.Uz(r * z_prev))
        return q * z_prev + (1.0 - q) * z_tilde                 # [B, width, H, W]


class ATNAggregation1d(nn.Module):
    """Attention-based aggregation (COMPOL-ATN) for 1-D signals.

    Computes z per spatial location by attending over M process features.
    Maintains full spatial structure [B, width, L].
    """

    def __init__(self, width: int, n_proc: int):
        super().__init__()
        self.scale = width ** 0.5
        self.c  = nn.Linear(width, width)
        self.Wq = nn.Linear(width, width)
        self.Wk = nn.Linear(width, width)
        self.Wa = nn.Linear(width, width)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """xs: list of M tensors [B, width, L].  Returns [B, width, L]."""
        B, width, L = xs[0].shape
        V = torch.stack(xs, dim=1)          # [B, M, width, L]
        V = V.permute(0, 3, 1, 2)          # [B, L, M, width]
        V = self.c(V)                       # [B, L, M, width]  (c_{l-1})
        Q = self.Wq(V.mean(-2, keepdim=True))  # [B, L, 1, width]
        K = self.Wk(V)                         # [B, L, M, width]
        A = self.Wa(V)                         # [B, L, M, width]
        scores = (Q @ K.transpose(-2, -1)) / self.scale  # [B, L, 1, M]
        z = (F.softmax(scores, dim=-1) @ A).squeeze(-2)  # [B, L, width]
        return z.permute(0, 2, 1).contiguous()            # [B, width, L]


class GRUAggregation1d(nn.Module):
    """GRU-based recurrent aggregation (COMPOL-RNN) for 1-D signals."""

    def __init__(self, width: int, n_proc: int):
        super().__init__()
        in_dim = width * n_proc
        self.Wq = nn.Conv1d(in_dim, width, 1)
        self.Uq = nn.Conv1d(width,  width, 1)
        self.Wr = nn.Conv1d(in_dim, width, 1)
        self.Ur = nn.Conv1d(width,  width, 1)
        self.Wz = nn.Conv1d(in_dim, width, 1)
        self.Uz = nn.Conv1d(width,  width, 1)

    def forward(self, xs: list[torch.Tensor], z_prev: torch.Tensor) -> torch.Tensor:
        V = torch.cat(xs, dim=1)
        q = torch.sigmoid(self.Wq(V) + self.Uq(z_prev))
        r = torch.sigmoid(self.Wr(V) + self.Ur(z_prev))
        z_tilde = torch.tanh(self.Wz(V) + self.Uz(r * z_prev))
        return q * z_prev + (1.0 - q) * z_tilde


# ---------------------------------------------------------------------------
# COMPOL blocks (FNO + z injection)
# ---------------------------------------------------------------------------

class COMPOLBlock2d(nn.Module):
    """
    One COMPOL layer (2-D).

    Computation:
        v_l^m = GELU( SpectralConv(v_{l-1}^m) + W(v_{l-1}^m) + W_z · z )
    where z is the shared aggregated latent from all processes.
    """

    def __init__(self, width: int, modes1: int, modes2: int, n_proc: int,
                 aggr_type: str = 'atn'):
        super().__init__()
        # Per-process intra-dynamics (standard FNO blocks, no extra norm)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_proc)]
        )
        self.w = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_proc)]
        )
        # Per-process z-injection weight (W_z in paper's b_{l-1})
        self.wz = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_proc)]
        )
        # Shared aggregation module
        if aggr_type == 'atn':
            self.aggregation = ATNAggregation2d(width, n_proc)
            self.is_rnn = False
        elif aggr_type == 'rnn':
            self.aggregation = GRUAggregation2d(width, n_proc)
            self.is_rnn = True
        else:
            raise ValueError(f"aggr_type must be 'atn' or 'rnn', got {aggr_type!r}")

    def forward(self, xs: list[torch.Tensor],
                z_prev: torch.Tensor | None = None) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Returns:
            xs_new: updated process features
            z:      aggregated latent (updated GRU state for RNN; fresh for ATN)
        """
        if self.is_rnn:
            z = self.aggregation(xs, z_prev)
        else:
            z = self.aggregation(xs)

        xs_new = []
        for i, x in enumerate(xs):
            xs_new.append(F.gelu(self.spectral[i](x) + self.w[i](x) + self.wz[i](z)))
        return xs_new, z


class COMPOLBlock1d(nn.Module):
    """One COMPOL layer (1-D)."""

    def __init__(self, width: int, modes: int, n_proc: int, aggr_type: str = 'atn'):
        super().__init__()
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_proc)]
        )
        self.w = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_proc)]
        )
        self.wz = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_proc)]
        )
        if aggr_type == 'atn':
            self.aggregation = ATNAggregation1d(width, n_proc)
            self.is_rnn = False
        elif aggr_type == 'rnn':
            self.aggregation = GRUAggregation1d(width, n_proc)
            self.is_rnn = True
        else:
            raise ValueError(f"aggr_type must be 'atn' or 'rnn', got {aggr_type!r}")

    def forward(self, xs: list[torch.Tensor],
                z_prev: torch.Tensor | None = None) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self.is_rnn:
            z = self.aggregation(xs, z_prev)
        else:
            z = self.aggregation(xs)

        xs_new = []
        for i, x in enumerate(xs):
            xs_new.append(F.gelu(self.spectral[i](x) + self.w[i](x) + self.wz[i](z)))
        return xs_new, z


# ---------------------------------------------------------------------------
# Full COMPOL models
# ---------------------------------------------------------------------------

class COMPOL2d(nn.Module):
    """
    2-D COMPOL (Sun et al. 2025).

    Args:
        in_channels:  channels per process (typically 1).
        out_channels: output channels per process.
        n_proc:       number of coupled processes.
        modes1/2:     Fourier modes per spatial direction.
        width:        hidden channel width.
        n_layers:     number of COMPOL blocks.
        n_heads:      unused (kept for API compatibility; paper uses single-head ATN).
        padding:      zero-padding before spectral layers.
        aggr_type:    'atn' (COMPOL-ATN) or 'rnn' (COMPOL-RNN).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_proc: int = 2,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        n_layers: int = 4,
        n_heads: int = 4,   # kept for API compat
        padding: int = 9,
        aggr_type: str = 'atn',
    ):
        super().__init__()
        self.n_proc = n_proc
        self.padding = padding
        self.aggr_type = aggr_type

        # Per-process lifting: (in_channels + 2 grid coords) -> width
        self.lifts = nn.ModuleList(
            [nn.Linear(in_channels + 2, width) for _ in range(n_proc)]
        )
        self.blocks = nn.ModuleList(
            [COMPOLBlock2d(width, modes1, modes2, n_proc, aggr_type)
             for _ in range(n_layers)]
        )
        # Per-process projection: width -> 128 -> out_channels
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_channels))
            for _ in range(n_proc)
        ])
        self._width = width

    @staticmethod
    def _get_grid(B: int, H: int, W: int, device) -> torch.Tensor:
        gx = torch.linspace(0, 1, H, device=device).view(1, H, 1, 1).expand(B, -1, W, 1)
        gy = torch.linspace(0, 1, W, device=device).view(1, 1, W, 1).expand(B, H, -1, 1)
        return torch.cat([gx, gy], dim=-1)   # [B, H, W, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, total_C, H, W = x.shape
        C_per = total_C // self.n_proc
        grid = self._get_grid(B, H, W, x.device)

        # Lift each process
        xs = []
        for i in range(self.n_proc):
            xi = x[:, i * C_per:(i + 1) * C_per].permute(0, 2, 3, 1)  # [B, H, W, C_per]
            xi = torch.cat([xi, grid], dim=-1)                           # [B, H, W, C+2]
            xi = self.lifts[i](xi).permute(0, 3, 1, 2)                 # [B, width, H, W]
            if self.padding > 0:
                xi = F.pad(xi, [0, self.padding, 0, self.padding])
            xs.append(xi)

        # Initial GRU state (zeros for RNN; unused for ATN)
        pH = H + self.padding if self.padding > 0 else H
        pW = W + self.padding if self.padding > 0 else W
        z = torch.zeros(B, self._width, pH, pW, device=x.device)

        # COMPOL blocks
        for blk in self.blocks:
            xs, z = blk(xs, z)

        # Remove padding + project
        outs = []
        for i in range(self.n_proc):
            xi = xs[i]
            if self.padding > 0:
                xi = xi[..., :-self.padding, :-self.padding]
            xi = xi.permute(0, 2, 3, 1)                                 # [B, H, W, width]
            xi = self.projs[i](xi).permute(0, 3, 1, 2)                 # [B, out_c, H, W]
            outs.append(xi)

        return torch.cat(outs, dim=1)                                    # [B, n_proc*out_c, H, W]


class COMPOL1d(nn.Module):
    """
    1-D COMPOL (Sun et al. 2025).

    Input:  [B, n_proc * in_channels, L]
    Output: [B, n_proc * out_channels, L]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_proc: int = 2,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,   # kept for API compat
        aggr_type: str = 'atn',
    ):
        super().__init__()
        self.n_proc = n_proc
        self.aggr_type = aggr_type
        self._width = width

        self.lifts = nn.ModuleList(
            [nn.Linear(in_channels + 1, width) for _ in range(n_proc)]
        )
        self.blocks = nn.ModuleList(
            [COMPOLBlock1d(width, modes, n_proc, aggr_type) for _ in range(n_layers)]
        )
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_channels))
            for _ in range(n_proc)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, total_C, L = x.shape
        C_per = total_C // self.n_proc
        grid = torch.linspace(0, 1, L, device=x.device).view(1, L, 1).expand(B, -1, 1)

        xs = []
        for i in range(self.n_proc):
            xi = x[:, i * C_per:(i + 1) * C_per].permute(0, 2, 1)     # [B, L, C_per]
            xi = torch.cat([xi, grid], dim=-1)                           # [B, L, C+1]
            xi = self.lifts[i](xi).permute(0, 2, 1)                     # [B, width, L]
            xs.append(xi)

        z = torch.zeros(B, self._width, L, device=x.device)

        for blk in self.blocks:
            xs, z = blk(xs, z)

        outs = []
        for i in range(self.n_proc):
            xi = xs[i].permute(0, 2, 1)                                  # [B, L, width]
            xi = self.projs[i](xi).permute(0, 2, 1)                     # [B, out_c, L]
            outs.append(xi)

        return torch.cat(outs, dim=1)
