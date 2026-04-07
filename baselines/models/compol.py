"""
COMPOL — Sun et al. (2025)
arXiv:2501.17296  "COMPOL: A Unified Neural Operator Framework for
Scalable Multi-Physics Simulations"

COMPOL is specifically designed for coupled / multi-physics PDE systems.
Architecture (as described in the paper):
  1. Process-specific FNO encoders — each process has its own spectral layers
     that learn the intra-process dynamics in the Fourier domain.
  2. Cross-process attention (COMPOL coupling layers) — enable information
     exchange between process representations at each layer.
  3. Process-specific decoders — project back to the output field.

Supports both 1-D and 2-D spatial domains.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import SpectralConv2d, SpectralConv1d


# ---------------------------------------------------------------------------
# Cross-process attention (coupling layer)
# ---------------------------------------------------------------------------

class CrossProcessAttention(nn.Module):
    """
    Couples n_proc process representations via cross-attention.

    Each process acts as a *query* and attends to the flattened spatial tokens
    from *all* processes (keys/values). This makes information from every
    process available when updating any single process.

    For efficiency, we average-pool the spatial dimensions first so that
    the sequence length equals n_proc, then broadcast the update back.
    """

    def __init__(self, dim: int, n_proc: int, n_heads: int = 4):
        super().__init__()
        self.n_proc = n_proc
        self.n_heads = n_heads
        self.head_dim = max(dim // n_heads, 1)
        inner = self.head_dim * n_heads

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, 2 * inner, bias=False)
        self.out_proj = nn.Linear(inner, dim)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            xs: list of n_proc tensors, each [B, C, *spatial]
        Returns:
            list of n_proc tensors with same shapes, coupling applied.
        """
        shapes = [x.shape for x in xs]
        B = shapes[0][0]
        C = shapes[0][1]
        H = self.n_heads
        hd = self.head_dim

        # Global pool each process -> [B, C] summary token, then stack [B, n_proc, C]
        summaries = torch.stack(
            [x.flatten(2).mean(dim=-1) for x in xs], dim=1
        )                                                       # [B, n_proc, C]
        summaries = self.norm(summaries)

        # Each process queries all process summaries
        q = self.to_q(summaries).reshape(B, self.n_proc, H, hd).permute(0, 2, 1, 3)
        kv = self.to_kv(summaries)                             # [B, n_proc, 2*inner]
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, self.n_proc, H, hd).permute(0, 2, 1, 3)
        v = v.reshape(B, self.n_proc, H, hd).permute(0, 2, 1, 3)

        scale = math.sqrt(hd)
        attn = F.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)  # [B, H, n, n]
        ctx = (attn @ v).permute(0, 2, 1, 3).reshape(B, self.n_proc, H * hd)
        delta = self.out_proj(ctx)                             # [B, n_proc, C]

        # Broadcast spatial correction: add delta to each spatial location
        outs = []
        for i, x in enumerate(xs):
            d = delta[:, i].reshape(B, C, *([1] * (x.dim() - 2)))  # [B, C, 1, ...]
            outs.append(x + d)
        return outs


# ---------------------------------------------------------------------------
# COMPOL layer = per-process FNO block + cross-process attention
# ---------------------------------------------------------------------------

class COMPOLBlock2d(nn.Module):
    """One COMPOL layer for 2-D spatial fields."""

    def __init__(self, width: int, modes1: int, modes2: int, n_proc: int, n_heads: int = 4):
        super().__init__()
        # Per-process intra-process FNO blocks
        self.fno_blocks = nn.ModuleList([
            nn.ModuleList([
                SpectralConv2d(width, width, modes1, modes2),
                nn.Conv2d(width, width, 1),
            ])
            for _ in range(n_proc)
        ])
        self.coupling = CrossProcessAttention(width, n_proc, n_heads)
        self.norms = nn.ModuleList([nn.GroupNorm(min(8, width), width) for _ in range(n_proc)])

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        # Intra-process FNO step
        xs_new = []
        for i, x in enumerate(xs):
            sp, pw = self.fno_blocks[i]
            xs_new.append(F.gelu(self.norms[i](sp(x) + pw(x))))
        # Cross-process coupling
        xs_new = self.coupling(xs_new)
        return xs_new


class COMPOLBlock1d(nn.Module):
    """One COMPOL layer for 1-D spatial signals."""

    def __init__(self, width: int, modes: int, n_proc: int, n_heads: int = 4):
        super().__init__()
        self.fno_blocks = nn.ModuleList([
            nn.ModuleList([
                SpectralConv1d(width, width, modes),
                nn.Conv1d(width, width, 1),
            ])
            for _ in range(n_proc)
        ])
        self.coupling = CrossProcessAttention(width, n_proc, n_heads)
        self.norms = nn.ModuleList([nn.GroupNorm(min(8, width), width) for _ in range(n_proc)])

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs_new = []
        for i, x in enumerate(xs):
            sp, pw = self.fno_blocks[i]
            xs_new.append(F.gelu(self.norms[i](sp(x) + pw(x))))
        xs_new = self.coupling(xs_new)
        return xs_new


# ---------------------------------------------------------------------------
# Full COMPOL models
# ---------------------------------------------------------------------------

class COMPOL2d(nn.Module):
    """
    2-D COMPOL: process-specific FNO encoders + cross-process attention + decoders.

    For coupled PDEs with n_proc processes.

    Input:  [B, n_proc * in_channels, H, W]  (processes concatenated)
    Output: [B, n_proc * out_channels, H, W]

    Args:
        in_channels:  channels per process (typically 1).
        out_channels: output channels per process (typically 1).
        n_proc:       number of coupled processes.
        modes1/2:     Fourier modes per direction.
        width:        hidden channel width.
        n_layers:     number of COMPOL blocks.
        n_heads:      attention heads for cross-process coupling.
        padding:      zero-padding applied before spectral layers.
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
        n_heads: int = 4,
        padding: int = 9,
    ):
        super().__init__()
        self.n_proc = n_proc
        self.padding = padding

        # Per-process lifting: (in_channels + 2 grid coords) -> width
        self.lifts = nn.ModuleList(
            [nn.Linear(in_channels + 2, width) for _ in range(n_proc)]
        )
        self.blocks = nn.ModuleList(
            [COMPOLBlock2d(width, modes1, modes2, n_proc, n_heads) for _ in range(n_layers)]
        )
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_channels))
            for _ in range(n_proc)
        ])

    @staticmethod
    def _get_grid(B: int, H: int, W: int, device) -> torch.Tensor:
        gx = torch.linspace(0, 1, H, device=device).view(1, H, 1, 1).expand(B, -1, W, 1)
        gy = torch.linspace(0, 1, W, device=device).view(1, 1, W, 1).expand(B, H, -1, 1)
        return torch.cat([gx, gy], dim=-1)  # [B, H, W, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, total_C, H, W = x.shape
        C_per = total_C // self.n_proc

        grid = self._get_grid(B, H, W, x.device)  # [B, H, W, 2]

        # Lift each process
        xs = []
        for i in range(self.n_proc):
            xi = x[:, i * C_per:(i + 1) * C_per]             # [B, C_per, H, W]
            xi = xi.permute(0, 2, 3, 1)                       # [B, H, W, C_per]
            xi = torch.cat([xi, grid], dim=-1)                 # [B, H, W, C_per+2]
            xi = self.lifts[i](xi).permute(0, 3, 1, 2)        # [B, width, H, W]
            if self.padding > 0:
                xi = F.pad(xi, [0, self.padding, 0, self.padding])
            xs.append(xi)

        # COMPOL blocks
        for blk in self.blocks:
            xs = blk(xs)

        # Remove padding + project
        outs = []
        for i in range(self.n_proc):
            xi = xs[i]
            if self.padding > 0:
                xi = xi[..., :-self.padding, :-self.padding]
            xi = xi.permute(0, 2, 3, 1)                       # [B, H, W, width]
            xi = self.projs[i](xi).permute(0, 3, 1, 2)        # [B, out_c, H, W]
            outs.append(xi)

        return torch.cat(outs, dim=1)                          # [B, n_proc*out_c, H, W]


class COMPOL1d(nn.Module):
    """
    1-D COMPOL.

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
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_proc = n_proc

        self.lifts = nn.ModuleList(
            [nn.Linear(in_channels + 1, width) for _ in range(n_proc)]
        )
        self.blocks = nn.ModuleList(
            [COMPOLBlock1d(width, modes, n_proc, n_heads) for _ in range(n_layers)]
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
            xi = x[:, i * C_per:(i + 1) * C_per].permute(0, 2, 1)  # [B, L, C_per]
            xi = torch.cat([xi, grid], dim=-1)                       # [B, L, C_per+1]
            xi = self.lifts[i](xi).permute(0, 2, 1)                  # [B, width, L]
            xs.append(xi)

        for blk in self.blocks:
            xs = blk(xs)

        outs = []
        for i in range(self.n_proc):
            xi = xs[i].permute(0, 2, 1)                              # [B, L, width]
            xi = self.projs[i](xi).permute(0, 2, 1)                  # [B, out_c, L]
            outs.append(xi)

        return torch.cat(outs, dim=1)
