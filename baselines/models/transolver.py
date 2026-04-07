"""
Transolver — Wu et al. (2024)
https://arxiv.org/abs/2402.02366

"Transolver: A Fast Transformer Solver for PDEs on General Geometries"

Key idea: Physics-Attention slices the spatially dense field into a small set
of *physics tokens* via learned soft assignments; a standard Transformer
processes those compact tokens; the tokens are then projected back to the
full resolution.  This avoids the O(N²) cost of full spatial attention.

For coupled PDEs, all process channels are concatenated on input.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PhysicsAttention(nn.Module):
    """
    Slice the dense input field into M physics tokens.

    Each spatial point i is assigned to M tokens with attention weights
    a_{i,m} (softmax over M); the token is the weighted sum of input features.
    During decoding, the per-point output is a weighted sum of token outputs
    using the *same* soft assignment.
    """

    def __init__(self, dim: int, n_slices: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_slices = n_slices
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Projection for physics tokens (query side)
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        # Learned slice weights: each point -> M assignment logits
        self.slice_fc = nn.Linear(dim, n_slices)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, N, dim]   (N = H*W spatial points)
        Returns:
            out: [B, N, dim]
        """
        B, N, D = x.shape
        M = self.n_slices
        H = self.n_heads
        hd = self.head_dim

        # Soft assignment [B, N, M]
        weights = F.softmax(self.slice_fc(x), dim=1)      # softmax over spatial dim

        # Aggregate to physics tokens: [B, M, D]
        tokens = torch.bmm(weights.transpose(1, 2), x)    # [B, M, D]

        # Multi-head self-attention on physics tokens
        qkv = self.to_qkv(tokens).reshape(B, M, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                           # each [B, H, M, hd]
        scale = math.sqrt(hd)
        attn = (q @ k.transpose(-2, -1)) / scale          # [B, H, M, M]
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        token_out = (attn @ v).transpose(1, 2).reshape(B, M, D)  # [B, M, D]
        token_out = self.out_proj(token_out)

        # Deslice: weighted sum back to spatial resolution [B, N, D]
        out = torch.bmm(weights, token_out)                # [B, N, D]
        return out


class TransolverBlock(nn.Module):
    """
    One Transolver layer: Physics-Attention + FFN with pre-norm.
    """

    def __init__(self, dim: int, n_slices: int, n_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, n_slices, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 2-D Transolver
# ---------------------------------------------------------------------------

class Transolver2d(nn.Module):
    """
    2-D Transolver.

    Args:
        in_channels:  number of input channels (n_processes for coupled PDEs).
        out_channels: number of output channels.
        dim:          model hidden dimension.
        n_slices:     number of physics tokens (M).
        n_heads:      attention heads.
        n_layers:     number of Transolver blocks.
        mlp_ratio:    FFN expansion ratio.
        dropout:      dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int = 128,
        n_slices: int = 32,
        n_heads: int = 8,
        n_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        # +2 for (x, y) grid coordinates
        self.embed = nn.Linear(in_channels + 2, dim)
        self.blocks = nn.ModuleList([
            TransolverBlock(dim, n_slices, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels)

    @staticmethod
    def _make_grid(H: int, W: int, device) -> torch.Tensor:
        gy = torch.linspace(0, 1, H, device=device)
        gx = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [N, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        N = H * W

        grid = self._make_grid(H, W, x.device)                  # [N, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1)              # [B, N, 2]

        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)         # [B, N, C]
        x_flat = torch.cat([x_flat, grid], dim=-1)               # [B, N, C+2]
        x_flat = self.embed(x_flat)                              # [B, N, dim]

        for blk in self.blocks:
            x_flat = blk(x_flat)

        x_flat = self.norm(x_flat)                               # [B, N, dim]
        out = self.head(x_flat)                                  # [B, N, out_c]
        return out.permute(0, 2, 1).view(B, -1, H, W)


# ---------------------------------------------------------------------------
# 1-D Transolver
# ---------------------------------------------------------------------------

class Transolver1d(nn.Module):
    """1-D Transolver."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int = 128,
        n_slices: int = 32,
        n_heads: int = 8,
        n_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Linear(in_channels + 1, dim)
        self.blocks = nn.ModuleList([
            TransolverBlock(dim, n_slices, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        B, C, L = x.shape
        grid = torch.linspace(0, 1, L, device=x.device).view(1, L, 1).expand(B, -1, 1)
        x_t = torch.cat([x.permute(0, 2, 1), grid], dim=-1)    # [B, L, C+1]
        x_t = self.embed(x_t)                                   # [B, L, dim]
        for blk in self.blocks:
            x_t = blk(x_t)
        x_t = self.norm(x_t)
        out = self.head(x_t)                                    # [B, L, out_c]
        return out.permute(0, 2, 1)                             # [B, out_c, L]
