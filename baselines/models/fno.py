"""
Fourier Neural Operator (FNO) — Li et al. (2021)
https://arxiv.org/abs/2010.08895

For coupled PDEs: all process channels are concatenated along the channel dim.
Supports both 1-D and 2-D problems.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2-D FNO
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """2-D Fourier integral operator: FFT -> multiply Fourier modes -> IFFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: [B, in_c, h, w]   w: [in_c, out_c, h, w]
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNOBlock2d(nn.Module):
    """One FNO layer: SpectralConv + pointwise Conv + GELU."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x) + self.w(x))


class FNO2d(nn.Module):
    """
    2-D Fourier Neural Operator.

    Args:
        modes1, modes2: number of Fourier modes retained in each spatial direction.
        width:          hidden channel width.
        in_channels:    number of input channels (n_processes × 1 for coupled PDEs).
        out_channels:   number of output channels.
        n_layers:       number of FNO blocks.
        padding:        zero-padding applied before spectral layers (edge effects).
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        in_channels: int = 2,
        out_channels: int = 2,
        n_layers: int = 4,
        padding: int = 9,
    ):
        super().__init__()
        self.padding = padding

        # Lifting: (in_channels + 2 grid coords) -> width
        self.fc0 = nn.Linear(in_channels + 2, width)

        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes1, modes2) for _ in range(n_layers)]
        )

        # Projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    @staticmethod
    def _get_grid(B: int, H: int, W: int, device) -> torch.Tensor:
        gx = torch.linspace(0, 1, H, device=device).view(1, H, 1, 1).expand(B, -1, W, 1)
        gy = torch.linspace(0, 1, W, device=device).view(1, 1, W, 1).expand(B, H, -1, 1)
        return torch.cat([gx, gy], dim=-1)  # [B, H, W, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)                     # [B, H, W, C]
        x = torch.cat([x, self._get_grid(B, H, W, x.device)], dim=-1)  # [B, H, W, C+2]
        x = self.fc0(x).permute(0, 3, 1, 2)           # [B, width, H, W]

        x = F.pad(x, [0, self.padding, 0, self.padding])
        for blk in self.blocks:
            x = blk(x)
        x = x[..., :-self.padding, :-self.padding]

        x = x.permute(0, 2, 3, 1)                     # [B, H, W, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x).permute(0, 3, 1, 2)           # [B, out_channels, H, W]
        return x


# ---------------------------------------------------------------------------
# 1-D FNO
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """1-D Fourier integral operator."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.weights.shape[1], L // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=L)


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x) + self.w(x))


class FNO1d(nn.Module):
    """
    1-D Fourier Neural Operator.

    Args:
        modes:       number of Fourier modes retained.
        width:       hidden channel width.
        in_channels: number of input channels.
        out_channels:number of output channels.
        n_layers:    number of FNO blocks.
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        in_channels: int = 2,
        out_channels: int = 2,
        n_layers: int = 4,
    ):
        super().__init__()
        self.fc0 = nn.Linear(in_channels + 1, width)  # +1 for 1-D grid coord
        self.blocks = nn.ModuleList(
            [FNOBlock1d(width, modes) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        B, C, L = x.shape
        grid = torch.linspace(0, 1, L, device=x.device).view(1, L, 1).expand(B, -1, 1)
        x = x.permute(0, 2, 1)                      # [B, L, C]
        x = torch.cat([x, grid], dim=-1)             # [B, L, C+1]
        x = self.fc0(x).permute(0, 2, 1)            # [B, width, L]
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 2, 1)                      # [B, L, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x).permute(0, 2, 1)            # [B, out_channels, L]
        return x
