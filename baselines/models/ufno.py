"""
U-shaped Fourier Neural Operator (U-FNO) — Wen et al. (2022)
https://arxiv.org/abs/2109.03697

Enhances FNO by adding a local multi-scale (U-Net) path inside each layer,
so the model can capture both global (spectral) and local (convolutional)
spatial patterns simultaneously.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import SpectralConv2d, SpectralConv1d


# ---------------------------------------------------------------------------
# 2-D U-FNO
# ---------------------------------------------------------------------------

class LocalPath2d(nn.Module):
    """
    Two-level U-Net local path used inside each U-FNO block.
    Captures short-range features missed by truncated spectral convolutions.
    Uses bilinear upsampling to handle arbitrary (including odd) spatial sizes.
    """

    def __init__(self, width: int):
        super().__init__()
        mid = width * 2
        self.down1 = nn.Sequential(
            nn.Conv2d(width, mid, 3, stride=2, padding=1),
            nn.GroupNorm(8, mid),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(mid, mid * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, mid * 2),
            nn.GELU(),
        )
        self.up1_conv = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 3, padding=1),
            nn.GroupNorm(8, mid),
            nn.GELU(),
        )
        # skip connection from down1 (mid) + up1 output (mid) -> width
        self.up2_conv = nn.Sequential(
            nn.Conv2d(mid * 2, width, 3, padding=1),
            nn.GroupNorm(8, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.down1(x)           # [B, mid, H/2, W/2]
        s2 = self.down2(s1)          # [B, mid*2, H/4, W/4]
        # Upsample to exactly match s1 spatial size
        u1 = F.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1_conv(u1)       # [B, mid, H/2, W/2]
        # Upsample to exactly match x spatial size
        u2 = F.interpolate(torch.cat([u1, s1], dim=1),
                           size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.up2_conv(u2)     # [B, width, H, W]


class UFNOBlock2d(nn.Module):
    """
    One U-FNO layer:
        output = GELU( SpectralConv(x) + LocalPath(x) + W(x) )
    where W is a 1×1 residual convolution and LocalPath is the multi-scale local path.
    """

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = LocalPath2d(width)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.local(x) + self.w(x))


class UFNO2d(nn.Module):
    """
    2-D U-shaped Fourier Neural Operator.

    Args:
        modes1, modes2: Fourier modes in each spatial direction.
        width:          hidden channel width.
        in_channels:    input channels (n_processes for coupled PDEs).
        out_channels:   output channels.
        n_layers:       number of U-FNO blocks.
        padding:        zero-padding before spectral layers.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        in_channels: int = 2,
        out_channels: int = 2,
        n_layers: int = 4,
        padding: int = 9,
    ):
        super().__init__()
        self.padding = padding

        self.fc0 = nn.Linear(in_channels + 2, width)

        self.blocks = nn.ModuleList(
            [UFNOBlock2d(width, modes1, modes2) for _ in range(n_layers)]
        )

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    @staticmethod
    def _get_grid(B: int, H: int, W: int, device) -> torch.Tensor:
        gx = torch.linspace(0, 1, H, device=device).view(1, H, 1, 1).expand(B, -1, W, 1)
        gy = torch.linspace(0, 1, W, device=device).view(1, 1, W, 1).expand(B, H, -1, 1)
        return torch.cat([gx, gy], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = torch.cat([x, self._get_grid(B, H, W, x.device)], dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)         # [B, width, H, W]

        x = F.pad(x, [0, self.padding, 0, self.padding])
        for blk in self.blocks:
            x = blk(x)
        x = x[..., :-self.padding, :-self.padding]

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x).permute(0, 3, 1, 2)
        return x


# ---------------------------------------------------------------------------
# 1-D U-FNO
# ---------------------------------------------------------------------------

class LocalPath1d(nn.Module):
    """Two-level U-Net local path for 1-D signals."""

    def __init__(self, width: int):
        super().__init__()
        mid = width * 2
        self.down1 = nn.Sequential(
            nn.Conv1d(width, mid, 3, stride=2, padding=1), nn.GELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(mid, mid * 2, 3, stride=2, padding=1), nn.GELU()
        )
        self.up1_conv = nn.Sequential(nn.Conv1d(mid * 2, mid, 3, padding=1), nn.GELU())
        self.up2_conv = nn.Conv1d(mid * 2, width, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.down1(x)
        s2 = self.down2(s1)
        u1 = F.interpolate(s2, size=s1.shape[-1:], mode="linear", align_corners=False)
        u1 = self.up1_conv(u1)
        u2 = F.interpolate(torch.cat([u1, s1], dim=1),
                           size=x.shape[-1:], mode="linear", align_corners=False)
        return self.up2_conv(u2)


class UFNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = LocalPath1d(width)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.local(x) + self.w(x))


class UFNO1d(nn.Module):
    """1-D U-shaped Fourier Neural Operator."""

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        in_channels: int = 2,
        out_channels: int = 2,
        n_layers: int = 4,
    ):
        super().__init__()
        self.fc0 = nn.Linear(in_channels + 1, width)
        self.blocks = nn.ModuleList(
            [UFNOBlock1d(width, modes) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        grid = torch.linspace(0, 1, L, device=x.device).view(1, L, 1).expand(B, -1, 1)
        x = torch.cat([x.permute(0, 2, 1), grid], dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).permute(0, 2, 1)
