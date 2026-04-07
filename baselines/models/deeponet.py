"""
Deep Operator Network (DeepONet) — Lu et al. (2021)
https://arxiv.org/abs/1910.03193

Adapted for grid-based PDE data using a CNN branch network and a
coordinate-MLP trunk network. For coupled PDEs, all process channels
are concatenated in the branch input; the output has the same number
of channels (multi-output DeepONet with one trunk per output channel).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBranch2d(nn.Module):
    """
    CNN branch: encodes the input function field (sensor values) into a
    p-dimensional basis coefficient vector.
    Input:  [B, in_channels, H, W]
    Output: [B, p]
    """

    def __init__(self, in_channels: int, p: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 2, p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, p]


class TrunkMLP2d(nn.Module):
    """
    Trunk network: encodes query coordinates (x, y) into basis functions.
    Input:  [N, 2]  (N = H * W query points)
    Output: [N, p * out_channels]  (p basis functions per output channel)
    """

    def __init__(self, p: int, out_channels: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, p * out_channels))
        self.net = nn.Sequential(*layers)
        self.p = p
        self.out_channels = out_channels

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [N, 2]
        out = self.net(coords)                     # [N, p * out_channels]
        return out.view(-1, self.out_channels, self.p)  # [N, out_c, p]


class DeepONet2d(nn.Module):
    """
    2-D DeepONet for grid data.

    The branch network encodes the input field; the trunk network encodes
    spatial coordinates. The output at each grid point is computed as the
    inner product between the branch features and trunk basis functions:

        u(x)_c = sum_k  branch_k  *  trunk(x)_{c,k}

    Args:
        in_channels:  input channels (n_processes for coupled PDEs).
        out_channels: output channels.
        p:            number of basis functions (branch/trunk dimension).
        width:        CNN branch hidden width.
        trunk_hidden: trunk MLP hidden dimension.
        trunk_depth:  trunk MLP depth.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        p: int = 128,
        width: int = 64,
        trunk_hidden: int = 128,
        trunk_depth: int = 4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.p = p

        self.branch = CNNBranch2d(in_channels, p, width)
        self.trunk = TrunkMLP2d(p, out_channels, trunk_hidden, trunk_depth)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    @staticmethod
    def _make_grid(H: int, W: int, device) -> torch.Tensor:
        """Return normalised grid coordinates [H*W, 2]."""
        gy = torch.linspace(0, 1, H, device=device)
        gx = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [H*W, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        N = H * W

        branch_feat = self.branch(x)               # [B, p]
        coords = self._make_grid(H, W, x.device)   # [N, 2]
        trunk_feat = self.trunk(coords)             # [N, out_c, p]

        # Inner product: [B, p] x [N, out_c, p] -> [B, N, out_c]
        out = torch.einsum("bp,ncp->bnc", branch_feat, trunk_feat)
        out = out + self.bias                       # [B, N, out_c]
        out = out.permute(0, 2, 1).view(B, self.out_channels, H, W)
        return out


# ---------------------------------------------------------------------------
# 1-D DeepONet
# ---------------------------------------------------------------------------

class MLPBranch1d(nn.Module):
    """MLP branch for 1-D problems (sensor values flattened)."""

    def __init__(self, in_channels: int, L: int, p: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * L, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkMLP1d(nn.Module):
    """Trunk network for 1-D problems."""

    def __init__(self, p: int, out_channels: int, hidden: int = 128, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, p * out_channels))
        self.net = nn.Sequential(*layers)
        self.p = p
        self.out_channels = out_channels

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out = self.net(coords)                                 # [L, p * out_c]
        return out.view(-1, self.out_channels, self.p)         # [L, out_c, p]


class DeepONet1d(nn.Module):
    """
    1-D DeepONet.

    Args:
        in_channels:  input channels.
        out_channels: output channels.
        L:            sequence length (needed by MLP branch).
        p:            number of basis functions.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        L: int = 256,
        p: int = 128,
        branch_hidden: int = 256,
        trunk_hidden: int = 128,
        trunk_depth: int = 4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.p = p

        self.branch = MLPBranch1d(in_channels, L, p, branch_hidden)
        self.trunk = TrunkMLP1d(p, out_channels, trunk_hidden, trunk_depth)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        B, C, L = x.shape
        coords = torch.linspace(0, 1, L, device=x.device).unsqueeze(-1)  # [L, 1]

        branch_feat = self.branch(x)            # [B, p]
        trunk_feat = self.trunk(coords)         # [L, out_c, p]

        out = torch.einsum("bp,lcp->blc", branch_feat, trunk_feat)
        out = out + self.bias                   # [B, L, out_c]
        return out.permute(0, 2, 1)             # [B, out_c, L]
