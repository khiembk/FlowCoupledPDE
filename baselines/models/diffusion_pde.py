"""
Conditional Diffusion model for PDE prediction using the EDM framework.

Wraps SongUNet from DiffusionPDE (Huang et al., 2024) as a conditional
denoising model:
  - Training: learns p(target | source) via EDM denoising
  - Inference: forward(source) runs EDM ODE (Heun) sampling

Input/output interface matches other baselines:
  pred = model(x)   x,pred: [B, n_proc, H, W]

Repo dependency: /home/u.kt348068/khiemtt/DiffusionPDE must exist.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_DPDE_ROOT = Path("/home/u.kt348068/khiemtt/DiffusionPDE")
if str(_DPDE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DPDE_ROOT))

from training.networks import SongUNet   # noqa: E402


class DiffusionPDE(nn.Module):
    """
    EDM-based conditional diffusion model for coupled PDE prediction.

    Architecture:
      SongUNet (DDPM++) with in_channels = 2*n_proc:
        first n_proc channels  = c_in * noisy_target  (scaled)
        last  n_proc channels  = normalized source     (condition)
      out_channels = n_proc  (denoised target prediction)

    Data normalization: data_mean / data_std are registered buffers
    set from the training set before training begins.
    """

    def __init__(
        self,
        n_proc:            int   = 2,
        img_resolution:    int   = 64,
        model_channels:    int   = 64,
        channel_mult             = (1, 2, 2, 2),
        num_blocks:        int   = 4,
        attn_resolutions         = (16,),
        dropout:           float = 0.10,
        # EDM parameters
        sigma_data:        float = 0.5,
        sigma_min:         float = 0.002,
        sigma_max:         float = 80.0,
        P_mean:            float = -1.2,
        P_std:             float = 1.2,
        # Inference
        num_steps:         int   = 20,
        rho:               float = 7.0,
    ):
        super().__init__()

        self.n_proc         = n_proc
        self.img_resolution = img_resolution
        self.sigma_data     = sigma_data
        self.sigma_min      = sigma_min
        self.sigma_max      = sigma_max
        self.P_mean         = P_mean
        self.P_std          = P_std
        self.num_steps      = num_steps
        self.rho            = rho

        self.unet = SongUNet(
            img_resolution   = img_resolution,
            in_channels      = 2 * n_proc,   # noisy target + source condition
            out_channels     = n_proc,
            model_channels   = model_channels,
            channel_mult     = list(channel_mult),
            num_blocks       = num_blocks,
            attn_resolutions = list(attn_resolutions),
            dropout          = dropout,
            label_dim        = 0,
            embedding_type   = "positional",
            encoder_type     = "standard",
            decoder_type     = "standard",
        )

        # Per-channel data normalization buffers (set before training)
        self.register_buffer("data_mean", torch.zeros(n_proc))
        self.register_buffer("data_std",  torch.ones(n_proc))

    # ── normalisation ────────────────────────────────────────────────────────

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        m = self.data_mean.view(1, -1, 1, 1)
        s = self.data_std.view(1,  -1, 1, 1)
        return (x - m) / s

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        m = self.data_mean.view(1, -1, 1, 1)
        s = self.data_std.view(1,  -1, 1, 1)
        return x * s + m

    # ── EDM denoiser ─────────────────────────────────────────────────────────

    def _denoise(self, x_noisy: torch.Tensor,
                 cond: torch.Tensor,
                 sigma: torch.Tensor) -> torch.Tensor:
        """
        x_noisy, cond: [B, n_proc, H, W]  (normalised)
        sigma:         [B]
        returns:       [B, n_proc, H, W]  (denoised, normalised)
        """
        s = sigma.view(-1, 1, 1, 1).to(x_noisy.dtype)

        c_skip  = self.sigma_data ** 2 / (s ** 2 + self.sigma_data ** 2)
        c_out   = s * self.sigma_data / (s ** 2 + self.sigma_data ** 2).sqrt()
        c_in    = 1.0 / (s ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = sigma.log() / 4.0

        net_in = torch.cat([c_in * x_noisy, cond], dim=1)  # [B, 2*n_proc, H, W]
        F_x    = self.unet(net_in, c_noise)
        return c_skip * x_noisy + c_out * F_x

    # ── training loss ─────────────────────────────────────────────────────────

    def compute_loss(self, source: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
        """
        EDM denoising loss conditioned on source.
        source, target: [B, n_proc, H, W]  (raw, un-normalised)
        """
        src  = self._norm(source)
        tgt  = self._norm(target)
        B    = tgt.shape[0]
        dev  = tgt.device

        sigma = (torch.randn(B, device=dev) * self.P_std + self.P_mean).exp()
        noise = torch.randn_like(tgt)
        D_yn  = self._denoise(tgt + noise * sigma.view(B, 1, 1, 1), src, sigma)

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return (weight.view(B, 1, 1, 1) * (D_yn - tgt) ** 2).mean()

    # ── ODE sampling ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _sample(self, source_norm: torch.Tensor,
                num_steps: int) -> torch.Tensor:
        """
        Heun's 2nd-order ODE sampler (Karras et al., 2022).
        source_norm: [B, n_proc, H, W]  (normalised)
        returns:     [B, n_proc, H, W]  (normalised target)
        """
        B, C, H, W = source_norm.shape[0], self.n_proc, self.img_resolution, self.img_resolution
        dev   = source_norm.device
        dtype = source_norm.dtype

        # Karras sigma schedule
        rho    = self.rho
        idx    = torch.arange(num_steps, device=dev, dtype=torch.float64)
        t_max  = self.sigma_max ** (1 / rho)
        t_min  = max(self.sigma_min, 1e-4) ** (1 / rho)
        sigmas = ((t_max + idx / (num_steps - 1) * (t_min - t_max)) ** rho).to(dtype)
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])   # append σ=0

        x = torch.randn(B, C, H, W, device=dev, dtype=dtype) * sigmas[0]

        for i, (s_cur, s_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            D_cur  = self._denoise(x, source_norm, s_cur.expand(B))
            d_cur  = (x - D_cur) / s_cur
            x_next = x + (s_next - s_cur) * d_cur

            if i < num_steps - 1 and s_next > 0:
                D_next  = self._denoise(x_next, source_norm, s_next.expand(B))
                d_next  = (x_next - D_next) / s_next
                x_next  = x + (s_next - s_cur) * (0.5 * d_cur + 0.5 * d_next)

            x = x_next

        return x

    # ── public interface ──────────────────────────────────────────────────────

    def forward(self, source: torch.Tensor,
                num_steps: int | None = None) -> torch.Tensor:
        """
        Predict next state from current state.
        source: [B, n_proc, H, W]
        returns [B, n_proc, H, W]
        """
        n = num_steps if num_steps is not None else self.num_steps
        pred_norm = self._sample(self._norm(source), n)
        return self._denorm(pred_norm)
