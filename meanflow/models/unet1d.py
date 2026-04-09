"""
SongUNet1d — 1-D analogue of SongUNet for flow matching on sequence data
(e.g. Lotka-Volterra).

Interface mirrors SongUNet exactly:
    forward(x, time_steps, aug_cond=None)
    where time_steps = (t, h), both [B] tensors.
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Linear, GroupNorm, PositionalEmbedding, FourierEmbedding, QKVAttention, weight_init
from .groupnorm import group_norm


# ---------------------------------------------------------------------------
# 1-D conv with optional up/downsampling
# ---------------------------------------------------------------------------

class Conv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=3, bias=True,
        up=False, down=False,
        init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.kernel = kernel

        if kernel:
            fan_in  = in_channels  * kernel
            fan_out = out_channels * kernel
            self.weight = nn.Parameter(
                weight_init([out_channels, in_channels, kernel],
                            mode=init_mode, fan_in=fan_in, fan_out=fan_out) * init_weight
            )
            self.bias = nn.Parameter(
                weight_init([out_channels], mode=init_mode, fan_in=fan_in, fan_out=fan_out) * init_bias
            ) if bias else None
        else:
            self.weight = None
            self.bias   = None

    def forward(self, x):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.down:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        if self.weight is not None:
            pad = self.kernel // 2
            x = F.conv1d(x, self.weight.to(x.dtype),
                         self.bias.to(x.dtype) if self.bias is not None else None,
                         padding=pad)
        return x


# ---------------------------------------------------------------------------
# 1-D residual block with optional self-attention
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, emb_channels,
        up=False, down=False,
        attention=False, num_heads=1,
        dropout=0.0, skip_scale=1.0, eps=1e-5,
        init=None, init_zero=None, init_attn=None,
    ):
        super().__init__()
        init      = init      or dict(init_mode='kaiming_normal')
        init_zero = init_zero or dict(init_mode='kaiming_normal', init_weight=1e-5)
        init_attn = init_attn or init

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_heads    = num_heads if attention else 0
        self.dropout      = dropout
        self.skip_scale   = skip_scale

        self.norm0  = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0  = Conv1d(in_channels, out_channels, kernel=3, up=up, down=down, **init)
        self.affine = Linear(emb_channels, out_channels * 2, **init)
        self.norm1  = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1  = Conv1d(out_channels, out_channels, kernel=3, **init_zero)

        if in_channels != out_channels or up or down:
            self.skip = Conv1d(in_channels, out_channels, kernel=1, up=up, down=down, **init)
        else:
            self.skip = None

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv   = Conv1d(out_channels, out_channels * 3, kernel=1, **init_attn)
            self.proj  = Conv1d(out_channels, out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).to(x.dtype)   # [B, 2C, 1]
        scale, shift = params.chunk(2, dim=1)
        x = F.silu(shift + self.norm1(x) * (scale + 1))

        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = x + (self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            qkv = self.qkv(self.norm2(x))          # [B, 3C, L]
            a   = QKVAttention(n_heads=self.num_heads)(qkv)
            x   = self.proj(a) + x
            x   = x * self.skip_scale

        return x


# ---------------------------------------------------------------------------
# SongUNet1d
# ---------------------------------------------------------------------------

class SongUNet1d(nn.Module):
    """
    1-D UNet for flow matching on sequences.

    Args:
        seq_len           (int)  : Input sequence length L (must be divisible by 2^(n_levels-1)).
        in_channels       (int)  : Input channels (e.g. 2 for concatenated processes).
        out_channels      (int)  : Output channels (e.g. 1 for predicted velocity).
        model_channels    (int)  : Base channel width. Default 64.
        channel_mult      (tuple): Per-level channel multipliers. Default (1,2,2,2).
        channel_mult_emb  (int)  : Time-embedding width = model_channels * channel_mult_emb.
        num_blocks        (int)  : ResBlock1d count per encoder/decoder level.
        attn_resolutions  (tuple): Sequence lengths at which to apply self-attention.
        dropout           (float): Dropout probability.
        channel_mult_noise(int)  : Noise-embedding size = model_channels * channel_mult_noise.
        embedding_type    (str)  : 'positional' or 'fourier'.
        use_checkpoint    (bool) : Unused; kept for interface parity with SongUNet.
    """

    def __init__(
        self,
        seq_len,
        in_channels,
        out_channels,
        model_channels    = 64,
        channel_mult      = (1, 2, 2, 2),
        channel_mult_emb  = 4,
        num_blocks        = 2,
        attn_resolutions  = (32,),
        dropout           = 0.1,
        channel_mult_noise= 2,
        embedding_type    = 'positional',
        use_checkpoint    = False,
    ):
        super().__init__()
        assert embedding_type in ('positional', 'fourier')

        emb_channels   = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise

        init      = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=math.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            dropout=dropout,
            skip_scale=math.sqrt(0.5),
            eps=1e-6,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # --- time embedding ---
        if embedding_type == 'positional':
            self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        else:
            self.map_noise = FourierEmbedding(num_channels=noise_channels)
        self.map_layer0 = Linear(noise_channels * 2, emb_channels, **init)
        self.map_layer1 = Linear(emb_channels, emb_channels, **init)

        n_levels = len(channel_mult)

        # --- encoder ---
        # enc_blocks[level] = nn.ModuleList of ResBlock1d
        # enc_down[level]   = downsampling ResBlock1d (None for level 0)
        self.enc_in   = Conv1d(in_channels, model_channels * channel_mult[0], kernel=3, **init)
        self.enc_blocks = nn.ModuleList()
        self.enc_downs  = nn.ModuleList()
        self._enc_channels = []   # output channel count per level (after blocks)

        cout = model_channels * channel_mult[0]
        for level in range(n_levels):
            res  = seq_len >> level
            mult = channel_mult[level]
            attn = res in attn_resolutions

            # downsampling (skip for level 0, already done by enc_in)
            if level > 0:
                self.enc_downs.append(
                    ResBlock1d(cout, cout, down=True, attention=False, num_heads=1, **block_kwargs)
                )
            else:
                self.enc_downs.append(None)

            blocks = nn.ModuleList()
            for idx in range(num_blocks):
                cin  = cout
                cout = model_channels * mult
                blocks.append(ResBlock1d(cin, cout, attention=attn, num_heads=1, **block_kwargs))
            self.enc_blocks.append(blocks)
            self._enc_channels.append(cout)

        # --- decoder ---
        # dec_blocks[level] = nn.ModuleList of ResBlock1d (takes skip-cat input at first block)
        # dec_ups[level]    = upsampling ResBlock1d (None for last level)
        self.dec_blocks = nn.ModuleList()
        self.dec_ups    = nn.ModuleList()

        for level in reversed(range(n_levels)):
            res  = seq_len >> level
            mult = channel_mult[level]
            attn = res in attn_resolutions
            skip_ch = self._enc_channels[level]

            blocks = nn.ModuleList()
            for idx in range(num_blocks + 1):
                cin  = cout + (skip_ch if idx == 0 else 0)
                cout = model_channels * mult
                blocks.append(ResBlock1d(cin, cout, attention=attn, num_heads=1, **block_kwargs))
            self.dec_blocks.append(blocks)   # stored in reversed order

            if level > 0:
                self.dec_ups.append(
                    ResBlock1d(cout, cout, up=True, attention=False, num_heads=1, **block_kwargs)
                )
            else:
                self.dec_ups.append(None)

        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv1d(cout, out_channels, kernel=3, **init_zero)

    # ------------------------------------------------------------------

    def _time_emb(self, t, h):
        emb_t = self.map_noise(t)
        emb_t = emb_t.reshape(emb_t.shape[0], 2, -1).flip(1).reshape(*emb_t.shape)
        emb_h = self.map_noise(h)
        emb_h = emb_h.reshape(emb_h.shape[0], 2, -1).flip(1).reshape(*emb_h.shape)
        emb = torch.cat([emb_t, emb_h], dim=1)
        return F.silu(self.map_layer1(F.silu(self.map_layer0(emb))))

    def forward(self, x, time_steps, aug_cond=None):
        """
        Args:
            x          : [B, in_channels, L]
            time_steps : tuple (t, h), each [B]
            aug_cond   : ignored (kept for interface compatibility)
        Returns:
            [B, out_channels, L]
        """
        assert isinstance(time_steps, tuple) and len(time_steps) == 2
        t, h = time_steps
        emb = self._time_emb(t, h)

        # encoder — save one skip per level
        x = self.enc_in(x)
        skips = []
        for level, (down, blocks) in enumerate(zip(self.enc_downs, self.enc_blocks)):
            if down is not None:
                x = down(x, emb)
            for blk in blocks:
                x = blk(x, emb)
            skips.append(x)

        # decoder — reverse levels
        for dec_blocks, dec_up in zip(self.dec_blocks, self.dec_ups):
            skip = skips.pop()
            for idx, blk in enumerate(dec_blocks):
                if idx == 0:
                    x = torch.cat([x, skip], dim=1)
                x = blk(x, emb)
            if dec_up is not None:
                x = dec_up(x, emb)

        return self.out_conv(F.silu(self.out_norm(x)))
