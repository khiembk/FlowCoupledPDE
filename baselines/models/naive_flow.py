"""
Naive (independent) Mean-Flow model for coupled PDE benchmarks.

N independent UNet networks, one per process. Each network receives only its
own process as input (in_channels=1) with no cross-process coupling.
Training uses the same MeanFlow local (JVP-based) + global objectives as
CoupledFlow.

Approximate parameter counts per network:
  2-D (64×64): ~6–7 M   (SongUNet, model_channels=64, channel_mult=[1,2,2])
  1-D (len=256): ~4.5 M  (SongUNet1d, model_channels=64, channel_mult=(1,2,2,2))
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

_MEANFLOW_DIR = str(Path(__file__).resolve().parent.parent.parent / "meanflow")
if _MEANFLOW_DIR not in sys.path:
    sys.path.insert(0, _MEANFLOW_DIR)

from models.unet import SongUNet
from models.unet1d import SongUNet1d
from models.ema import init_ema, update_ema_net
from models.time_sampler import sample_two_timesteps


# Lightweight 2-D config — each net receives only its own single channel
NAIVE_FLOW_2D_CONFIG = {
    "img_resolution": 64,
    "in_channels": 1,
    "out_channels": 1,
    "model_channels": 64,
    "channel_mult_noise": 1,
    "resample_filter": [1, 1],
    "channel_mult": [1, 2, 2],
    "num_blocks": 2,
    "encoder_type": "standard",
    "decoder_type": "standard",
    "use_checkpoint": False,
}

# Lightweight 1-D config
NAIVE_FLOW_1D_CONFIG = {
    "seq_len": 256,
    "in_channels": 1,
    "out_channels": 1,
    "model_channels": 64,
    "channel_mult": (1, 2, 2, 2),
    "num_blocks": 2,
    "attn_resolutions": (32,),
    "channel_mult_noise": 2,
    "embedding_type": "positional",
}


class NaiveMeanFlow(nn.Module):
    """
    N independent Mean-Flow networks — one per process.

    Each network_i receives only z_i as input. No cross-process interaction.
    Training loss = MeanFlow local loss (JVP-based) + global loss,
    applied independently to every process.
    """

    def __init__(self, arch, net_configs_list, args):
        super().__init__()
        self.args = args
        self.n_proc = len(net_configs_list)

        # Trainable networks: net1, net2, …, netN
        for i, cfg in enumerate(net_configs_list, start=1):
            self.add_module(f"net{i}", arch(**cfg))

        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        # Primary EMA: net1_ema, …, netN_ema
        for i, cfg in enumerate(net_configs_list, start=1):
            net = self._modules[f"net{i}"]
            self.add_module(
                f"net{i}_ema",
                init_ema(net, arch(**copy.deepcopy(cfg)), args.ema_decay),
            )

        # Additional EMA decays: net1_ema_extra1, …
        self.ema_decays = list(getattr(args, "ema_decays", []))
        for j, decay in enumerate(self.ema_decays, start=1):
            for i, cfg in enumerate(net_configs_list, start=1):
                net = self._modules[f"net{i}"]
                self.add_module(
                    f"net{i}_ema_extra{j}",
                    init_ema(net, arch(**copy.deepcopy(cfg)), decay),
                )

    # ------------------------------------------------------------------ #
    # EMA
    # ------------------------------------------------------------------ #

    def update_ema(self):
        self.num_updates += 1
        n = self.num_updates
        for i in range(1, self.n_proc + 1):
            update_ema_net(self._modules[f"net{i}"], self._modules[f"net{i}_ema"], n)
        for j in range(1, len(self.ema_decays) + 1):
            for i in range(1, self.n_proc + 1):
                update_ema_net(
                    self._modules[f"net{i}"],
                    self._modules[f"net{i}_ema_extra{j}"],
                    n,
                )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _expand_time_like(self, t, x):
        return t.view(-1, *([1] * (x.ndim - 1)))

    def _u_i(self, z_i, t, r, net):
        """Single forward pass for process i — only z_i as spatial input."""
        h = t - r
        return net(z_i, (t.view(-1), h.view(-1)), aug_cond=None)

    def _adaptive_reduce(self, sq):
        if getattr(self.args, "use_adaptive_weight", True):
            wt = (sq.detach() + self.args.norm_eps) ** self.args.norm_p
            sq = sq / wt
        return sq.mean()

    # ------------------------------------------------------------------ #
    # Loss functions
    # ------------------------------------------------------------------ #

    def _local_loss_one_process(self, i, z_t_i, v_t_i, t_b, r_b):
        """MeanFlow local loss for process i via JVP (no other processes used)."""
        net = self._modules[f"net{i}"]
        dtdt = torch.ones_like(t_b)
        drdt = torch.zeros_like(r_b)

        def u_func(z, t, r):
            return self._u_i(z, t, r, net=net)

        u_pred = u_func(z_t_i, t_b, r_b)

        old_ckpt = getattr(net, "use_checkpoint", False)
        net.use_checkpoint = False
        with torch.no_grad():
            _, dudt = torch.func.jvp(
                u_func,
                (z_t_i, t_b, r_b),
                (v_t_i, dtdt, drdt),
            )
        net.use_checkpoint = old_ckpt

        u_tgt = v_t_i - (t_b - r_b) * dudt
        sq = ((u_pred - u_tgt.detach()) ** 2).flatten(1).sum(dim=1)
        return self._adaptive_reduce(sq)

    def forward_local_loss(self, sources, targets):
        """
        Local MeanFlow loss — linear interpolation per process independently.

        sources / targets: lists of N tensors, each [B, 1, *spatial].
        """
        bsz = sources[0].shape[0]
        device = sources[0].device
        dtype = sources[0].dtype

        t, r = sample_two_timesteps(self.args, num_samples=bsz, device=device)
        t = t.to(dtype=dtype)
        r = r.to(dtype=dtype)
        t_b = self._expand_time_like(t, sources[0])
        r_b = self._expand_time_like(r, sources[0])

        total = sources[0].new_zeros(())
        for i in range(1, self.n_proc + 1):
            src, tgt = sources[i - 1], targets[i - 1]
            z_t_i = (1.0 - t_b) * tgt + t_b * src   # linear interpolant
            v_t_i = src - tgt                          # constant velocity
            total = total + self._local_loss_one_process(i, z_t_i, v_t_i, t_b, r_b)
        return total

    def forward_loss(self, sources, targets):
        """Training loss: local MeanFlow only (no global loss)."""
        return self.forward_local_loss(sources, targets)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(self, sources, nets=None):
        """
        One-step MeanFlow inference.

        Args:
            sources: list of N tensors [B, 1, *spatial]
            nets: list of N networks to use; defaults to EMA networks.

        Returns:
            List of N predicted target tensors.
        """
        bsz = sources[0].shape[0]
        device = sources[0].device
        dtype = sources[0].dtype

        t1 = torch.ones(bsz, device=device, dtype=dtype)
        r0 = torch.zeros(bsz, device=device, dtype=dtype)
        t1_b = self._expand_time_like(t1, sources[0])
        r0_b = self._expand_time_like(r0, sources[0])

        preds = []
        for i in range(1, self.n_proc + 1):
            net = (
                self._modules[f"net{i}_ema"] if nets is None else nets[i - 1]
            )
            u = self._u_i(sources[i - 1], t1_b, r0_b, net=net)
            preds.append(sources[i - 1] - u)
        return preds


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #

def build_naive_flow_model(n_proc: int, dim: str, args) -> NaiveMeanFlow:
    """
    Build a NaiveMeanFlow model.

    Args:
        n_proc: number of processes (1 network per process)
        dim:    '2d' for 2-D spatial fields, '1d' for 1-D sequences
        args:   parsed args namespace (must contain dropout, ema_decay, ema_decays,
                norm_p, norm_eps, and all MeanFlow time-sampler fields)

    Returns:
        NaiveMeanFlow instance
    """
    if dim == "2d":
        arch = SongUNet
        base_cfg = copy.deepcopy(NAIVE_FLOW_2D_CONFIG)
    elif dim == "1d":
        arch = SongUNet1d
        base_cfg = copy.deepcopy(NAIVE_FLOW_1D_CONFIG)
    else:
        raise ValueError(f"Unknown dim: {dim!r}. Expected '2d' or '1d'.")

    base_cfg["dropout"] = args.dropout
    net_configs_list = [copy.deepcopy(base_cfg) for _ in range(n_proc)]
    return NaiveMeanFlow(arch=arch, net_configs_list=net_configs_list, args=args)
