import copy
import torch
import torch.nn as nn

from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net


# ─────────────────────────────────────────────────────────────────────────────
# Bezier encoding networks (time-independent φ_i, so ∂_t φ_i = 0)
# ─────────────────────────────────────────────────────────────────────────────

class BezierEncodingNet2d(nn.Module):
    """
    Time-independent Bezier control-point encoder for 2-D coupled processes.

    Inputs  : z0_list — list of n tensors [B, C, H, W] at t=0  (= target)
              z1_list — list of n tensors [B, C, H, W] at t=1  (= source)
    Outputs : list of n phi tensors [B, C, H, W]

    Because φ_i does not depend on t:
        ∂_t φ_i = 0
        v^t_i   = (z_i^1 - z_i^0) + (2t - 1) φ_i
    """

    def __init__(self, n_proc: int = 2, channels_per_proc: int = 1, hidden: int = 64):
        super().__init__()
        in_ch = 2 * n_proc * channels_per_proc  # z0 and z1 for every process
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.heads = nn.ModuleList([
            nn.Conv2d(hidden, channels_per_proc, 1) for _ in range(n_proc)
        ])

    def forward(self, z0_list, z1_list):
        x = torch.cat(z0_list + z1_list, dim=1)  # [B, 2*n*C, H, W]
        h = self.shared(x)
        return [head(h) for head in self.heads]   # list of n tensors [B, C, H, W]


class BezierEncodingNet1d(nn.Module):
    """
    Time-independent Bezier control-point encoder for 1-D coupled processes.

    Same interface as BezierEncodingNet2d but operates on [B, C, L] tensors.
    """

    def __init__(self, n_proc: int = 2, channels_per_proc: int = 1, hidden: int = 64):
        super().__init__()
        in_ch = 2 * n_proc * channels_per_proc
        self.shared = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.heads = nn.ModuleList([
            nn.Conv1d(hidden, channels_per_proc, 1) for _ in range(n_proc)
        ])

    def forward(self, z0_list, z1_list):
        x = torch.cat(z0_list + z1_list, dim=1)  # [B, 2*n*C, L]
        h = self.shared(x)
        return [head(h) for head in self.heads]   # list of n tensors [B, C, L]


# ─────────────────────────────────────────────────────────────────────────────
# CoupledFlow with Bezier curve trajectory
# ─────────────────────────────────────────────────────────────────────────────

class CoupledFlowBezier(nn.Module):
    """
    Two-process CoupledFlow using a quadratic Bezier trajectory.

    The intermediate representation is:
        z^t_i = (1-t) z_i^0 + t z_i^1 + t(t-1) φ_i(Z(0), Z(1))

    φ_i is time-independent (Option 1), so ∂_t φ_i = 0 and the instant
    velocity simplifies to:
        v^t_i = (z_i^1 - z_i^0) + (2t - 1) φ_i

    Convention (matches the rest of the codebase):
        source = z(t=1),  target = z(t=0)
    """

    def __init__(self, arch, args, net1_configs, net2_configs, encoding_net: nn.Module):
        """
        Args:
            arch          : UNet class (e.g. SongUNet / SongUNet1d)
            args          : training args namespace
            net1_configs  : dict of kwargs for net1
            net2_configs  : dict of kwargs for net2
            encoding_net  : already-instantiated BezierEncodingNet2d/1d
        """
        super().__init__()
        self.args = args

        self.net1 = arch(**net1_configs)
        self.net2 = arch(**net2_configs)
        self.encoding_net = encoding_net

        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        self.net1_ema = init_ema(self.net1, arch(**copy.deepcopy(net1_configs)), args.ema_decay)
        self.net2_ema = init_ema(self.net2, arch(**copy.deepcopy(net2_configs)), args.ema_decay)
        self.encoding_net_ema = init_ema(
            self.encoding_net, copy.deepcopy(encoding_net), args.ema_decay
        )

        self.ema_decays = list(getattr(args, "ema_decays", []))
        for i, decay in enumerate(self.ema_decays):
            self.add_module(
                f"net1_ema{i+1}",
                init_ema(self.net1, arch(**copy.deepcopy(net1_configs)), decay),
            )
            self.add_module(
                f"net2_ema{i+1}",
                init_ema(self.net2, arch(**copy.deepcopy(net2_configs)), decay),
            )
            self.add_module(
                f"encoding_net_ema{i+1}",
                init_ema(self.encoding_net, copy.deepcopy(encoding_net), decay),
            )

    # ─── EMA ──────────────────────────────────────────────────────────────────

    def update_ema(self):
        self.num_updates += 1
        n = self.num_updates
        update_ema_net(self.net1, self.net1_ema, n)
        update_ema_net(self.net2, self.net2_ema, n)
        update_ema_net(self.encoding_net, self.encoding_net_ema, n)
        for i in range(len(self.ema_decays)):
            update_ema_net(self.net1, self._modules[f"net1_ema{i+1}"], n)
            update_ema_net(self.net2, self._modules[f"net2_ema{i+1}"], n)
            update_ema_net(self.encoding_net, self._modules[f"encoding_net_ema{i+1}"], n)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _expand_time_like(self, t, x):
        """[B] -> [B, 1, 1, ..., 1]"""
        return t.view(-1, *([1] * (x.ndim - 1)))

    # ─── Bezier trajectory ────────────────────────────────────────────────────

    def bezier_build_z_t_vt(self, source_1, source_2, target_1, target_2, t):
        """
        Build Bezier trajectory point z(t) and instant velocity v(t).

        z^t_i = (1-t) z_i^0 + t z_i^1 + t(t-1) φ_i          (position)
        v^t_i = (z_i^1 - z_i^0) + (2t-1) φ_i                 (velocity, ∂_t φ_i=0)

        Args:
            source_{1,2}: z(t=1),  target_{1,2}: z(t=0)
            t: [B, 1, ..., 1]

        Returns:
            zt_1, zt_2 : trajectory points at time t
            vt_1, vt_2 : instant velocities at time t
        """
        # φ_i from the encoding net (time-independent)
        # z(t=0) = target,  z(t=1) = source
        phis = self.encoding_net([target_1, target_2], [source_1, source_2])
        phi_1, phi_2 = phis[0], phis[1]

        # Bezier interpolation
        zt_1 = (1.0 - t) * target_1 + t * source_1 + t * (t - 1.0) * phi_1
        zt_2 = (1.0 - t) * target_2 + t * source_2 + t * (t - 1.0) * phi_2

        # Instant velocity (∂_t φ_i = 0, so the last term vanishes)
        vt_1 = (source_1 - target_1) + (2.0 * t - 1.0) * phi_1
        vt_2 = (source_2 - target_2) + (2.0 * t - 1.0) * phi_2

        return zt_1, zt_2, vt_1, vt_2

    # ─── Flow network calls ───────────────────────────────────────────────────

    def _u1(self, z1, z2, t, r, net=None):
        net = self.net1 if net is None else net
        h = t - r
        inp = torch.cat([z1, z2], dim=1)
        return net(inp, (t.view(-1), h.view(-1)), aug_cond=None)

    def _u2(self, z1, z2, t, r, net=None):
        net = self.net2 if net is None else net
        h = t - r
        inp = torch.cat([z1, z2], dim=1)
        return net(inp, (t.view(-1), h.view(-1)), aug_cond=None)

    # ─── Loss helpers ─────────────────────────────────────────────────────────

    def _adaptive_reduce(self, sq):
        if getattr(self.args, "use_adaptive_weight", True):
            wt = (sq.detach() + self.args.norm_eps) ** self.args.norm_p
            sq = sq / wt
        return sq.mean()

    def _local_loss_one_process(self, which, zt_1, zt_2, v1, v2, t_b, r_b):
        """MeanFlow local loss for one process via JVP."""
        dtdt = torch.ones_like(t_b)
        drdt = torch.zeros_like(r_b)

        net  = self.net1 if which == 1 else self.net2
        u_fn = self._u1  if which == 1 else self._u2

        def u_func(z1, z2, t, r):
            return u_fn(z1, z2, t, r, net=net)

        u_pred = u_func(zt_1, zt_2, t_b, r_b)

        # torch.utils.checkpoint does not implement a JVP rule; disable
        # temporarily (no activations need saving inside no_grad anyway).
        old_ckpt = getattr(net, "use_checkpoint", False)
        net.use_checkpoint = False
        with torch.no_grad():
            _, dudt = torch.func.jvp(
                u_func,
                (zt_1, zt_2, t_b, r_b),
                (v1,   v2,   dtdt, drdt),
            )
        net.use_checkpoint = old_ckpt

        v_target = v1 if which == 1 else v2
        u_tgt = v_target - (t_b - r_b) * dudt
        sq = ((u_pred - u_tgt.detach()) ** 2).flatten(1).sum(dim=1)
        return self._adaptive_reduce(sq)

    # ─── Forward losses ───────────────────────────────────────────────────────

    def forward_global_loss(self, source_1, source_2, target_1, target_2):
        """Endpoint velocity matching loss (t=1, r=0). No JVP, no φ."""
        device = source_1.device
        dtype  = source_1.dtype
        bsz    = source_1.shape[0]

        t1   = torch.ones(bsz,  device=device, dtype=dtype)
        r0   = torch.zeros(bsz, device=device, dtype=dtype)
        t1_b = self._expand_time_like(t1, source_1)
        r0_b = self._expand_time_like(r0, source_1)

        v1_global = source_1 - target_1
        v2_global = source_2 - target_2

        u1 = self._u1(source_1, source_2, t1_b, r0_b, net=self.net1)
        u2 = self._u2(source_1, source_2, t1_b, r0_b, net=self.net2)

        sq1 = ((u1 - v1_global) ** 2).flatten(1).sum(dim=1)
        sq2 = ((u2 - v2_global) ** 2).flatten(1).sum(dim=1)
        return self._adaptive_reduce(sq1) + self._adaptive_reduce(sq2)

    def forward_local_loss(self, source_1, source_2, target_1, target_2):
        """Instantaneous velocity loss along the Bezier trajectory."""
        device = source_1.device
        dtype  = source_1.dtype

        t, r = sample_two_timesteps(self.args, num_samples=source_1.shape[0], device=device)
        t = t.to(dtype=dtype)
        r = r.to(dtype=dtype)

        t_b = self._expand_time_like(t, source_1)
        r_b = self._expand_time_like(r, source_1)

        z_t_1, z_t_2, v_t_1, v_t_2 = self.bezier_build_z_t_vt(
            source_1, source_2, target_1, target_2, t_b
        )

        local1 = self._local_loss_one_process(1, z_t_1, z_t_2, v_t_1, v_t_2, t_b, r_b)
        local2 = self._local_loss_one_process(2, z_t_1, z_t_2, v_t_1, v_t_2, t_b, r_b)
        return local1 + local2

    # ─── Sampling ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, source_1, source_2, net1=None, net2=None):
        """One-step prediction from source (t=1) to target (t=0)."""
        net1 = self.net1_ema if net1 is None else net1
        net2 = self.net2_ema if net2 is None else net2

        device = source_1.device
        dtype  = source_1.dtype
        bsz    = source_1.shape[0]

        t1   = torch.ones(bsz,  device=device, dtype=dtype)
        r0   = torch.zeros(bsz, device=device, dtype=dtype)
        t1_b = self._expand_time_like(t1, source_1)
        r0_b = self._expand_time_like(r0, source_1)

        u1 = self._u1(source_1, source_2, t1_b, r0_b, net=net1)
        u2 = self._u2(source_1, source_2, t1_b, r0_b, net=net2)

        target_1 = source_1 - u1
        target_2 = source_2 - u2
        return target_1, target_2
