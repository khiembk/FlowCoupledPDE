"""
CoupledFlowNProc — N-process coupled flow for arbitrary number of processes.

Designed as a lightweight variant for larger systems (e.g., THM with 5 processes).
Does NOT modify CoupledFlow (2-process) or CoupledFlowBZ (3-process).

Each of the N networks receives all N processes concatenated as input and predicts
the velocity field for its own process. Uses the same MeanFlow local+global
objective and DICE+GP coupling as the 2-proc and 3-proc variants.
"""
import copy
import torch
import torch.nn as nn

from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net
from models.gp_velocity import VectorValuedGP


class CoupledFlowNProc(nn.Module):
    """
    N-process coupled flow (N >= 2).

    Args:
        arch: network constructor (e.g., SongUNet)
        args: training arguments namespace
        net_configs_list: list of N config dicts (one per process, each with
                          in_channels=N set by the caller)
    """

    def __init__(self, arch, args, net_configs_list):
        super().__init__()
        self.args = args
        self.n_proc = len(net_configs_list)
        assert self.n_proc >= 2, "Need at least 2 processes"

        self.nets = nn.ModuleList([arch(**cfg) for cfg in net_configs_list])
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        # Primary EMA (one per process)
        self.nets_ema = nn.ModuleList([
            init_ema(net, arch(**copy.deepcopy(cfg)), args.ema_decay)
            for net, cfg in zip(self.nets, net_configs_list)
        ])

        # Extra EMA decays
        self.ema_decays = list(getattr(args, "ema_decays", []))
        for ei, decay in enumerate(self.ema_decays):
            for pi, (net, cfg) in enumerate(zip(self.nets, net_configs_list)):
                self.add_module(
                    f"net{pi+1}_ema{ei+1}",
                    init_ema(net, arch(**copy.deepcopy(cfg)), decay)
                )

        # GP modules for DICE coupling (one per possible driver process)
        if getattr(args, "use_gp", False):
            log_ls = getattr(args, "gp_log_length_scale", 0.0)
            for i in range(self.n_proc):
                self.add_module(f"gp_{i+1}", VectorValuedGP(log_length_scale=log_ls))

    # -----------------------------------------------------------------------
    # EMA
    # -----------------------------------------------------------------------

    def update_ema(self):
        self.num_updates += 1
        n = self.num_updates
        for i, net in enumerate(self.nets):
            update_ema_net(net, self.nets_ema[i], n)
        for ei in range(len(self.ema_decays)):
            for pi, net in enumerate(self.nets):
                update_ema_net(net, self._modules[f"net{pi+1}_ema{ei+1}"], n)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _expand_time_like(self, t, x):
        return t.view(-1, *([1] * (x.ndim - 1)))

    def _adaptive_reduce(self, sq):
        if getattr(self.args, "use_adaptive_weight", True):
            wt = (sq.detach() + self.args.norm_eps) ** self.args.norm_p
            sq = sq / wt
        return sq.mean()

    def _u_i(self, i, zs, t, r, net=None):
        """Predict velocity for process i given all N processes concatenated."""
        net = self.nets[i] if net is None else net
        h = t - r
        inp = torch.cat(zs, dim=1)
        return net(inp, (t.view(-1), h.view(-1)), aug_cond=None)

    # -----------------------------------------------------------------------
    # Interpolation / GP coupling
    # -----------------------------------------------------------------------

    def gp_build_zt_vt(self, sources, targets, t_b):
        """
        DICE coupling for N processes.
        Uniformly picks one driver; all other N-1 processes are GP-driven.
        The masks are mutually exclusive so successive torch.where calls
        correctly select per-sample trajectories.
        """
        B = sources[0].shape[0]
        device = sources[0].device
        ndim = sources[0].ndim

        # Start from linear interpolants
        zts = [(1.0 - t_b) * tgt + t_b * src for src, tgt in zip(sources, targets)]
        vts = [src - tgt for src, tgt in zip(sources, targets)]

        dice = torch.randint(0, self.n_proc, (B,), device=device)
        for i in range(self.n_proc):
            mask_i = (dice == i).view(B, *([1] * (ndim - 1)))
            gp_i = self._modules[f"gp_{i+1}"]
            for j in range(self.n_proc):
                if j != i:
                    z_gp, v_gp = gp_i(t_b, targets[i], sources[i], targets[j], sources[j])
                    zts[j] = torch.where(mask_i, z_gp, zts[j])
                    vts[j] = torch.where(mask_i, v_gp, vts[j])

        return zts, vts

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------

    def _local_loss_one_process(self, i, zs, vs, t_b, r_b):
        dtdt = torch.ones_like(t_b)
        drdt = torch.zeros_like(r_b)
        net = self.nets[i]

        def u_func(*args):
            zs_in = list(args[:self.n_proc])
            t, r = args[self.n_proc], args[self.n_proc + 1]
            return self._u_i(i, zs_in, t, r, net=net)

        u_pred = u_func(*zs, t_b, r_b)

        old_ckpt = getattr(net, 'use_checkpoint', False)
        net.use_checkpoint = False
        with torch.no_grad():
            _, dudt = torch.func.jvp(
                u_func,
                (*zs, t_b, r_b),
                (*vs, dtdt, drdt),
            )
        net.use_checkpoint = old_ckpt

        u_tgt = vs[i] - (t_b - r_b) * dudt
        sq = ((u_pred - u_tgt.detach()) ** 2).flatten(1).sum(dim=1)
        return self._adaptive_reduce(sq)

    def forward_global_loss(self, sources, targets):
        device = sources[0].device
        dtype = sources[0].dtype
        bsz = sources[0].shape[0]

        t1 = torch.ones(bsz, device=device, dtype=dtype)
        r0 = torch.zeros(bsz, device=device, dtype=dtype)
        t1_b = self._expand_time_like(t1, sources[0])
        r0_b = self._expand_time_like(r0, sources[0])

        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for i in range(self.n_proc):
            u_i = self._u_i(i, sources, t1_b, r0_b)
            v_global = sources[i] - targets[i]
            sq = ((u_i - v_global) ** 2).flatten(1).sum(dim=1)
            loss = loss + self._adaptive_reduce(sq)
        return loss

    def forward_local_loss(self, sources, targets):
        device = sources[0].device
        dtype = sources[0].dtype

        t, r = sample_two_timesteps(self.args, num_samples=sources[0].shape[0], device=device)
        t = t.to(dtype=dtype)
        r = r.to(dtype=dtype)
        t_b = self._expand_time_like(t, sources[0])
        r_b = self._expand_time_like(r, sources[0])

        if getattr(self.args, "use_gp", False):
            zts, vts = self.gp_build_zt_vt(sources, targets, t_b)
        else:
            zts = [(1.0 - t_b) * tgt + t_b * src for src, tgt in zip(sources, targets)]
            vts = [src - tgt for src, tgt in zip(sources, targets)]

        return sum(
            self._local_loss_one_process(i, zts, vts, t_b, r_b)
            for i in range(self.n_proc)
        )

    def forward_combined_loss(self, sources, targets):
        return self.forward_local_loss(sources, targets) + self.forward_global_loss(sources, targets)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, sources, nets=None):
        if nets is None:
            nets = list(self.nets_ema)
        device = sources[0].device
        dtype = sources[0].dtype
        bsz = sources[0].shape[0]

        t1 = torch.ones(bsz, device=device, dtype=dtype)
        r0 = torch.zeros(bsz, device=device, dtype=dtype)
        t1_b = self._expand_time_like(t1, sources[0])
        r0_b = self._expand_time_like(r0, sources[0])

        return [
            sources[i] - self._u_i(i, sources, t1_b, r0_b, net=nets[i])
            for i in range(self.n_proc)
        ]
