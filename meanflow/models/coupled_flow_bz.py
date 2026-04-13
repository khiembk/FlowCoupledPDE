"""
CoupledFlowBZ — 3-process coupled flow for Belousov-Zhabotinsky (BZ) data.

Completely separate from CoupledFlow (2-process) to avoid any impact on
GS/LV training. Each network receives all 3 processes concatenated as input
and outputs the velocity field for one process.
"""
import copy
import torch
import torch.nn as nn

from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net
from models.gp_velocity import VectorValuedGP


class CoupledFlowBZ(nn.Module):

    def __init__(self, arch, args, net1_configs, net2_configs, net3_configs):
        super().__init__()
        self.args = args

        self.net1 = arch(**net1_configs)
        self.net2 = arch(**net2_configs)
        self.net3 = arch(**net3_configs)

        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        self.net1_ema = init_ema(self.net1, arch(**copy.deepcopy(net1_configs)), args.ema_decay)
        self.net2_ema = init_ema(self.net2, arch(**copy.deepcopy(net2_configs)), args.ema_decay)
        self.net3_ema = init_ema(self.net3, arch(**copy.deepcopy(net3_configs)), args.ema_decay)

        self.ema_decays = list(getattr(args, "ema_decays", []))
        for i, decay in enumerate(self.ema_decays):
            self.add_module(f"net1_ema{i+1}", init_ema(self.net1, arch(**copy.deepcopy(net1_configs)), decay))
            self.add_module(f"net2_ema{i+1}", init_ema(self.net2, arch(**copy.deepcopy(net2_configs)), decay))
            self.add_module(f"net3_ema{i+1}", init_ema(self.net3, arch(**copy.deepcopy(net3_configs)), decay))

        # GP modules for DICE coupling (one per driver process)
        if getattr(args, "use_gp", False):
            log_ls = getattr(args, "gp_log_length_scale", 0.0)
            self.gp_1 = VectorValuedGP(log_length_scale=log_ls)  # process 1 drives 2 & 3
            self.gp_2 = VectorValuedGP(log_length_scale=log_ls)  # process 2 drives 1 & 3
            self.gp_3 = VectorValuedGP(log_length_scale=log_ls)  # process 3 drives 1 & 2

    def update_ema(self):
        self.num_updates += 1
        n = self.num_updates
        update_ema_net(self.net1, self.net1_ema, n)
        update_ema_net(self.net2, self.net2_ema, n)
        update_ema_net(self.net3, self.net3_ema, n)
        for i in range(len(self.ema_decays)):
            update_ema_net(self.net1, self._modules[f"net1_ema{i+1}"], n)
            update_ema_net(self.net2, self._modules[f"net2_ema{i+1}"], n)
            update_ema_net(self.net3, self._modules[f"net3_ema{i+1}"], n)

    def _expand_time_like(self, t, x):
        return t.view(-1, *([1] * (x.ndim - 1)))

    # ------------------------------------------------------------------
    # Velocity predictors — each net sees all 3 processes concatenated
    # ------------------------------------------------------------------

    def _u1(self, z1, z2, z3, t, r, net=None):
        net = self.net1 if net is None else net
        h = t - r
        inp = torch.cat([z1, z2, z3], dim=1)
        return net(inp, (t.view(-1), h.view(-1)), aug_cond=None)

    def _u2(self, z1, z2, z3, t, r, net=None):
        net = self.net2 if net is None else net
        h = t - r
        inp = torch.cat([z1, z2, z3], dim=1)
        return net(inp, (t.view(-1), h.view(-1)), aug_cond=None)

    def _u3(self, z1, z2, z3, t, r, net=None):
        net = self.net3 if net is None else net
        h = t - r
        inp = torch.cat([z1, z2, z3], dim=1)
        return net(inp, (t.view(-1), h.view(-1)), aug_cond=None)

    def gp_build_zt_vt(self, source_1, source_2, source_3,
                        target_1, target_2, target_3, t_b):
        """
        DICE strategy for 3 processes (prob 1/3 each):
          - dice=0: process 1 linear driver, processes 2 & 3 GP-driven by gp_1
          - dice=1: process 2 linear driver, processes 1 & 3 GP-driven by gp_2
          - dice=2: process 3 linear driver, processes 1 & 2 GP-driven by gp_3
        """
        B = source_1.shape[0]
        device = source_1.device

        # Linear interpolants and velocities for all 3 processes
        zt_1_lin = (1.0 - t_b) * target_1 + t_b * source_1
        zt_2_lin = (1.0 - t_b) * target_2 + t_b * source_2
        zt_3_lin = (1.0 - t_b) * target_3 + t_b * source_3
        v_1_lin  = source_1 - target_1
        v_2_lin  = source_2 - target_2
        v_3_lin  = source_3 - target_3

        # GP: driver=1 → driven 2 and 3
        zt_2_gp1, vt_2_gp1 = self.gp_1(t_b, target_1, source_1, target_2, source_2)
        zt_3_gp1, vt_3_gp1 = self.gp_1(t_b, target_1, source_1, target_3, source_3)
        # GP: driver=2 → driven 1 and 3
        zt_1_gp2, vt_1_gp2 = self.gp_2(t_b, target_2, source_2, target_1, source_1)
        zt_3_gp2, vt_3_gp2 = self.gp_2(t_b, target_2, source_2, target_3, source_3)
        # GP: driver=3 → driven 1 and 2
        zt_1_gp3, vt_1_gp3 = self.gp_3(t_b, target_3, source_3, target_1, source_1)
        zt_2_gp3, vt_2_gp3 = self.gp_3(t_b, target_3, source_3, target_2, source_2)

        # Categorical DICE: 0, 1, 2 with equal probability 1/3
        dice = torch.randint(0, 3, (B,), device=device)
        shape = (B,) + (1,) * (source_1.ndim - 1)
        m0 = (dice == 0).view(shape)  # proc 1 is driver
        m1 = (dice == 1).view(shape)  # proc 2 is driver
        m2 = (dice == 2).view(shape)  # proc 3 is driver

        zt_1 = torch.where(m0, zt_1_lin,  torch.where(m1, zt_1_gp2, zt_1_gp3))
        zt_2 = torch.where(m0, zt_2_gp1,  torch.where(m1, zt_2_lin,  zt_2_gp3))
        zt_3 = torch.where(m0, zt_3_gp1,  torch.where(m1, zt_3_gp2,  zt_3_lin))
        vt_1 = torch.where(m0, v_1_lin,   torch.where(m1, vt_1_gp2,  vt_1_gp3))
        vt_2 = torch.where(m0, vt_2_gp1,  torch.where(m1, v_2_lin,   vt_2_gp3))
        vt_3 = torch.where(m0, vt_3_gp1,  torch.where(m1, vt_3_gp2,  v_3_lin))

        return zt_1, zt_2, zt_3, vt_1, vt_2, vt_3

    def _adaptive_reduce(self, sq):
        if getattr(self.args, "use_adaptive_weight", True):
            wt = (sq.detach() + self.args.norm_eps) ** self.args.norm_p
            sq = sq / wt
        return sq.mean()

    # ------------------------------------------------------------------
    # Local loss with JVP for one process
    # ------------------------------------------------------------------

    def _local_loss_one_process(self, which, zt_1, zt_2, zt_3, v1, v2, v3, t_b, r_b):
        dtdt = torch.ones_like(t_b)
        drdt = torch.zeros_like(r_b)

        net  = {1: self.net1, 2: self.net2, 3: self.net3}[which]
        u_fn = {1: self._u1,  2: self._u2,  3: self._u3}[which]

        def u_func(z1, z2, z3, t, r):
            return u_fn(z1, z2, z3, t, r, net=net)

        u_pred = u_func(zt_1, zt_2, zt_3, t_b, r_b)

        old_ckpt = getattr(net, 'use_checkpoint', False)
        net.use_checkpoint = False
        with torch.no_grad():
            _, dudt = torch.func.jvp(
                u_func,
                (zt_1, zt_2, zt_3, t_b,  r_b),
                (v1,   v2,   v3,   dtdt, drdt),
            )
        net.use_checkpoint = old_ckpt

        v_target = {1: v1, 2: v2, 3: v3}[which]
        u_tgt = v_target - (t_b - r_b) * dudt
        sq = ((u_pred - u_tgt.detach()) ** 2).flatten(1).sum(dim=1)
        return self._adaptive_reduce(sq)

    # ------------------------------------------------------------------
    # Global loss (t=1, r=0 endpoints)
    # ------------------------------------------------------------------

    def forward_global_loss(self, source_1, source_2, source_3, target_1, target_2, target_3):
        device = source_1.device
        dtype = source_1.dtype
        bsz = source_1.shape[0]

        t1 = torch.ones(bsz, device=device, dtype=dtype)
        r0 = torch.zeros(bsz, device=device, dtype=dtype)
        t1_b = self._expand_time_like(t1, source_1)
        r0_b = self._expand_time_like(r0, source_1)

        u1 = self._u1(source_1, source_2, source_3, t1_b, r0_b)
        u2 = self._u2(source_1, source_2, source_3, t1_b, r0_b)
        u3 = self._u3(source_1, source_2, source_3, t1_b, r0_b)

        v1_global = source_1 - target_1
        v2_global = source_2 - target_2
        v3_global = source_3 - target_3

        loss = (
            self._adaptive_reduce(((u1 - v1_global) ** 2).flatten(1).sum(dim=1))
            + self._adaptive_reduce(((u2 - v2_global) ** 2).flatten(1).sum(dim=1))
            + self._adaptive_reduce(((u3 - v3_global) ** 2).flatten(1).sum(dim=1))
        )
        return loss

    # ------------------------------------------------------------------
    # Local loss (intermediate time t with JVP)
    # ------------------------------------------------------------------

    def forward_local_loss(self, source_1, source_2, source_3, target_1, target_2, target_3):
        device = source_1.device
        dtype = source_1.dtype

        t, r = sample_two_timesteps(self.args, num_samples=source_1.shape[0], device=device)
        t = t.to(dtype=dtype)
        r = r.to(dtype=dtype)
        t_b = self._expand_time_like(t, source_1)
        r_b = self._expand_time_like(r, source_1)

        if getattr(self.args, "use_gp", False):
            zt_1, zt_2, zt_3, v_t_1, v_t_2, v_t_3 = self.gp_build_zt_vt(
                source_1, source_2, source_3, target_1, target_2, target_3, t_b
            )
        else:
            zt_1 = (1.0 - t_b) * target_1 + t_b * source_1
            zt_2 = (1.0 - t_b) * target_2 + t_b * source_2
            zt_3 = (1.0 - t_b) * target_3 + t_b * source_3
            v_t_1 = source_1 - target_1
            v_t_2 = source_2 - target_2
            v_t_3 = source_3 - target_3

        local1 = self._local_loss_one_process(1, zt_1, zt_2, zt_3, v_t_1, v_t_2, v_t_3, t_b, r_b)
        local2 = self._local_loss_one_process(2, zt_1, zt_2, zt_3, v_t_1, v_t_2, v_t_3, t_b, r_b)
        local3 = self._local_loss_one_process(3, zt_1, zt_2, zt_3, v_t_1, v_t_2, v_t_3, t_b, r_b)
        return local1 + local2 + local3

    def forward_combined_loss(self, source_1, source_2, source_3, target_1, target_2, target_3):
        local  = self.forward_local_loss(source_1, source_2, source_3, target_1, target_2, target_3)
        global_ = self.forward_global_loss(source_1, source_2, source_3, target_1, target_2, target_3)
        return local + global_

    @torch.no_grad()
    def sample(self, source_1, source_2, source_3, net1=None, net2=None, net3=None):
        net1 = self.net1_ema if net1 is None else net1
        net2 = self.net2_ema if net2 is None else net2
        net3 = self.net3_ema if net3 is None else net3

        bsz = source_1.shape[0]
        device = source_1.device
        dtype = source_1.dtype

        t1 = torch.ones(bsz, device=device, dtype=dtype)
        r0 = torch.zeros(bsz, device=device, dtype=dtype)
        t1_b = self._expand_time_like(t1, source_1)
        r0_b = self._expand_time_like(r0, source_1)

        u1 = self._u1(source_1, source_2, source_3, t1_b, r0_b, net=net1)
        u2 = self._u2(source_1, source_2, source_3, t1_b, r0_b, net=net2)
        u3 = self._u3(source_1, source_2, source_3, t1_b, r0_b, net=net3)

        return source_1 - u1, source_2 - u2, source_3 - u3
