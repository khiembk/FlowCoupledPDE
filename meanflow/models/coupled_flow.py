import copy
import torch
import torch.nn as nn

from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net
from models.gp_velocity import VectorValuedGP


class CoupledFlow(nn.Module):
   

    def __init__(self, arch, args, net1_configs, net2_configs):
        super().__init__()
        self.args = args

        self.net1 = arch(**net1_configs)
        self.net2 = arch(**net2_configs)

        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        self.net1_ema = init_ema(self.net1, arch(**copy.deepcopy(net1_configs)), args.ema_decay)
        self.net2_ema = init_ema(self.net2, arch(**copy.deepcopy(net2_configs)), args.ema_decay)

        self.ema_decays = list(getattr(args, "ema_decays", []))
        for i, decay in enumerate(self.ema_decays):
            self.add_module(
                f"net1_ema{i+1}",
                init_ema(self.net1, arch(**copy.deepcopy(net1_configs)), decay)
            )
            self.add_module(
                f"net2_ema{i+1}",
                init_ema(self.net2, arch(**copy.deepcopy(net2_configs)), decay)
            )

        # Vector-valued GP modules for DICE strategy (only created when use_gp=True)
        if getattr(args, "use_gp", False):
            log_ls = getattr(args, "gp_log_length_scale", 0.0)
            self.gp_12 = VectorValuedGP(log_length_scale=log_ls)  # process 1 drives process 2
            self.gp_21 = VectorValuedGP(log_length_scale=log_ls)  # process 2 drives process 1

    def update_ema(self):
        self.num_updates += 1
        num_updates = self.num_updates

        update_ema_net(self.net1, self.net1_ema, num_updates)
        update_ema_net(self.net2, self.net2_ema, num_updates)

        for i in range(len(self.ema_decays)):
            update_ema_net(self.net1, self._modules[f"net1_ema{i+1}"], num_updates)
            update_ema_net(self.net2, self._modules[f"net2_ema{i+1}"], num_updates)

    def _expand_time_like(self, t, x):
        # [B] -> [B, 1, 1, 1, ...]
        return t.view(-1, *([1] * (x.ndim - 1)))

    def naive_build_zt_vt(self, source_1, source_2, target_1, target_2, t):

        # naive implementation

        zt_1 = (1.0 - t) * target_1 + t * source_1
        zt_2 = (1.0 - t) * target_2 + t * source_2
        v_t_1 = source_1 - target_1
        v_t_2 = source_2 - target_2

        return zt_1, zt_2, v_t_1, v_t_2

    def gp_build_zt_vt(self, source_1, source_2, target_1, target_2, t):
        """
        Build (z_t, instant_velocity) using vector-valued GPs with the DICE strategy.

        DICE strategy (per sample in the batch):
          - With prob dice_prob : process 1 is the linear driver, process 2 is GP-driven.
          - With prob 1-dice_prob: process 2 is the linear driver, process 1 is GP-driven.

        The driver process uses the standard linear interpolant:
            z^t_driver = (1-t)*z^0_driver + t*z^1_driver

        The driven process uses the GP posterior mean conditioned on the two endpoints
        of the driver and driven trajectories, and its instant velocity is the time
        derivative of that posterior mean (analytic, see VectorValuedGP).

        Args:
            source_i  = z^1_i  (field at t=1, i.e. initial condition)
            target_i  = z^0_i  (field at t=0, i.e. PDE solution)
            t: [B, 1, ..., 1]  current time

        Returns:
            zt_1, zt_2:  interpolated fields at time t
            v_t_1, v_t_2: instant velocities (dz_i/dt) at time t
        """
        B = source_1.shape[0]
        device = source_1.device

        # DICE coin flip — one random draw per sample in the batch
        dice_prob = getattr(self.args, "dice_prob", 0.5)
        mask = (torch.rand(B, device=device) < dice_prob).view(
            B, *([1] * (source_1.ndim - 1))
        )  # True => case A

        # --- Case A: process 1 linear, process 2 GP-driven ---
        # Linear interpolant and constant velocity for process 1
        zt_1_lin = (1.0 - t) * target_1 + t * source_1
        vt_1_lin = source_1 - target_1

        # GP maps (t, z^t_1) -> z^t_2; training data: endpoints of both processes
        # z_driver_0 = target_1 (at t=0), z_driver_1 = source_1 (at t=1)
        # z_driven_0 = target_2 (at t=0), z_driven_1 = source_2 (at t=1)
        zt_2_gp, vt_2_gp = self.gp_12(t, target_1, source_1, target_2, source_2)

        # --- Case B: process 2 linear, process 1 GP-driven ---
        zt_2_lin = (1.0 - t) * target_2 + t * source_2
        vt_2_lin = source_2 - target_2

        zt_1_gp, vt_1_gp = self.gp_21(t, target_2, source_2, target_1, source_1)

        # Select per-sample based on DICE mask
        zt_1  = torch.where(mask, zt_1_lin, zt_1_gp)
        zt_2  = torch.where(mask, zt_2_gp,  zt_2_lin)
        vt_1  = torch.where(mask, vt_1_lin, vt_1_gp)
        vt_2  = torch.where(mask, vt_2_gp,  vt_2_lin)

        return zt_1, zt_2, vt_1, vt_2



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

    def _adaptive_reduce(self, sq):
        # sq: [B]
        if getattr(self.args, "use_adaptive_weight", True):
            wt = (sq.detach() + self.args.norm_eps) ** self.args.norm_p
            sq = sq / wt
        return sq.mean()

    def _local_loss_one_process(self, which, zt_1, zt_2, v1, v2, t_b, r_b):
        dtdt = torch.ones_like(t_b)
        drdt = torch.zeros_like(r_b)

        if which == 1:
            def u_func(z1, z2, t, r):
                return self._u1(z1, z2, t, r, net=self.net1)
            u_pred = u_func(zt_1, zt_2, t_b, r_b)
        else:
            def u_func(z1, z2, t, r):
                return self._u2(z1, z2, t, r, net=self.net2)
            u_pred = u_func(zt_1, zt_2, t_b, r_b)

       
        with torch.no_grad():
            _, dudt = torch.func.jvp(
                u_func,
                (zt_1, zt_2, t_b, r_b),
                (v1,   v2,   dtdt, drdt),
            )

        v_target = v1 if which == 1 else v2
        u_tgt = v_target - (t_b - r_b) * dudt
        sq = ((u_pred - u_tgt.detach()) ** 2).flatten(1).sum(dim=1)
        return self._adaptive_reduce(sq)

    

    def forward_global_loss(self, source_1, source_2, target_1, target_2):
        device = source_1.device
        dtype = source_1.dtype
        bsz = source_1.shape[0]

        t1 = torch.ones(bsz, device=device, dtype=dtype)
        r0 = torch.zeros(bsz, device=device, dtype=dtype)
        t1_b = self._expand_time_like(t1, source_1)
        r0_b = self._expand_time_like(r0, source_1)

        v1_global =  source_1 - target_1
        v2_global =  source_2 - target_2

        u1 = self._u1(source_1, source_2, t1_b, r0_b, net=self.net1)
        u2 = self._u2(source_1, source_2, t1_b, r0_b, net=self.net2)

        sq1 = ((u1 - v1_global) ** 2).flatten(1).sum(dim=1)
        sq2 = ((u2 - v2_global) ** 2).flatten(1).sum(dim=1)

        return self._adaptive_reduce(sq1) + self._adaptive_reduce(sq2)

    def forward_local_loss(self, source_1, source_2, target_1, target_2):
       
        
        device = source_1.device
        dtype = source_1.dtype

        t, r = sample_two_timesteps(self.args, num_samples= source_1.shape[0], device=device)
        t = t.to(dtype=dtype)
        r = r.to(dtype=dtype)

        t_b = self._expand_time_like(t, source_1)
        r_b = self._expand_time_like(r, source_1)

        if getattr(self.args, "use_gp", False):
            z_t_1, z_t_2, v_t_1, v_t_2 = self.gp_build_zt_vt(source_1, source_2, target_1, target_2, t_b)
        else:
            z_t_1, z_t_2, v_t_1, v_t_2 = self.naive_build_zt_vt(source_1, source_2, target_1, target_2, t_b)

       

        local1 = self._local_loss_one_process(1, z_t_1, z_t_2, v_t_1, v_t_2, t_b, r_b)
        local2 = self._local_loss_one_process(2, z_t_1, z_t_2, v_t_1, v_t_2, t_b, r_b)
        
        local_loss = local1 + local2

        return local_loss
    

    def forward_combined_loss(self, source_1, source_2, target_1, target_2, aug_cond=None, lambda_local = 1, lambda_global = 1):
        # print("source_1", source_1.shape)
        # print("source_2", source_2.shape)
        # print("target_1", target_1.shape)
        # print("target_2", target_2.shape)
        
        local_loss = self.forward_local_loss(source_1, source_2, target_1, target_2) 
        global_loss = self.forward_global_loss(source_1, source_2, target_1, target_2)
        
        return lambda_global*global_loss + lambda_local*local_loss

    @torch.no_grad()
    def sample(self, source_1, source_2, net1=None, net2=None):
        
        net1 = self.net1_ema if net1 is None else net1
        net2 = self.net2_ema if net2 is None else net2

        device = source_1.device
        dtype = source_1.dtype
        bsz = source_1.shape[0]

        t1 = torch.ones(bsz, device=device, dtype=dtype)
        r0 = torch.zeros(bsz, device=device, dtype=dtype)
        t1_b = self._expand_time_like(t1, source_1)
        r0_b = self._expand_time_like(r0, source_1)

        u1 = self._u1(source_1, source_2, t1_b, r0_b, net=net1)
        u2 = self._u2(source_1, source_2, t1_b, r0_b, net=net2)

        target_1 = source_1 - u1
        target_2 = source_2 - u2
        return target_1, target_2