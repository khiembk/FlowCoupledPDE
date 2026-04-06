"""
Belousov-Zhabotinsky (BZ) reaction dataset generator.

Implements the 3-variable Oregonator model on a 1-D spatial domain with
periodic boundary conditions.  The three coupled fields are:

    u  ~  [HBrO2]  (activator / bromous acid)
    v  ~  [Ce4+]   (oxidised catalyst)
    w  ~  [Br-]    (inhibitor / bromide)

Dimensionless Tyson-Fife form (Tyson & Fife, 1980):

    ε  * ∂u/∂t = u(1-u) - w*(u-q)/(u+q)   +  D_u * ∂²u/∂x²
       * ∂v/∂t = u  -  v                    +  D_v * ∂²v/∂x²
    δ  * ∂w/∂t = u  -  w                    +  D_w * ∂²w/∂x²

The format of the saved tensor mirrors the Gray-Scott convention used in this
repo:  [N_traj, N_env, 3, T, N_x]

References
----------
Tyson & Fife (1980) "Target patterns in a realistic model of the
    Belousov-Zhabotinsky reaction", J. Chem. Phys. 73, 2224.
Petrov et al. (1993) "Controlling chaos in the BZ reaction", Nature 361, 240.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from functools import partial


class BZDataset(Dataset):
    """
    Generates 1-D Belousov-Zhabotinsky trajectories on the fly (buffered).

    Each sample is a single trajectory of shape [3, T, N_x]:
        channel 0 : u field
        channel 1 : v field
        channel 2 : w field

    Args:
        num_traj_per_env: number of distinct initial conditions per environment
        n_points:  spatial resolution (number of grid points), default 256
        time_horizon: total integration time
        dt_eval:   time between saved snapshots
        params:    list of dicts with keys {eps, delta, q, D_u, D_v, D_w}
        method:    ODE solver, default 'RK45'
        group:     'train' or 'test' (controls RNG seed offset)
    """

    def __init__(
        self,
        num_traj_per_env: int,
        n_points: int,
        time_horizon: float,
        dt_eval: float,
        params: list,
        method: str = "RK45",
        group: str = "train",
    ):
        super().__init__()
        self.num_traj_per_env = num_traj_per_env
        self.num_env = len(params)
        self.len = num_traj_per_env * self.num_env
        self.n_points = n_points
        self.time_horizon = float(time_horizon)
        self.dt_eval = dt_eval
        self.n_steps = int(time_horizon / dt_eval)
        self.params_eq = params
        self.method = method
        self.test = group == "test"
        self.max_seed = np.iinfo(np.int32).max
        self.buffer = {}
        self.indices = [
            list(range(e * num_traj_per_env, (e + 1) * num_traj_per_env))
            for e in range(self.num_env)
        ]
        # finite-difference Laplacian matrix (periodic BC, 2nd order)
        self._L = self._build_laplacian(n_points, dx=100.0 / n_points)

    # ------------------------------------------------------------------
    # Spatial operators
    # ------------------------------------------------------------------

    @staticmethod
    def _build_laplacian(n: int, dx: float) -> np.ndarray:
        """Circulant finite-difference Laplacian for periodic BC."""
        diag = -2.0 * np.ones(n)
        off = np.ones(n - 1)
        L = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        L[0, -1] = 1.0
        L[-1, 0] = 1.0
        return L / (dx ** 2)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def _rhs(self, t: float, state: np.ndarray, env: int) -> np.ndarray:
        eps = self.params_eq[env]["eps"]
        delta = self.params_eq[env]["delta"]
        q = self.params_eq[env]["q"]
        D_u = self.params_eq[env]["D_u"]
        D_v = self.params_eq[env]["D_v"]
        D_w = self.params_eq[env]["D_w"]
        L = self._L

        n = self.n_points
        u = state[0:n]
        v = state[n:2 * n]
        w = state[2 * n:3 * n]

        # Oregonator kinetics + diffusion
        denom = np.clip(u + q, 1e-10, None)
        du = (1.0 / eps) * (u * (1.0 - u) - w * (u - q) / denom) + D_u * (L @ u)
        dv = (u - v) + D_v * (L @ v)
        dw = (1.0 / delta) * (u - w) + D_w * (L @ w)

        return np.concatenate([du, dv, dw])

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def _get_init_cond(self, traj_index: int) -> np.ndarray:
        seed = traj_index if not self.test else self.max_seed - traj_index
        rng = np.random.default_rng(seed)
        n = self.n_points

        # Perturb near the steady state (u*≈1-q, v*≈u*, w*≈u*)
        u_ss = 1.0 - 0.01  # near the oxidised steady state
        u0 = u_ss * np.ones(n) + 0.05 * rng.standard_normal(n)
        v0 = u_ss * np.ones(n) + 0.05 * rng.standard_normal(n)
        w0 = u_ss * np.ones(n) + 0.05 * rng.standard_normal(n)

        # Random localised excitation patch to seed wave propagation
        r = max(1, n // 16)
        start = rng.integers(0, n - r)
        u0[start:start + r] = 0.1
        w0[start:start + r] = 0.9

        # Clip to physically meaningful range
        u0 = np.clip(u0, 1e-4, 1.0 - 1e-4)
        v0 = np.clip(v0, 1e-4, 1.0 - 1e-4)
        w0 = np.clip(w0, 1e-4, 1.0 - 1e-4)

        return np.concatenate([u0, v0, w0])

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> dict:
        env = index // self.num_traj_per_env
        traj_idx = index % self.num_traj_per_env

        if index not in self.buffer:
            y0 = self._get_init_cond(traj_idx)
            t_span = (0.0, self.time_horizon)
            t_eval = np.arange(0.0, self.time_horizon, self.dt_eval)

            sol = solve_ivp(
                partial(self._rhs, env=env),
                t_span,
                y0,
                method=self.method,
                t_eval=t_eval,
                rtol=1e-5,
                atol=1e-7,
                dense_output=False,
            )

            n = self.n_points
            T = sol.y.shape[1]

            # sol.y : (3*n, T)
            u_traj = torch.tensor(sol.y[0:n].T, dtype=torch.float32)        # [T, n]
            v_traj = torch.tensor(sol.y[n:2*n].T, dtype=torch.float32)      # [T, n]
            w_traj = torch.tensor(sol.y[2*n:3*n].T, dtype=torch.float32)    # [T, n]

            # stack -> [3, T, n]
            state = torch.stack([u_traj, v_traj, w_traj], dim=0)
            self.buffer[index] = state.numpy()

        state = torch.from_numpy(self.buffer[index])  # [3, T, n]
        t = torch.arange(self.n_steps, dtype=torch.float32) * self.dt_eval

        return {"state": state, "t": t, "env": env}


# ------------------------------------------------------------------
# Default parameter sets (different wave speeds / oscillation periods)
# ------------------------------------------------------------------

def default_bz_params() -> list:
    """
    Four BZ environments with slightly varying ε, δ, q.
    Keeping D_u, D_v, D_w fixed; varying kinetics to get different dynamics.
    """
    base = dict(D_u=2e-3, D_v=0.0, D_w=1e-3)
    envs = [
        dict(eps=0.04,  delta=0.50, q=0.002, **base),
        dict(eps=0.05,  delta=0.45, q=0.002, **base),
        dict(eps=0.035, delta=0.55, q=0.003, **base),
        dict(eps=0.045, delta=0.40, q=0.0015, **base),
    ]
    return envs
