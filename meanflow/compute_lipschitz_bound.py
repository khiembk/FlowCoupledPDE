"""
compute_lipschitz_bound.py
==========================
Computes the Lipschitz constant curve L*_t and the smoothness constant

    K_{0,τ} = exp( ∫₀¹ L*_t dt )

for two choices of flow-path interpolant:

  • "linear" — Appendix C.1:  Z_t = (1−t)Z₀ + t Z₁
  • "gp"     — Appendix C.2:  Z_t = ω₁(t)Z₀ + ω₂(t)Z₁  via per-sample SE-GP

In both cases the target flow map is

    M^tar(t, z) = E[ Ż_t | Z_t = z ]

and its Lipschitz constant at time t is

    L*_t = sup_z  ‖∇_z M^tar(t, z)‖_op   (spectral / operator norm)

estimated empirically over the training set via Nadaraya-Watson regression
+ PyTorch autograd Jacobians.

The integral  ∫₀¹ L*_t dt  is computed with the trapezoidal rule over
T = 100 equally-spaced points.

Dataset shape convention
------------------------
  z0_all, z1_all : torch.Tensor  of shape  [N, n, d]
    N  = number of training samples
    n  = number of coupled processes  (e.g. 2 for Lotka-Volterra)
    d  = spatial dimension / mesh size (flattened)

References
----------
  • Appendix A  (GP implementation details)
  • Appendix C  (smoothness of velocity fields)
  • Theorem 1   (K_{0,τ} links L_local → W₂ error)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian as torch_jacobian

# ═══════════════════════════════════════════════════════════════
# 1.  SE-Kernel primitives  (Appendix A, Eq. 44)
# ═══════════════════════════════════════════════════════════════

def se_kernel(xi: torch.Tensor, xj: torch.Tensor,
              sigma_f: float = 1.0, l: float = 1.0) -> torch.Tensor:
    """
    Scalar squared-exponential kernel between two vectors.

        k(xi, xj) = σ_f² exp( −‖xi − xj‖² / (2ℓ²) )
    """
    diff = xi - xj
    return sigma_f ** 2 * torch.exp(-0.5 / l ** 2 * (diff * diff).sum())


def dk_dt_se(x_star: torch.Tensor,
             xj: torch.Tensor,
             k_val: torch.Tensor,
             dx_star_dt: torch.Tensor,
             l: float = 1.0) -> torch.Tensor:
    """
    Time-derivative of k(x*(t), xj) where xj is a fixed training point.

        d/dt k(x*, xj) = k(x*, xj) · [−(x* − xj)/ℓ²] · dx*/dt
    """
    return k_val * ((-(x_star - xj) / l ** 2) * dx_star_dt).sum()


# ═══════════════════════════════════════════════════════════════
# 2.  Per-sample GP interpolant  (Appendix A, Eqs. 33–44)
#
#   Training set (2 points per sample):
#     X = { (0, z0_drv), (1, z1_drv) }
#     y = { z0_tgt,       z1_tgt     }
#
#   Query:  x* = (t, z_t_drv)  where z_t_drv = linear interpolant
#
#   GP mean  (Eq. 43):
#     f*(x*) = (k_x*^T · K(X,X)⁻¹) ⊗ I_d · y
#            = ω · y,   ω ∈ ℝ²
#
#   Instant velocity  (Eq. 19–20):
#     u_ins_tgt = dω/dt · y
# ═══════════════════════════════════════════════════════════════

def gp_state_and_velocity(
    t: torch.Tensor,
    z0_drv: torch.Tensor, z1_drv: torch.Tensor,   # driver at τ=0 and τ=1  [d]
    z0_tgt: torch.Tensor, z1_tgt: torch.Tensor,   # target at τ=0 and τ=1  [d]
    sigma_f: float = 1.0,
    l: float = 1.0,
):
    """
    Returns (z_t_tgt, u_ins_tgt) for a single sample using the per-sample GP.

    z_t_tgt   : predicted state of the target process at flow-time t   [d]
    u_ins_tgt : instant velocity  d/dt z_t_tgt                         [d]
    """
    # ── training inputs x₀ = (0, z₀_drv), x₁ = (1, z₁_drv) ──
    x_tr = [
        torch.cat([torch.zeros(1, dtype=t.dtype), z0_drv]),   # [d+1]
        torch.cat([torch.ones(1,  dtype=t.dtype), z1_drv]),   # [d+1]
    ]
    y_tr = torch.stack([z0_tgt, z1_tgt])    # [2, d]

    # ── query:  x* = (t, z_t_drv)  where z_t_drv is the driver linear interp ──
    z_t_drv = (1.0 - t) * z0_drv + t * z1_drv
    x_star  = torch.cat([t.unsqueeze(0), z_t_drv])           # [d+1]

    # ── 2×2 kernel matrix  K(X, X) ──
    K = torch.stack([
        torch.stack([se_kernel(x_tr[0], x_tr[0], sigma_f, l),
                     se_kernel(x_tr[0], x_tr[1], sigma_f, l)]),
        torch.stack([se_kernel(x_tr[1], x_tr[0], sigma_f, l),
                     se_kernel(x_tr[1], x_tr[1], sigma_f, l)]),
    ])                                       # [2, 2]

    # Add jitter for numerical stability
    K = K + 1e-6 * torch.eye(2, dtype=K.dtype)
    K_inv = torch.inverse(K)                 # [2, 2]

    # ── query-to-train covariance  k_x* = [k(x*, x₀), k(x*, x₁)] ──
    k_s = torch.stack([
        se_kernel(x_star, x_tr[0], sigma_f, l),
        se_kernel(x_star, x_tr[1], sigma_f, l),
    ])                                       # [2]

    # ── GP weights  ω = k_x*^T K⁻¹  (Eq. 43 without B, which cancels) ──
    omega = k_s @ K_inv                      # [2]

    # ── predicted state ──
    z_t_tgt = omega @ y_tr                   # [d]

    # ── instant velocity: dω/dt · y ──
    #    dx*/dt = (1,  z₁_drv − z₀_drv)  since x*(t) = (t, (1−t)z₀ + tz₁)
    dx_star_dt = torch.cat([
        torch.ones(1,  dtype=t.dtype),
        (z1_drv - z0_drv),
    ])                                       # [d+1]

    dk_s_dt = torch.stack([
        dk_dt_se(x_star, x_tr[0], k_s[0], dx_star_dt, l),
        dk_dt_se(x_star, x_tr[1], k_s[1], dx_star_dt, l),
    ])                                       # [2]

    d_omega_dt = dk_s_dt @ K_inv             # [2]
    u_ins_tgt  = d_omega_dt @ y_tr           # [d]

    return z_t_tgt, u_ins_tgt


# ═══════════════════════════════════════════════════════════════
# 3.  Build (Z_t, Ż_t) dataset at a given t
#     for both interpolant choices
# ═══════════════════════════════════════════════════════════════

def build_flow_samples_linear(
    t: torch.Tensor,
    z0_all: torch.Tensor,
    z1_all: torch.Tensor,
):
    """
    Appendix C.1 — linear interpolant.

        Z_t  = (1−t) Z₀ + t Z₁
        Ż_t  = Z₁ − Z₀    (constant in t)

    z0_all, z1_all : [N, n, d]
    Returns Z_t [N, n·d], Zdot_t [N, n·d]
    """
    N = z0_all.shape[0]
    Z_t    = ((1.0 - t) * z0_all + t * z1_all).reshape(N, -1)
    Zdot_t = (z1_all - z0_all).reshape(N, -1)
    return Z_t, Zdot_t


def build_flow_samples_gp(
    t: torch.Tensor,
    z0_all: torch.Tensor,
    z1_all: torch.Tensor,
    sigma_f: float = 1.0,
    l: float = 1.0,
):
    """
    Appendix C.2 — GP-based interpolant with Dice strategy.

    For each sample k and each choice of driver process, the driven processes
    are predicted by the per-sample GP.  We average over all n driver choices
    (uniform Dice probability p = 1/n) to get a single (Z_t^k, Ż_t^k).

    z0_all, z1_all : [N, n, d]
    Returns Z_t [N, n·d], Zdot_t [N, n·d]
    """
    N, n, d = z0_all.shape
    Z_t_list, Zdot_t_list = [], []

    for k in range(N):
        # Accumulate over all driver choices, then average (Dice strategy)
        Z_dice_sum    = torch.zeros(n * d, dtype=z0_all.dtype)
        Zdot_dice_sum = torch.zeros(n * d, dtype=z0_all.dtype)

        for drv in range(n):
            z_parts, zdot_parts = [], []
            for proc in range(n):
                if proc == drv:
                    # Driver: linear interpolant (Eq. 22)
                    zt_p   = (1.0 - t) * z0_all[k, proc] + t * z1_all[k, proc]
                    zdot_p = z1_all[k, proc] - z0_all[k, proc]
                else:
                    # Driven: GP prediction (Appendix A)
                    zt_p, zdot_p = gp_state_and_velocity(
                        t,
                        z0_all[k, drv],  z1_all[k, drv],
                        z0_all[k, proc], z1_all[k, proc],
                        sigma_f, l,
                    )
                z_parts.append(zt_p)
                zdot_parts.append(zdot_p)

            Z_dice_sum    += torch.cat(z_parts)
            Zdot_dice_sum += torch.cat(zdot_parts)

        Z_t_list.append(Z_dice_sum / n)
        Zdot_t_list.append(Zdot_dice_sum / n)

    return torch.stack(Z_t_list), torch.stack(Zdot_t_list)   # [N, n·d]


# ═══════════════════════════════════════════════════════════════
# 4.  Nadaraya-Watson estimator for M^tar(t, z)
#
#     M^tar(t, z) = E[Ż_t | Z_t = z]  ≈  Σ_k w_k Ż_k / Σ_k w_k
#     w_k = exp(−‖z − Z_t^k‖² / (2 bw²))
#
#     Uses log-softmax for numerical stability.
# ═══════════════════════════════════════════════════════════════

def nadaraya_watson(
    z: torch.Tensor,            # query point  [D]
    Z_support: torch.Tensor,    # support set  [N, D]
    Zdot_support: torch.Tensor, # labels       [N, D]
    bw: float = 1.0,
) -> torch.Tensor:
    """Returns M^tar(z) = Σ_k w_k Ż_k   (weighted mean of velocities)."""
    dists   = ((Z_support - z.unsqueeze(0)) ** 2).sum(-1)   # [N]
    log_w   = -0.5 / bw ** 2 * dists
    weights = torch.softmax(log_w, dim=0)                    # [N]
    return (weights.unsqueeze(-1) * Zdot_support).sum(0)     # [D]


# ═══════════════════════════════════════════════════════════════
# 5.  L*_t at a single time point
#
#   L*_t = sup_z ‖∇_z M^tar(t, z)‖_op
#
#   Estimated as: max over training points of spectral norm of Jacobian.
#   Jacobian is computed by torch.autograd.functional.jacobian.
# ═══════════════════════════════════════════════════════════════

def compute_L_star_t(
    Z_t: torch.Tensor,      # [N, D]
    Zdot_t: torch.Tensor,   # [N, D]
    bw: float = 1.0,
) -> torch.Tensor:
    """
    Returns the empirical  L*_t = max_k ‖J_k‖_op
    where J_k = ∇_z M^tar evaluated at z = Z_t[k].
    """
    N, D = Z_t.shape
    max_spectral_norm = torch.tensor(0.0, dtype=Z_t.dtype)

    # Detach support arrays so autograd only flows through the query z
    Z_sup    = Z_t.detach()
    Zdot_sup = Zdot_t.detach()

    for i in range(N):
        z_i = Z_t[i].detach().clone()   # [D]

        def M_tar(z):
            return nadaraya_watson(z, Z_sup, Zdot_sup, bw)

        # Jacobian shape: [D, D]
        J = torch_jacobian(M_tar, z_i)
        s = torch.linalg.matrix_norm(J, ord=2)   # largest singular value
        if s > max_spectral_norm:
            max_spectral_norm = s

    return max_spectral_norm


# ═══════════════════════════════════════════════════════════════
# 6.  Main function: sweep t ∈ [0, 1], collect L*_t,
#     then integrate to get K_{0,τ}
# ═══════════════════════════════════════════════════════════════

def compute_K_and_L_curve(
    z0_all: torch.Tensor,       # [N, n, d]  initial conditions
    z1_all: torch.Tensor,       # [N, n, d]  solutions at time τ
    mode: str   = "gp",         # "gp" | "linear"
    T: int      = 100,          # number of quadrature points
    bw: float   = 1.0,          # NW bandwidth (auto-set if None)
    sigma_f: float = 1.0,       # GP signal std
    l: float    = 1.0,          # GP length-scale
    eps: float  = 1e-2,         # avoid t→0 instability in linear 1/t gradient
    verbose: bool = True,
):
    """
    Compute the L*_t curve and K_{0,τ} = exp(∫₀¹ L*_t dt).

    Parameters
    ----------
    z0_all  : [N, n, d]   initial-condition snapshots
    z1_all  : [N, n, d]   solution snapshots at τ
    mode    : "gp" or "linear"
    T       : quadrature grid size (default 100)
    bw      : NW kernel bandwidth.  If 0.0, auto-set to median inter-point dist.
    sigma_f : GP hyperparameter σ_f
    l       : GP hyperparameter ℓ
    eps     : left endpoint of t-grid (avoids 1/t singularity)
    verbose : print progress

    Returns
    -------
    t_grid   : Tensor [T]   time grid
    L_curve  : Tensor [T]   L*_t estimates
    integral : scalar       ∫₀¹ L*_t dt   (trapezoidal)
    K_0tau   : scalar       exp(integral)
    """
    assert mode in ("gp", "linear"), "mode must be 'gp' or 'linear'"

    t_grid  = torch.linspace(eps, 1.0, T, dtype=z0_all.dtype)
    L_curve = torch.zeros(T, dtype=z0_all.dtype)

    for idx, t_val in enumerate(t_grid):
        if verbose and (idx % 10 == 0 or idx == T - 1):
            print(f"  [{mode.upper()}] step {idx+1:3d}/{T}   t = {t_val:.4f}")

        t = t_val.clone()

        # ── Step 1: build (Z_t, Ż_t) dataset ──
        if mode == "linear":
            Z_t, Zdot_t = build_flow_samples_linear(t, z0_all, z1_all)
        else:
            Z_t, Zdot_t = build_flow_samples_gp(t, z0_all, z1_all, sigma_f, l)

        # ── Step 2: auto-bandwidth via median heuristic if requested ──
        effective_bw = bw
        if bw == 0.0:
            dists = torch.cdist(Z_t, Z_t)
            mask  = dists > 0
            effective_bw = float(dists[mask].median()) / np.sqrt(2.0)
            effective_bw = max(effective_bw, 1e-6)

        # ── Step 3: compute L*_t ──
        L_curve[idx] = compute_L_star_t(Z_t, Zdot_t, bw=effective_bw)

    # ── Step 4: trapezoidal integration ──
    integral = torch.trapezoid(L_curve, t_grid)
    K_0tau   = torch.exp(integral)

    return t_grid, L_curve, integral, K_0tau


# ═══════════════════════════════════════════════════════════════
# 7.  Plotting utility
# ═══════════════════════════════════════════════════════════════

def plot_L_curve(t_grid, L_curve_linear, L_curve_gp,
                 integral_linear, integral_gp,
                 K_linear, K_gp):
    """Side-by-side comparison of L*_t for both interpolants."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, L_curve, integral, K, label, color in zip(
        axes,
        [L_curve_linear, L_curve_gp],
        [integral_linear, integral_gp],
        [K_linear, K_gp],
        ["Linear interpolant (App. C.1)", "GP modeling (App. C.2)"],
        ["steelblue", "darkorange"],
    ):
        ax.plot(t_grid.numpy(), L_curve.numpy(), color=color, linewidth=2)
        ax.fill_between(t_grid.numpy(), 0, L_curve.numpy(),
                        alpha=0.15, color=color)
        ax.set_xlabel("Flow time  t", fontsize=12)
        ax.set_ylabel("L*_t  (spectral norm of Jacobian)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.text(0.05, 0.92,
                f"∫L*_t dt = {integral:.4f}\nK_{{0,τ}} = {K:.4f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("L_curve_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved: L_curve_comparison.png")


# ═══════════════════════════════════════════════════════════════
# 8.  Quick demo  (replace with your actual dataset)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)

    # ── Synthetic toy dataset mimicking a 2-process, 4-point mesh ──
    N   = 20     # training samples
    n   = 2      # number of coupled processes
    d   = 4      # spatial dimension (flattened mesh size)
    T   = 100    # quadrature grid size

    z0_all = torch.randn(N, n, d)
    z1_all = z0_all + 0.3 * torch.randn(N, n, d)   # small perturbation

    print("=" * 60)
    print("Computing L*_t curve — LINEAR interpolant")
    print("=" * 60)
    t_grid, L_linear, integ_linear, K_linear = compute_K_and_L_curve(
        z0_all, z1_all,
        mode    = "linear",
        T       = T,
        bw      = 0.0,   # auto bandwidth
        verbose = True,
    )
    print(f"\n  ∫₀¹ L*_t dt  (linear) = {integ_linear:.6f}")
    print(f"  K_{{0,τ}}     (linear) = {K_linear:.6f}")

    print("\n" + "=" * 60)
    print("Computing L*_t curve — GP modeling")
    print("=" * 60)
    t_grid, L_gp, integ_gp, K_gp = compute_K_and_L_curve(
        z0_all, z1_all,
        mode    = "gp",
        T       = T,
        sigma_f = 1.0,
        l       = 1.0,
        bw      = 0.0,   # auto bandwidth
        verbose = True,
    )
    print(f"\n  ∫₀¹ L*_t dt  (GP)     = {integ_gp:.6f}")
    print(f"  K_{{0,τ}}     (GP)     = {K_gp:.6f}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  K_{{0,τ}} linear  = {K_linear:.4f}  "
          f"(bound multiplier in Theorem 1: {K_linear**2:.4f})")
    print(f"  K_{{0,τ}} GP      = {K_gp:.4f}  "
          f"(bound multiplier in Theorem 1: {K_gp**2:.4f})")
    print(f"\n  GP modeling {'reduces' if K_gp < K_linear else 'increases'} K by "
          f"{abs(K_gp - K_linear) / K_linear * 100:.1f}% vs linear")

    plot_L_curve(t_grid, L_linear, L_gp,
                 integ_linear, integ_gp,
                 K_linear, K_gp)
