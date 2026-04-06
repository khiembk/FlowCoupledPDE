import torch
import torch.nn as nn


class VectorValuedGP(nn.Module):
    """
    Vector-valued Gaussian Process mapping (t, z_driver) -> z_driven.

    Conditions on exactly 2 training points per sample — the flow endpoints:
        x_0 = (t=0, z_driver_0)  ->  y_0 = z_driven_0
        x_1 = (t=1, z_driver_1)  ->  y_1 = z_driven_1

    The query point is x* = (t, z_driver_t) where z_driver_t is the linear
    interpolant of the driver at time t.

    Kernel (separable, noise-free):
        K(x_i, x_j) = k(x_i, x_j) * B
    where k is the squared-exponential (SE) kernel and B (d x d, PSD) encodes
    output correlations.  Because B cancels in the posterior mean formula, only
    the scalar SE kernel matters for the mean and its time derivative.

    With 2 training points, the posterior mean is analytically:
        z_driven(t) = w0(t) * z_driven_0 + w1(t) * z_driven_1

    and the instant velocity (d/dt posterior mean) is:
        u_ins(t)    = dw0/dt(t) * z_driven_0 + dw1/dt(t) * z_driven_1

    The weight coefficients depend only on:
        one_plus_D = 1 + ||z_driver_1 - z_driver_0||^2   (per sample)
    and the length-scale ell (learnable).

    Reference: "Flow-based Framework Solving Multiple Processes PDEs", Sec. 3.3
    """

    def __init__(self, log_length_scale: float = 0.0):
        super().__init__()
        # Learnable log length-scale; ell = exp(log_length_scale)
        self.log_length_scale = nn.Parameter(torch.tensor(float(log_length_scale)))

    @property
    def length_scale(self) -> torch.Tensor:
        return self.log_length_scale.exp()

    def forward(
        self,
        t: torch.Tensor,
        z_driver_0: torch.Tensor,
        z_driver_1: torch.Tensor,
        z_driven_0: torch.Tensor,
        z_driven_1: torch.Tensor,
    ):
        """
        Args:
            t:            [B, 1, ..., 1]  current time in [0, 1]
            z_driver_0:   [B, C, ...]     driver field at t=0
            z_driver_1:   [B, C, ...]     driver field at t=1
            z_driven_0:   [B, C, ...]     driven field at t=0
            z_driven_1:   [B, C, ...]     driven field at t=1

        Returns:
            z_driven_t:   [B, C, ...]     GP posterior mean at time t
            u_ins_driven: [B, C, ...]     d/dt(z_driven_t)
        """
        B = z_driver_0.shape[0]
        t_s = t.view(B)                 # [B]
        ell_sq = self.length_scale ** 2  # scalar

        # D² = ||z_driver_1 - z_driver_0||² per sample, normalized by feature
        # dimension so that ell is independent of spatial resolution.
        # Without normalization D² ~ C*H*W * O(1), making a = exp(-D²/2ell²) ≈ 0
        # for any reasonable ell, collapsing the GP to uncorrelated endpoints.
        d = z_driver_0[0].numel()  # C * H * W
        D_sq = ((z_driver_1 - z_driver_0).flatten(1) ** 2).sum(dim=1) / d  # [B]

        # Because z_driver_t = (1-t)*z_driver_0 + t*z_driver_1,
        #   ||x* - x_0||² = t²  + ||z_driver_t - z_driver_0||²  (normalized)
        #                 = t²  + t² * D²  = t²*(1 + D²)
        #   ||x* - x_1||² = (1-t)²*(1 + D²)
        #   ||x_0 - x_1||² =        (1 + D²)
        one_plus_D = 1.0 + D_sq         # [B], O(1) after normalization

        # Normalized SE kernel values (sigma_f² cancels in posterior mean)
        k0 = torch.exp(-t_s ** 2           * one_plus_D / (2.0 * ell_sq))  # [B]
        k1 = torch.exp(-(1.0 - t_s) ** 2  * one_plus_D / (2.0 * ell_sq))  # [B]
        a  = torch.exp(-one_plus_D                       / (2.0 * ell_sq))  # [B]

        # K_train = [[1, a], [a, 1]]  =>  K_train^{-1} = 1/(1-a²) * [[1,-a],[-a,1]]
        denom = 1.0 - a ** 2 + 1e-8    # [B]
        w0 = (k0 - a * k1) / denom     # [B]
        w1 = (k1 - a * k0) / denom     # [B]

        shape = (B,) + (1,) * (z_driven_0.ndim - 1)
        z_driven_t = w0.view(shape) * z_driven_0 + w1.view(shape) * z_driven_1

        # Time derivatives of kernel values
        # dk0/dt = k0 * (-t * one_plus_D / ell²)
        # dk1/dt = k1 * (+( 1-t) * one_plus_D / ell²)
        dk0_dt = k0 * (-t_s           * one_plus_D / ell_sq)   # [B]
        dk1_dt = k1 * ((1.0 - t_s)   * one_plus_D / ell_sq)   # [B]

        dw0_dt = (dk0_dt - a * dk1_dt) / denom                  # [B]
        dw1_dt = (dk1_dt - a * dk0_dt) / denom                  # [B]

        u_ins_driven = dw0_dt.view(shape) * z_driven_0 + dw1_dt.view(shape) * z_driven_1

        return z_driven_t, u_ins_driven
