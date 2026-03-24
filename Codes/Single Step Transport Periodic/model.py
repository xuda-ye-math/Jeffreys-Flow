# ---- 1. Header and Imports ----
from parameters import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Seed
SEED = 28
np.random.seed(SEED)
torch.manual_seed(SEED)

# Constants
PI = math.pi


# ---- 2. Circular Rational Quadratic Spline Implementation ----

class Circular_Coupling_Layer(nn.Module):
    """
    Circular Coupling Layer using Rational Quadratic Splines (RQS).
    Designed for periodic domains [-pi, pi].

    Key Features:
    1. Periodic Conditioning: Uses [cos(x), sin(x)] as input to the context net.
    2. Circular Spline: Enforces d/dx at -pi equals d/dx at pi.
    """

    def __init__(self, dim, hidden_dim, num_bins, mask_parity):
        """
        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the context network.
            num_bins (int): Number of bins for the spline.
            mask_parity (int): 0 or 1, determines which half is transformed.
        """
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2
        self.mask_parity = mask_parity

        # Domain is strictly [-pi, pi] for Periodic Well
        self.bound = PI

        # Spline Parameters
        # For Circular Spline, we predict K widths, K heights, and K derivatives.
        # The (K+1)-th derivative is forced to be equal to the 0-th derivative.
        self.num_bins = num_bins
        self.param_dim = 3 * num_bins

        # Context Network
        # Input: split_dim * 2 (because we embed x into [cos(x), sin(x)])
        # Output: split_dim * param_dim
        self.net = nn.Sequential(
            nn.Linear(self.split_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (self.dim - self.split_dim) * self.param_dim)
        )

        # ---- Strict Identity Initialization ----
        with torch.no_grad():
            # 1. Zero out weights and biases of the final projection layer
            self.net[-1].weight.fill_(0.0)
            self.net[-1].bias.fill_(0.0)

            # 2. Initialize Derivatives to be exactly 1.0
            # unnormalized_derivative = log(exp(1 - min_derivative) - 1)
            min_derivative = 1e-3
            target_val = 1.0 - min_derivative
            inv_softplus_val = np.log(np.exp(target_val) - 1)

            # Reshape bias to [out_dims, param_dim]
            out_dim = self.dim - self.split_dim
            bias_reshaped = self.net[-1].bias.view(out_dim, self.param_dim)

            # The derivatives are the last (num_bins) elements in our 3*K structure
            # Structure: [Widths (K) | Heights (K) | Derivatives (K)]
            start_idx = 2 * self.num_bins
            bias_reshaped[:, start_idx:] = float(inv_softplus_val)

    def _circular_rqs(self, inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
                      inverse=False):
        """
        RQS logic adapted for Circular Domain [-pi, pi].
        """
        # Constants
        min_bin_width = 1e-3
        min_bin_height = 1e-3
        min_derivative = 1e-3

        # 0. Normalize Inputs from [-pi, pi] to [0, 1] for Spline Calculation
        # x_norm = (x - min) / (max - min)
        domain_min = -self.bound
        domain_len = 2 * self.bound  # 2*pi

        inputs_norm = (inputs - domain_min) / domain_len

        # 1. Parameter Normalization
        # widths: Softmax to sum to 1
        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * self.num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, (1, 0), mode='constant', value=0.0)
        cumwidths = cumwidths / cumwidths[..., -1:]  # Ensure sum is exactly 1.0

        # heights: Softmax to sum to 1
        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * self.num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, (1, 0), mode='constant', value=0.0)
        cumheights = cumheights / cumheights[..., -1:]  # Ensure sum is exactly 1.0

        # derivatives: Softplus
        derivatives = F.softplus(unnormalized_derivatives) + min_derivative
        # CIRCULAR CONSTRAINT: Pad the last derivative with the first one
        # d_K = d_0
        derivatives = torch.cat([derivatives, derivatives[..., :1]], dim=-1)

        # 2. Spline Transformation (Standard RQS logic on [0, 1])
        # We process all inputs (no masking needed as domain is wrapped/bounded)

        # Prepare outputs
        outputs_norm = torch.zeros_like(inputs_norm)
        logabsdet = torch.zeros_like(inputs_norm)

        # For numerical stability, clamp inputs to [0, 1] (handling potential float errors)
        # In a true circular flow, we would wrap inputs, but here we assume inputs are in [-pi, pi]
        inputs_norm = torch.clamp(inputs_norm, 0.0, 1.0 - 1e-6)

        if inverse:
            # --- Inverse (z -> x) ---
            bin_idx = torch.searchsorted(cumheights, inputs_norm.unsqueeze(-1)) - 1
            bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

            # Gather Params
            def gather(tensor):
                return tensor.gather(1, bin_idx).squeeze()

            input_w = gather(widths)
            input_h = gather(heights)
            input_cum_w = gather(cumwidths)
            input_cum_h = gather(cumheights)
            input_d = gather(derivatives)
            input_d_plus = gather(derivatives[..., 1:])  # Offset for d_plus

            # Quadratic Formula
            theta = (inputs_norm - input_cum_h)  # Relative Y

            # Coefficients: a*xi^2 + b*xi + c = 0
            # s = h/w
            s = input_h / input_w

            a = input_h * (s - input_d) + theta * (input_d + input_d_plus - 2 * s)
            b = input_h * input_d - theta * (input_d + input_d_plus - 2 * s)
            c = -s * theta

            delta = b.pow(2) - 4 * a * c
            xi = 2 * c / (-b - torch.sqrt(torch.abs(delta)))

            # Calculate Output (Normalized)
            outputs_norm = input_cum_w + xi * input_w

            # Log Determinant (Derivative of Inverse)
            numerator_d = (input_h ** 2) * (input_d_plus * xi ** 2 + 2 * s * xi * (1 - xi) + input_d * (1 - xi) ** 2)
            denominator_d = (s + (input_d + input_d_plus - 2 * s) * xi * (1 - xi)) ** 2
            derivative = numerator_d / denominator_d

            # Chain rule: log|dx/dy| (normalized scale)
            # We need to add log(domain_len) adjustment?
            # dy_phys/dx_phys = (dy_norm * L) / (dx_norm * L) = dy_norm/dx_norm. Scaling cancels.
            # So log_det is just the spline log_det.

            logabsdet = -(torch.log(derivative) - 2 * torch.log(input_w))

        else:
            # --- Forward (x -> z) ---
            bin_idx = torch.searchsorted(cumwidths, inputs_norm.unsqueeze(-1)) - 1
            bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

            def gather(tensor):
                return tensor.gather(1, bin_idx).squeeze()

            input_w = gather(widths)
            input_h = gather(heights)
            input_cum_w = gather(cumwidths)
            input_cum_h = gather(cumheights)
            input_d = gather(derivatives)
            input_d_plus = gather(derivatives[..., 1:])

            xi = (inputs_norm - input_cum_w) / input_w
            s = input_h / input_w

            # RQS Formula
            numerator = input_h * (s * xi ** 2 + input_d * xi * (1 - xi))
            denominator = s + (input_d + input_d_plus - 2 * s) * xi * (1 - xi)

            outputs_norm = input_cum_h + numerator / denominator

            # Derivative dy/dx
            d_numerator = (input_h ** 2) * (input_d_plus * xi ** 2 + 2 * s * xi * (1 - xi) + input_d * (1 - xi) ** 2)
            d_denominator = denominator ** 2

            logabsdet = torch.log(d_numerator) - torch.log(d_denominator) - 2 * torch.log(input_w)

        # 3. De-normalize Outputs from [0, 1] back to [-pi, pi]
        outputs = outputs_norm * domain_len + domain_min

        # Enforce Periodic Wrapping (Crucial for stability)
        # Map output back to [-pi, pi]
        outputs = torch.remainder(outputs - domain_min, domain_len) + domain_min

        return outputs, logabsdet

    def forward(self, x, inverse=False):
        """
        Main coupling logic.
        """
        # Split features
        if self.mask_parity == 0:
            x_id, x_tr = x[:, :self.split_dim], x[:, self.split_dim:]
        else:
            x_tr, x_id = x[:, :self.split_dim], x[:, self.split_dim:]

        # ---- Periodic Embedding ----
        # Instead of passing x_id directly, we pass [cos(x), sin(x)]
        # x_id is assumed to be in [-pi, pi].
        x_id_cos = torch.cos(x_id)
        x_id_sin = torch.sin(x_id)
        x_id_embedded = torch.cat([x_id_cos, x_id_sin], dim=1)

        # Run Context Net
        params = self.net(x_id_embedded)
        params = params.reshape(x.shape[0], -1, 3 * self.num_bins)  # 3*K params

        W = params[..., :self.num_bins]
        H = params[..., self.num_bins: 2 * self.num_bins]
        D = params[..., 2 * self.num_bins:]

        # Transform
        x_tr_flat = x_tr.reshape(-1)
        W_flat = W.reshape(-1, self.num_bins)
        H_flat = H.reshape(-1, self.num_bins)
        D_flat = D.reshape(-1, self.num_bins)

        y_tr_flat, log_det_flat = self._circular_rqs(x_tr_flat, W_flat, H_flat, D_flat, inverse=inverse)

        y_tr = y_tr_flat.reshape(x_tr.shape)
        log_det = log_det_flat.reshape(x_tr.shape).sum(dim=1)

        # Merge
        if self.mask_parity == 0:
            y = torch.cat([x_id, y_tr], dim=1)
        else:
            y = torch.cat([y_tr, x_id], dim=1)

        return y, log_det


# ---- 3. Flow Model Wrapper ----

class Normalizing_Flow(nn.Module):
    """
    Circular Spline Flow for Periodic Distributions.
    Domain: [-pi, pi]^D
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

        # Read Configuration from parameters.py
        dim = para.DIM
        hidden_dim = HIDDEN_DIM
        num_layers = NUM_LAYERS
        num_bins = NUM_BINS

        # Verify Bounds (Must be periodic for this model to make sense)
        # We assume strict [-pi, pi] for both dimensions
        print(f"[Model] Initializing {num_layers}-layer Circular Spline Flow.")
        print(f"        Domain: [-pi, pi] (Periodic)")

        for i in range(num_layers):
            mask_parity = i % 2
            # No need to pass limits, they are fixed to [-pi, pi]
            self.layers.append(Circular_Coupling_Layer(dim, hidden_dim, num_bins, mask_parity))

    def forward(self, x, inverse=False):
        """
        Bidirectional Forward Pass.
        """
        log_det_sum = 0
        z = x

        if not inverse:
            # Forward: x -> z (Layer 0 -> N)
            for layer in self.layers:
                z, log_det = layer(z, inverse=False)
                log_det_sum += log_det
        else:
            # Inverse: z -> x (Layer N -> 0)
            for layer in reversed(self.layers):
                z, log_det = layer(z, inverse=True)
                log_det_sum += log_det

        # Wrap final output to [-pi, pi] just in case
        z = torch.remainder(z - (-PI), 2 * PI) + (-PI)

        return z, log_det_sum


# ---- 4. Verification (Main) ----

if __name__ == "__main__":
    print(f"--- Verifying Circular Model for {para.NAME} ---")

    model = Normalizing_Flow()
    model.eval()

    # 1. Identity Check
    B = 10
    # Create random data inside [-pi, pi]
    x = (torch.rand(B, para.DIM) * 2 * PI) - PI

    with torch.no_grad():
        z, log_det_fwd = model(x, inverse=False)
        x_rec, log_det_inv = model(z, inverse=True)

        # For Identity check: F^-1(x) on raw x
        x_inv, _ = model(x, inverse=True)


    # Note: For circular variables, difference should be modulo 2pi
    # But for identity init (small deformation), direct diff is fine usually.
    def circ_diff(a, b):
        d = torch.abs(a - b)
        d = torch.min(d, 2 * PI - d)
        return d.max().item()


    print(f"\n[Identity F(x) approx x]")
    print(f"Max Circular Error: {circ_diff(z, x):.8e}")

    print(f"\n[Identity F^-1(x) approx x]")
    print(f"Max Circular Error: {circ_diff(x_inv, x):.8e}")

    print(f"\n[LogDet Zero Check]")
    print(f"Max |LogDet_Fwd|: {log_det_fwd.abs().max().item():.8e}")

    # 2. Periodicity "Seam" Test
    print(f"\n[Seam Test: Continuity at Boundary]")
    # Create x1 near -pi, x2 near pi (physically same point)
    # x1 = [-pi + epsilon, 0]
    # x2 = [ pi - epsilon, 0]
    eps = 1e-4
    x1 = torch.tensor([[-PI + eps, 0.0]])
    x2 = torch.tensor([[PI - eps, 0.0]])

    with torch.no_grad():
        z1, _ = model(x1, inverse=False)
        z2, _ = model(x2, inverse=False)

    # z1 should be very close to z2 (modulo 2pi)
    # Actually, if map is Identity, z1 ~ -pi, z2 ~ pi.
    # They are far in Euclidean but close in Manifold.
    # The transformation parameters (derivative, etc) at that point should be identical.

    print(f"Input x1: {x1.numpy()}")
    print(f"Input x2: {x2.numpy()}")
    print(f"Output z1: {z1.numpy()}")
    print(f"Output z2: {z2.numpy()}")

    seam_error = circ_diff(z1, z2)
    print(f"Seam Consistency Error (z1 vs z2): {seam_error:.8e}")

    if seam_error < 1e-3:  # Slightly loose tol for Seam due to float eps
        print(">> SUCCESS: Model respects periodicity.")
    else:
        # Note: If Identity, z1=-3.14, z2=3.14. Distance is 0 in circular.
        # circ_diff handles this.
        print(">> WARNING: Seam discontinuity detected.")