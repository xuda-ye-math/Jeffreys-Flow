# ---- 1. Header and Imports ----
from parameters import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Seed
SEED = 28
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---- 2. Rational Quadratic Spline Implementation ----

class Coupling_Layer(nn.Module):
    """
    Coupling_Layer using Rational Quadratic Splines (RQS).
    Splits input x into [x_id, x_tr]. Transforms x_tr based on x_id.
    """

    def __init__(self, dim, hidden_dim, num_bins, mask_parity, limits):
        """
        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the context network.
            num_bins (int): Number of bins for the spline.
            mask_parity (int): 0 or 1, determines which half is transformed.
            limits (list): [min, max] for the dimension being transformed.
        """
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2
        self.mask_parity = mask_parity

        # Determine dimensions for Conditioner (ID) and Transformed (TR) parts
        if mask_parity == 0:
            self.id_dim = self.split_dim
            self.tr_dim = dim - self.split_dim
        else:
            self.id_dim = dim - self.split_dim
            self.tr_dim = self.split_dim

        # Set bounds (Assumes isotropic limits for the transformed dimensions)
        self.limit_min = float(limits[0])
        self.limit_max = float(limits[1])

        # Spline Parameters
        # We need (3 * num_bins + 1) parameters per dimension being transformed.
        self.num_bins = num_bins
        self.param_dim = 3 * num_bins + 1

        # Context Network
        self.net = nn.Sequential(
            nn.Linear(self.id_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.tr_dim * self.param_dim)
        )

        # ---- Strict Identity Initialization ----
        with torch.no_grad():
            # 1. Zero out weights and biases of the final projection layer
            self.net[-1].weight.fill_(0.0)
            self.net[-1].bias.fill_(0.0)

            # 2. Initialize Derivatives to be exactly 1.0
            min_derivative = 1e-3
            target_val = 1.0 - min_derivative
            inv_softplus_val = np.log(np.exp(target_val) - 1)

            # Reshape bias to [tr_dim, param_dim]
            bias_reshaped = self.net[-1].bias.view(self.tr_dim, self.param_dim)

            # The derivatives are the last (num_bins + 1) elements
            start_idx = 2 * self.num_bins
            bias_reshaped[:, start_idx:] = float(inv_softplus_val)

    def _rational_quadratic_spline(self, inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
                                   inverse=False):
        """
        Core RQS Logic.
        """
        # Constants
        min_bin_width = 1e-3
        min_bin_height = 1e-3
        min_derivative = 1e-3

        # 1. Parameter Normalization
        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * self.num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, (1, 0), mode='constant', value=0.0)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * self.num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, (1, 0), mode='constant', value=0.0)

        derivatives = F.softplus(unnormalized_derivatives) + min_derivative
        derivatives = F.pad(derivatives, (1, 1), mode='constant', value=1.0)

        # 2. Masking (Identity outside bounds)
        left, right = self.limit_min, self.limit_max
        bottom, top = self.limit_min, self.limit_max

        inside_interval_mask = (inputs >= left) & (inputs <= right)
        outside_interval_mask = ~inside_interval_mask

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0.0

        if torch.sum(inside_interval_mask) == 0:
            return outputs, logabsdet

        inputs_in = inputs[inside_interval_mask]

        # 3. Spline Transformation
        if inverse:
            # --- Inverse (Generative: z -> x) ---
            inputs_norm = (inputs_in - bottom) / (top - bottom)

            bin_idx = torch.searchsorted(cumheights[inside_interval_mask, :], inputs_norm.unsqueeze(-1)) - 1
            bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

            def gather_param(tensor, idx):
                return tensor[inside_interval_mask].gather(1, idx).squeeze()

            input_w = gather_param(widths, bin_idx)
            input_h = gather_param(heights, bin_idx)
            input_cum_w = gather_param(cumwidths, bin_idx)
            input_cum_h = gather_param(cumheights, bin_idx)
            input_d = gather_param(derivatives, bin_idx)
            input_d_plus = gather_param(derivatives, bin_idx + 1)

            theta = (inputs_norm - input_cum_h)
            s = input_h / input_w

            a = input_h * (s - input_d) + theta * (input_d + input_d_plus - 2 * s)
            b = input_h * input_d - theta * (input_d + input_d_plus - 2 * s)
            c = -s * theta

            delta = b.pow(2) - 4 * a * c
            xi = 2 * c / (-b - torch.sqrt(torch.abs(delta)))

            outputs_norm = input_cum_w + xi * input_w
            outputs[inside_interval_mask] = outputs_norm * (right - left) + left

            numerator_d = (input_h ** 2) * (input_d_plus * xi ** 2 + 2 * s * xi * (1 - xi) + input_d * (1 - xi) ** 2)
            denominator_d = (s + (input_d + input_d_plus - 2 * s) * xi * (1 - xi)) ** 2
            derivative = numerator_d / denominator_d

            logabsdet[inside_interval_mask] = -(torch.log(derivative) - 2 * torch.log(input_w))

        else:
            # --- Forward (Normalizing: x -> z) ---
            inputs_norm = (inputs_in - left) / (right - left)

            bin_idx = torch.searchsorted(cumwidths[inside_interval_mask, :], inputs_norm.unsqueeze(-1)) - 1
            bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)

            def gather_param(tensor, idx):
                return tensor[inside_interval_mask].gather(1, idx).squeeze()

            input_w = gather_param(widths, bin_idx)
            input_h = gather_param(heights, bin_idx)
            input_cum_w = gather_param(cumwidths, bin_idx)
            input_cum_h = gather_param(cumheights, bin_idx)
            input_d = gather_param(derivatives, bin_idx)
            input_d_plus = gather_param(derivatives, bin_idx + 1)

            xi = (inputs_norm - input_cum_w) / input_w
            s = input_h / input_w

            numerator = input_h * (s * xi ** 2 + input_d * xi * (1 - xi))
            denominator = s + (input_d + input_d_plus - 2 * s) * xi * (1 - xi)

            outputs_norm = input_cum_h + numerator / denominator
            outputs[inside_interval_mask] = outputs_norm * (top - bottom) + bottom

            d_numerator = (input_h ** 2) * (input_d_plus * xi ** 2 + 2 * s * xi * (1 - xi) + input_d * (1 - xi) ** 2)
            d_denominator = denominator ** 2

            logabsdet[inside_interval_mask] = torch.log(d_numerator) - torch.log(d_denominator) - 2 * torch.log(input_w)

        return outputs, logabsdet

    def forward(self, x, inverse=False):
        """
        Main coupling logic.
        """
        # Split features based on parity
        if self.mask_parity == 0:
            x_id, x_tr = x[:, :self.split_dim], x[:, self.split_dim:]
        else:
            x_tr, x_id = x[:, :self.split_dim], x[:, self.split_dim:]

        # Run Context Net
        params = self.net(x_id)
        params = params.reshape(x.shape[0], -1, 3 * self.num_bins + 1)

        W = params[..., :self.num_bins]
        H = params[..., self.num_bins: 2 * self.num_bins]
        D = params[..., 2 * self.num_bins:]

        # Transform
        x_tr_flat = x_tr.reshape(-1)
        W_flat = W.reshape(-1, self.num_bins)
        H_flat = H.reshape(-1, self.num_bins)
        D_flat = D.reshape(-1, self.num_bins + 1)

        y_tr_flat, log_det_flat = self._rational_quadratic_spline(x_tr_flat, W_flat, H_flat, D_flat, inverse=inverse)

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
    RQS Flow constructed using parameters from parameters.py.
    STRICTLY relies on 'X_LIM_COMPUTE' for spline limits. No fallback logic.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

        # Read Configuration from parameters.py
        dim = para.DIM
        hidden_dim = HIDDEN_DIM
        num_layers = NUM_LAYERS
        num_bins = NUM_BINS

        # ---- Calculate Spline Limits (Strict) ----
        # Strategy:
        # Strictly check for global unified limit 'X_LIM_COMPUTE'.
        # Raise error if not found.

        if not hasattr(para, "X_LIM_COMPUTE"):
            raise ValueError(f"[Model] Error: 'X_LIM_COMPUTE' not found in parameters. Strict mode enabled.")

        lim = para.X_LIM_COMPUTE
        x_limits = [float(lim[0]), float(lim[1])]

        print(f"[Model] Initializing {num_layers}-layer Spline Flow.")
        print(f"        Dimensions: {dim}")
        print(f"        Global Spline Limits: {x_limits} (Strictly from X_LIM_COMPUTE)")

        for i in range(num_layers):
            mask_parity = i % 2
            self.layers.append(Coupling_Layer(dim, hidden_dim, num_bins, mask_parity, limits=x_limits))

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

        return z, log_det_sum


# ---- 4. Verification (Main) ----

if __name__ == "__main__":
    print(f"--- Verifying Model (Dim={para.DIM}) ---")

    model = Normalizing_Flow()
    model.eval()

    # 1. Identity Check
    B = 10

    # Retrieve the computed limits from the first layer for consistency
    limit_min = model.layers[0].limit_min
    limit_max = model.layers[0].limit_max
    width = limit_max - limit_min

    print(f"    Verification Bounds: [{limit_min:.2f}, {limit_max:.2f}]")

    # Ensure points are strictly inside to trigger spline logic
    # x ~ Uniform[limit_min + 0.5, limit_max - 0.5]
    x = torch.rand(B, para.DIM) * (width - 1.0) + (limit_min + 0.5)

    with torch.no_grad():
        # Forward pass F(x)
        z, log_det_fwd = model(x, inverse=False)

        # Inverse pass F^-1(x) (Using raw x input to check inverse Identity)
        x_inv, log_det_inv = model(x, inverse=True)

        # Invertibility check F^-1(F(x))
        x_rec, _ = model(z, inverse=True)

    # A. Check F(x) = x
    diff_id_fwd = (z - x).abs().max().item()
    print(f"\n[Identity F(x) approx x]")
    print(f"Max |F(x) - x|: {diff_id_fwd:.8e}")

    # B. Check F^-1(x) = x
    diff_id_inv = (x_inv - x).abs().max().item()
    print(f"\n[Identity F^-1(x) approx x]")
    print(f"Max |F^-1(x) - x|: {diff_id_inv:.8e}")

    # C. Check LogDet values (Should be 0 for Identity)
    max_ld_fwd = log_det_fwd.abs().max().item()
    max_ld_inv = log_det_inv.abs().max().item()
    print(f"\n[LogDet Zero Check]")
    print(f"Max |LogDet_Fwd|: {max_ld_fwd:.8e}")
    print(f"Max |LogDet_Inv|: {max_ld_inv:.8e}")

    # D. Invertibility Check (Consistency)
    diff_rec = (x_rec - x).abs().max().item()
    print(f"\n[Invertibility F^-1(F(x)) approx x]")
    print(f"Max |F^-1(F(x)) - x|: {diff_rec:.8e}")

    # Final Verdict
    if diff_id_fwd < 1e-4 and diff_id_inv < 1e-4 and max_ld_fwd < 1e-4 and max_ld_inv < 1e-4:
        print("\n>> SUCCESS: Model initialized as Strict Identity.")
    else:
        print("\n>> WARNING: Model NOT initialized as Identity.")