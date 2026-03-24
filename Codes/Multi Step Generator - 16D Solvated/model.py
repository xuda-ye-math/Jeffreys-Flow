import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

# ---- Import directly from parameters.py and utilities.py ----
from parameters import *
from utilities import *

# =========================================================================
# 1. RQS Helper Functions
# =========================================================================
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3



def _share_rqs_logic(inputs, widths, heights, derivatives, cumwidths, cumheights, inverse):
    """
    Shared algebraic logic for RQS (Forward and Inverse).
    Inputs:
        inputs: [N] (normalized to relative bin offset if fwd, or y_rel if inv?)
        Actually, let's standardize inputs to be the raw scalar inside the bin for standard RQS logic.
    """
    # This helper is kept inline in the main functions for clarity regarding
    # normalization differences between Circular and Euclidean.
    pass

# =========================================================================
# 2. Conditioner (Mixed Input Topology)
# =========================================================================
class MixedConditioner(nn.Module):
    def __init__(self, input_indices, output_dim, hidden_dim, periodic_indices):
        super().__init__()
        self.input_indices = input_indices
        self.periodic_set = set(periodic_indices)
        
        # Determine input dimension: Periodic gets 2 dims (cos, sin), Euclidean gets 1
        self.net_in_dim = 0
        self.mapping = [] # 'p' or 'e'
        
        for idx in input_indices:
            if idx.item() in self.periodic_set:
                self.net_in_dim += 2
                self.mapping.append('p')
            else:
                self.net_in_dim += 1
                self.mapping.append('e')
                
        self.net = nn.Sequential(
            nn.Linear(self.net_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Strict Identity Initialization
        # Zero out final layer weights and biases
        self.net[-1].weight.data.fill_(0.0)
        self.net[-1].bias.data.fill_(0.0)

    def forward(self, x_context):
        # x_context: [Batch, len(input_indices)]
        features = []
        for i, mode in enumerate(self.mapping):
            val = x_context[:, i:i+1]
            if mode == 'p':
                features.append(torch.cos(val))
                features.append(torch.sin(val))
            else:
                features.append(val)
        
        h = torch.cat(features, dim=1)
        return self.net(h)

# =========================================================================
# 3. Mixed Topology Coupling Layer
# =========================================================================
class MixedTopologyCouplingLayer(nn.Module):
    def __init__(self, dim, mask, periodic_indices, num_bins, 
                 periodic_bounds=(-np.pi, np.pi), euclidean_bounds=(-6.0, 6.0),
                 hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', mask)
        
        self.periodic_indices = set(periodic_indices)
        self.num_bins = num_bins
        self.periodic_bounds = periodic_bounds
        self.euclidean_bounds = euclidean_bounds
        
        # Identify indices
        self.transform_indices = torch.nonzero(mask).squeeze()
        self.context_indices = torch.nonzero(1 - mask).squeeze()
        
        # Handle scalar edge cases
        if self.transform_indices.ndim == 0: self.transform_indices = self.transform_indices.unsqueeze(0)
        if self.context_indices.ndim == 0: self.context_indices = self.context_indices.unsqueeze(0)
        
        self.n_transformed = len(self.transform_indices)
        
        # We need 3*K params for Circular, 3*K+1 for Euclidean.
        # To simplify vectorization, we predict 3*K+1 for everyone,
        # and ignore the last derivative for Circular cases.
        self.params_per_dim = 3 * num_bins + 1
        
        self.conditioner = MixedConditioner(
            input_indices=self.context_indices,
            output_dim=self.n_transformed * self.params_per_dim,
            hidden_dim=hidden_dim,
            periodic_indices=periodic_indices
        )
        
        # Initialize bias for derivatives to 1.0 (unnormalized = log(e^(1-eps)-1))
        target_val = 1.0 - DEFAULT_MIN_DERIVATIVE
        self.deriv_bias_val = np.log(np.exp(target_val) - 1)
        
        # Apply bias to the specific slice of the final layer
        # Output shape: [N_trans, 3*K+1]. 
        # Derivatives start at 2*K.
        with torch.no_grad():
            final_bias = self.conditioner.net[-1].bias
            final_bias = final_bias.view(self.n_transformed, self.params_per_dim)
            final_bias[:, 2*self.num_bins:] = float(self.deriv_bias_val)
            self.conditioner.net[-1].bias.copy_(final_bias.view(-1))

    def _circular_rqs(self, inputs, w, h, d, inverse):
        """
        Inputs: inputs [Batch], w/h/d [Batch, K]
        """
        # 1. Normalize Params
        widths = F.softmax(w, dim=-1)
        widths = DEFAULT_MIN_BIN_WIDTH + (1 - DEFAULT_MIN_BIN_WIDTH * self.num_bins) * widths
        
        heights = F.softmax(h, dim=-1)
        heights = DEFAULT_MIN_BIN_HEIGHT + (1 - DEFAULT_MIN_BIN_HEIGHT * self.num_bins) * heights
        
        derivatives = F.softplus(d) + DEFAULT_MIN_DERIVATIVE
        # Circular constraint: d_last = d_first
        d_periodic = torch.cat([derivatives, derivatives[..., :1]], dim=-1)
        
        # 2. Normalize Inputs to [0, 1]
        p_min, p_max = self.periodic_bounds
        domain = p_max - p_min
        
        # Wrap inputs
        inputs_wrapped = torch.remainder(inputs - p_min, domain) + p_min
        inputs_norm = (inputs_wrapped - p_min) / domain
        inputs_norm = torch.clamp(inputs_norm, 0.0, 1.0 - 1e-6)
        
        # 3. Setup Knots
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, (1, 0), mode='constant', value=0.0)
        cumwidths = cumwidths / cumwidths[..., -1:]
        
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, (1, 0), mode='constant', value=0.0)
        cumheights = cumheights / cumheights[..., -1:]
        
        # 4. Transform
        if inverse:
            bin_idx = torch.searchsorted(cumheights, inputs_norm.unsqueeze(-1)).squeeze(-1) - 1
            bin_idx = bin_idx.clamp(0, self.num_bins - 1)
            
            def gather(tensor, idx): return tensor.gather(-1, idx[..., None])[..., 0]
            
            w_b = gather(widths, bin_idx)
            h_b = gather(heights, bin_idx)
            cw_b = gather(cumwidths, bin_idx)
            ch_b = gather(cumheights, bin_idx)
            d_b = gather(d_periodic, bin_idx)
            d_b_plus = gather(d_periodic, bin_idx + 1)
            
            s = h_b / w_b
            theta = inputs_norm - ch_b
            
            a = h_b * (s - d_b) + theta * (d_b + d_b_plus - 2 * s)
            b = h_b * d_b - theta * (d_b + d_b_plus - 2 * s)
            c = -s * theta
            discriminant = b.pow(2) - 4 * a * c
            xi = 2 * c / (-b - torch.sqrt(torch.abs(discriminant)))
            
            outputs_norm = cw_b + xi * w_b
            
            # Derivative
            numer = (h_b ** 2) * (d_b_plus * xi ** 2 + 2 * s * xi * (1 - xi) + d_b * (1 - xi) ** 2)
            denom = (s + (d_b + d_b_plus - 2 * s) * xi * (1 - xi)) ** 2
            deriv = numer / denom
            
            # LogDet Correction: -log(deriv) + 2*log(w) to cancel w^2 scaling
            log_det = -torch.log(deriv) + 2 * torch.log(w_b)
            
        else:
            bin_idx = torch.searchsorted(cumwidths, inputs_norm.unsqueeze(-1)).squeeze(-1) - 1
            bin_idx = bin_idx.clamp(0, self.num_bins - 1)
            
            def gather(tensor, idx): return tensor.gather(-1, idx[..., None])[..., 0]
            
            w_b = gather(widths, bin_idx)
            h_b = gather(heights, bin_idx)
            cw_b = gather(cumwidths, bin_idx)
            ch_b = gather(cumheights, bin_idx)
            d_b = gather(d_periodic, bin_idx)
            d_b_plus = gather(d_periodic, bin_idx + 1)
            
            s = h_b / w_b
            xi = (inputs_norm - cw_b) / w_b
            
            numer = h_b * (s * xi ** 2 + d_b * xi * (1 - xi))
            denom = s + (d_b + d_b_plus - 2 * s) * xi * (1 - xi)
            
            outputs_norm = ch_b + numer / denom
            
            numer_d = (h_b ** 2) * (d_b_plus * xi ** 2 + 2 * s * xi * (1 - xi) + d_b * (1 - xi) ** 2)
            denom_d = denom ** 2
            deriv = numer_d / denom_d
            
            # LogDet Correction: log(deriv) - 2*log(w)
            log_det = torch.log(deriv) - 2 * torch.log(w_b)
            
        outputs = outputs_norm * domain + p_min
        outputs = torch.remainder(outputs - p_min, domain) + p_min
        
        return outputs, log_det

    def _euclidean_rqs(self, inputs, w, h, d, inverse):
        left, right = self.euclidean_bounds
        bottom, top = self.euclidean_bounds
        
        # Consistent Masking: Identity outside the defined bounds
        inside_mask = (inputs >= left) & (inputs <= right)
        outside_mask = ~inside_mask
        
        outputs = torch.zeros_like(inputs)
        log_det = torch.zeros_like(inputs)
        
        # Identity outside
        outputs[outside_mask] = inputs[outside_mask]
        log_det[outside_mask] = 0.0
        
        if not inside_mask.any():
            return outputs, log_det
            
        inputs_in = inputs[inside_mask]
        w_in = w[inside_mask]
        h_in = h[inside_mask]
        d_in = d[inside_mask]
        
        # Normalize Params
        widths = F.softmax(w_in, dim=-1)
        widths = DEFAULT_MIN_BIN_WIDTH + (1 - DEFAULT_MIN_BIN_WIDTH * self.num_bins) * widths
        
        heights = F.softmax(h_in, dim=-1)
        heights = DEFAULT_MIN_BIN_HEIGHT + (1 - DEFAULT_MIN_BIN_HEIGHT * self.num_bins) * heights
        
        derivatives = F.softplus(d_in) + DEFAULT_MIN_DERIVATIVE
        
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, (1, 0), mode='constant', value=0.0)
        cumwidths = cumwidths / cumwidths[..., -1:] 
        
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, (1, 0), mode='constant', value=0.0)
        cumheights = cumheights / cumheights[..., -1:]
        
        if inverse:
            # Inputs are already "z" (which corresponds to "y" in RQS notation)
            # Map from [bottom, top] -> [0, 1]
            inputs_norm = (inputs_in - bottom) / (top - bottom)
            
            # Use standard torch.searchsorted
            bin_idx = torch.searchsorted(cumheights, inputs_norm.unsqueeze(-1)).squeeze(-1) - 1
            bin_idx = bin_idx.clamp(0, self.num_bins - 1)
            
            def gather(tensor, idx): return tensor.gather(-1, idx[..., None])[..., 0]
            
            w_b = gather(widths, bin_idx)
            h_b = gather(heights, bin_idx)
            cw_b = gather(cumwidths, bin_idx)
            ch_b = gather(cumheights, bin_idx)
            d_b = gather(derivatives, bin_idx)
            d_b_plus = gather(derivatives, bin_idx + 1)
            
            s = h_b / w_b
            theta = inputs_norm - ch_b
            
            a = h_b * (s - d_b) + theta * (d_b + d_b_plus - 2 * s)
            b = h_b * d_b - theta * (d_b + d_b_plus - 2 * s)
            c = -s * theta
            discriminant = b.pow(2) - 4 * a * c
            xi = 2 * c / (-b - torch.sqrt(torch.abs(discriminant)))
            
            outputs_norm = cw_b + xi * w_b
            outputs[inside_mask] = outputs_norm * (right - left) + left
            
            numer = (h_b ** 2) * (d_b_plus * xi ** 2 + 2 * s * xi * (1 - xi) + d_b * (1 - xi) ** 2)
            denom = (s + (d_b + d_b_plus - 2 * s) * xi * (1 - xi)) ** 2
            deriv = numer / denom
            log_det[inside_mask] = -(torch.log(deriv) - 2 * torch.log(w_b))
            
        else:
            inputs_norm = (inputs_in - left) / (right - left)
            
            # Use standard torch.searchsorted
            bin_idx = torch.searchsorted(cumwidths, inputs_norm.unsqueeze(-1)).squeeze(-1) - 1
            bin_idx = bin_idx.clamp(0, self.num_bins - 1)
            
            def gather(tensor, idx): return tensor.gather(-1, idx[..., None])[..., 0]
            
            w_b = gather(widths, bin_idx)
            h_b = gather(heights, bin_idx)
            cw_b = gather(cumwidths, bin_idx)
            ch_b = gather(cumheights, bin_idx)
            d_b = gather(derivatives, bin_idx)
            d_b_plus = gather(derivatives, bin_idx + 1)
            
            s = h_b / w_b
            xi = (inputs_norm - cw_b) / w_b
            
            numer = h_b * (s * xi ** 2 + d_b * xi * (1 - xi))
            denom = s + (d_b + d_b_plus - 2 * s) * xi * (1 - xi)
            
            # Determine output in [0, 1] then scale
            outputs_norm = ch_b + numer / denom
            
            outputs[inside_mask] = outputs_norm * (top - bottom) + bottom
            
            numer_d = (h_b ** 2) * (d_b_plus * xi ** 2 + 2 * s * xi * (1 - xi) + d_b * (1 - xi) ** 2)
            denom_d = denom ** 2
            deriv = numer_d / denom_d
            log_det[inside_mask] = torch.log(deriv) - 2 * torch.log(w_b)

        return outputs, log_det

    def forward(self, x, inverse=False):
        x_ctx = x[:, self.context_indices]
        x_tr = x[:, self.transform_indices]
        
        params = self.conditioner(x_ctx)
        params = params.view(x.shape[0], self.n_transformed, -1)
        
        u_w = params[..., :self.num_bins]
        u_h = params[..., self.num_bins:2*self.num_bins]
        u_d = params[..., 2*self.num_bins:]
        
        z_tr = torch.zeros_like(x_tr)
        total_log_det = torch.zeros(x.shape[0], device=x.device)
        
        for i, global_idx_tensor in enumerate(self.transform_indices):
            global_idx = global_idx_tensor.item()
            is_periodic = global_idx in self.periodic_indices
            
            curr_x = x_tr[:, i]
            curr_w = u_w[:, i]
            curr_h = u_h[:, i]
            
            if is_periodic:
                curr_d = u_d[:, i, :self.num_bins] 
                y, ld = self._circular_rqs(curr_x, curr_w, curr_h, curr_d, inverse)
            else:
                curr_d = u_d[:, i]
                y, ld = self._euclidean_rqs(curr_x, curr_w, curr_h, curr_d, inverse)
            
            z_tr[:, i] = y
            total_log_det += ld
            
        z = x.clone()
        z[:, self.transform_indices] = z_tr
        return z, total_log_det

# =========================================================================
# 4. Normalizing Flow Model
# =========================================================================
class Normalizing_Flow(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dim = para.DIM
        self.periodic_indices = [0, 1]
        
        # Load Limits
        if hasattr(para, 'X_PERI_LIM'):
            p_lim = para.X_PERI_LIM
        else:
            p_lim = (-np.pi, np.pi)
            
        # Strict check for X_FREE_LIM_COMP or fallback to user requested [-12, 12]
        if hasattr(para, 'X_FREE_LIM_COMP'):
            e_lim = para.X_FREE_LIM_COMP
        else:
            e_lim = (-12.0, 12.0) # Updated default per user request

        print(f"[Model] Initializing Flow.")
        print(f"        Periodic Bounds: {p_lim}")
        print(f"        Euclidean Bounds: {e_lim}") # Verify this prints [-12, 12]

        self.layers = nn.ModuleList()
        
        for i in range(NUM_LAYERS):
            mask = torch.zeros(self.dim)
            if i % 2 == 0:
                mask[0::2] = 1.0 
            else:
                mask[1::2] = 1.0 
            
            self.layers.append(MixedTopologyCouplingLayer(
                dim=self.dim,
                mask=mask,
                periodic_indices=self.periodic_indices,
                num_bins=NUM_BINS,
                hidden_dim=HIDDEN_DIM,
                periodic_bounds=p_lim,
                euclidean_bounds=e_lim
            ))

    def forward(self, x, inverse=False):
        log_det_sum = 0
        z = x
        if not inverse:
            for layer in self.layers:
                z, ld = layer(z, inverse=False)
                log_det_sum += ld
        else:
            for layer in reversed(self.layers):
                z, ld = layer(z, inverse=True)
                log_det_sum += ld
        
        z[:, :2] = torch.remainder(z[:, :2] - (-np.pi), 2*np.pi) + (-np.pi)
        return z, log_det_sum

# =========================================================================
# 5. Verification
# =========================================================================
if __name__ == "__main__":
    from parameters import *
    from utilities import *
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(para, 'to'): para.to(device)
        
    print(f"--- Verifying Mixed Topology Flow (Dim={para.DIM}) on {device} ---")
    
    model = Normalizing_Flow().to(device)
    model.eval()
    
    x_test = generate_mixed_base_samples(para, n_samples=1000, device=device)
    
    # --- Test 1: Identity ---
    print("\n[Test 1] Identity Initialization & Invertibility")
    with torch.no_grad():
        z, log_det_fwd = model(x_test, inverse=False)
        x_rec, log_det_inv = model(z, inverse=True)
        
        diff_fwd = torch.abs(z - x_test)
        diff_fwd[:, :2] = torch.minimum(diff_fwd[:, :2], 2*np.pi - diff_fwd[:, :2])
        
        diff_rec = torch.abs(x_rec - x_test)
        diff_rec[:, :2] = torch.minimum(diff_rec[:, :2], 2*np.pi - diff_rec[:, :2])
        
        # LogDet Consistency Check
        ld_diff = torch.abs(log_det_fwd + log_det_inv).max().item()

    print(f"  Max |F(x) - x|       : {diff_fwd.max().item():.4e}")
    print(f"  Max |LogDet|         : {log_det_fwd.abs().max().item():.4e}")
    print(f"  Max |LogDet_F + LogDet_I| : {ld_diff:.4e}")
    print(f"  Max |F^-1(F(x)) - x| : {diff_rec.max().item():.4e}")
    
    if diff_fwd.max().item() < 1e-4 and diff_rec.max().item() < 1e-4:
        print(">> SUCCESS: Model initialized as Strict Identity.")
    else:
        print(">> FAIL: Model NOT initialized as Identity.")

    # --- Test 2: Randomized Parameters ---
    print("\n[Test 2] Randomized Parameters & Invertibility Check")
    print("  > Perturbing model parameters with random noise...")
    
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.01)

    with torch.no_grad():
        z_rand, log_det_rand_fwd = model(x_test, inverse=False)
        x_rec_rand, log_det_rand_inv = model(z_rand, inverse=True)
        
        # 1. Forward Drift
        diff_rand = torch.abs(z_rand - x_test)
        diff_rand[:, :2] = torch.minimum(diff_rand[:, :2], 2*np.pi - diff_rand[:, :2])
        
        # 2. Invertibility
        diff_rec_rand = torch.abs(x_rec_rand - x_test)
        diff_rec_rand[:, :2] = torch.minimum(diff_rec_rand[:, :2], 2*np.pi - diff_rec_rand[:, :2])
        
        # 3. LogDet Consistency Check (Crucial for Debug)
        # Should be near zero (LogDet_Inv = -LogDet_Fwd)
        ld_check = torch.abs(log_det_rand_fwd + log_det_rand_inv)
        max_ld_diff = ld_check.max().item()

        # 4. KL Divergence (E_KL and D_KL)
        # We use para.compute_potential_base for both source and target since we are testing
        # diffeomorphism of the base distribution itself.
        # Ensure U(x) contribution is included as requested.
        
        u_base = para.compute_potential_base
        
        # E_KL (Forward): KL(q || p)
        # log(q/p) = U_target(z) - U_source(x) - log_det
        log_ratios_fwd = u_base(z_rand) - u_base(x_test) - log_det_rand_fwd
        e_kl = torch.mean(log_ratios_fwd).item()
        
        # D_KL (Reverse): KL(p || q)
        # log(p/q) = U_source(x_rec) - U_target(z) - log_det_inv
        # We need to treat x_test as samples from the Target distribution (p_base)
        # and push them back through the inverse flow to estimate q_model.
        x_inv, log_det_inv_check = model(x_test, inverse=True)
        
        log_ratios_bwd = u_base(x_inv) - u_base(x_test) - log_det_inv_check
        d_kl = torch.mean(log_ratios_bwd).item()

    print(f"  Max |F(x) - x|       : {diff_rand.max().item():.4e} (Expected: Large)")
    print(f"  Max |F(x) - x|       : {diff_rand.max().item():.4e} (Expected: Large)")
    print(f"  Max |LogDet Fwd|     : {log_det_rand_fwd.abs().max().item():.4e}")
    print(f"  Max |LogDet_F + LogDet_I| : {max_ld_diff:.4e} (Should be ~0)")
    print(f"  Max |F^-1(F(x)) - x| : {diff_rec_rand.max().item():.4e} (Expected: Small)")
    print(f"  E_KL (Forward)       : {e_kl:.4e} (Should be >= 0)")
    print(f"  D_KL (Reverse)       : {d_kl:.4e} (Should be >= 0)")
    
    if diff_rand.max().item() > 1e-2 and diff_rec_rand.max().item() < 1e-4 and max_ld_diff < 1e-3:
        print(">> SUCCESS: Model transforms data non-trivially, is invertible, and preserves volume consistently.")
    else:
        print(">> FAIL: Check failed.")

    # --- Test 3: Identity Training Loop (Stability Check) ---
    print("\n[Test 3] Identity Training Loop (Stability Check)")
    print("  > Training Flow to map Base -> Base (Should converge to Identity)")
    
    # 1. Setup Data
    N_TRAIN = 200000
    mu_source = generate_mixed_base_samples(para, n_samples=N_TRAIN, device=device)
    mu_target = generate_mixed_base_samples(para, n_samples=N_TRAIN, device=device)
    
    # 2. Setup Hyperparameters (THETA = 0.5)
    THETA_TEST = 0.5
    
    # Calculate Variances (Bath dims only for scaling, similar to train_flow.py)
    # Note: Base distribution has unit variance in bath, so this should be ~14 (if dims=16)
    var_0 = torch.var(mu_source[:, 2:], dim=0).sum().item()
    var_1 = torch.var(mu_target[:, 2:], dim=0).sum().item()
    
    lambda_0 = THETA_TEST * var_0
    lambda_1 = (1.0 - THETA_TEST) * var_1
    
    print(f"  > Lambda_0: {lambda_0:.4f} | Lambda_1: {lambda_1:.4f}")
    
    # 3. Setup Model & Optimizer
    # Re-initialize model to be fresh (or close to identity)
    model_train = Normalizing_Flow().to(device)
    optimizer = optim.Adam(model_train.parameters(), lr=para.LR if hasattr(para, 'LR') else 1e-4) # Use para.LR or default
    
    BATCH_SIZE_TEST = 10000
    EPOCHS_TEST = 5  # Run for a few epochs to see trend
    ALPHA_TEST = 1.5 # Use Renyi Divergence (Alpha=1.5) to match train_flow.py stability
    
    model_train.train()
    u_base = para.compute_potential_base
    
    for epoch in range(EPOCHS_TEST):
        # [Crucial] Generate FRESH samples every epoch to prevent RQS overfitting finite samples
        # This acts as an "Infinite Data" stream, verifying true distribution stability.
        mu_source = generate_mixed_base_samples(para, n_samples=N_TRAIN, device=device)
        mu_target = generate_mixed_base_samples(para, n_samples=N_TRAIN, device=device)
        
        perm_0 = torch.randperm(N_TRAIN)
        perm_1 = torch.randperm(N_TRAIN)
        
        total_loss = 0.0
        total_e_kl = 0.0
        total_d_kl = 0.0
        total_pot_diff = 0.0
        total_log_det = 0.0

        num_batches = N_TRAIN // BATCH_SIZE_TEST
        
        for i in range(num_batches):
            idx_0 = perm_0[i * BATCH_SIZE_TEST : (i + 1) * BATCH_SIZE_TEST]
            idx_1 = perm_1[i * BATCH_SIZE_TEST : (i + 1) * BATCH_SIZE_TEST]
            
            x_batch_0 = mu_source[idx_0] # Source samples
            x_batch_1 = mu_target[idx_1] # Target samples
            
            optimizer.zero_grad()
            
            # --- Energy KL (Forward) -> Renyi Divergence ---
            z_fwd, log_det_fwd = model_train(x_batch_0, inverse=False)
            
            pot_diff = u_base(z_fwd) - u_base(x_batch_0)
            log_ratios_fwd = pot_diff - log_det_fwd
            
            if ALPHA_TEST == 1.0:
                loss_e_kl = torch.mean(log_ratios_fwd)
            else:
                # LogSumExp trick: (log_sum_exp_fwd - log N) / (alpha - 1)
                log_sum_exp_fwd = torch.logsumexp((ALPHA_TEST - 1) * log_ratios_fwd, dim=0)
                loss_e_kl = (log_sum_exp_fwd - np.log(log_ratios_fwd.size(0))) / (ALPHA_TEST - 1)
            
            # --- Data KL (Reverse) -> Renyi Divergence ---
            x_inv, log_det_inv = model_train(x_batch_1, inverse=True)
            
            log_ratios_bwd = u_base(x_inv) - u_base(x_batch_1) - log_det_inv
            
            if ALPHA_TEST == 1.0:
                loss_d_kl = torch.mean(log_ratios_bwd)
            else:
                log_sum_exp_bwd = torch.logsumexp((ALPHA_TEST - 1) * log_ratios_bwd, dim=0)
                loss_d_kl = (log_sum_exp_bwd - np.log(log_ratios_bwd.size(0))) / (ALPHA_TEST - 1)
            
            # Total Loss
            loss = lambda_0 * loss_e_kl + lambda_1 * loss_d_kl
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_e_kl += loss_e_kl.item()
            total_d_kl += loss_d_kl.item()
            total_pot_diff += torch.mean(pot_diff).item()
            total_log_det += torch.mean(log_det_fwd).item()
            
        avg_loss = total_loss / num_batches
        avg_e = total_e_kl / num_batches
        avg_d = total_d_kl / num_batches
        avg_pot = total_pot_diff / num_batches
        avg_ld = total_log_det / num_batches
        
        print(f"  [Epoch {epoch+1}/{EPOCHS_TEST}] Loss: {avg_loss:.4f}")
        print(f"       Renyi_E (alpha={ALPHA_TEST}): {avg_e:.4e} | Renyi_D: {avg_d:.4e}")
        print(f"       Pot Diff: {avg_pot:.4e} | LogDet: {avg_ld:.4e}")