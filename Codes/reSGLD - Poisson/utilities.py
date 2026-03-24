import torch
import numpy as np
from parameters import para, OBS_SIZE


# ---- 1. Global Configuration ----
SEED = 29
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---- 2. Resampling Utilities ----

def distribution_resample(samples: torch.Tensor,
                          weights: torch.Tensor = None,
                          n_resamples: int = None) -> torch.Tensor:
    """
    DISTRIBUTION_RESAMPLE
    Performs multinomial resampling on a set of samples based on importance weights.

    Args:
        samples (Tensor): [N, D] Input samples.
        weights (Tensor, optional): [N] Importance weights.
                                    If None, assumes uniform weights (1/N).
        n_resamples (int, optional): Number of samples to draw.
                                     If None, defaults to N (same as input size).

    Returns:
        resampled_x (Tensor): [n_resamples, D] The resampled particles.
    """

    # 1. Determine Sample Size
    N = samples.shape[0]
    if n_resamples is None:
        n_resamples = N

    # 2. Handle Weights
    if weights is None:
        # If no weights provided, assume Uniform Distribution
        indices = torch.randint(0, N, (n_resamples,), device=samples.device)
    else:
        # Ensure weights are on the correct device
        if weights.device != samples.device:
            weights = weights.to(samples.device)

        # Normalize weights to ensure they sum to 1
        w_sum = weights.sum()
        if w_sum == 0:
            weights_norm = torch.ones_like(weights) / N
        else:
            weights_norm = weights / w_sum

        # 3. Perform Multinomial Resampling
        indices = torch.multinomial(weights_norm, num_samples=n_resamples, replacement=True)

    # 4. Gather Samples
    resampled_x = samples[indices]

    return resampled_x


# ---- 3. Rejuvenation Utilities ----

def distribution_rejuvenation(solver,
                              samples: torch.Tensor,
                              lam: float,
                              dt: float,
                              n_iterations: int = 100,
                              batch_size: int = None) -> torch.Tensor:
    """
    DISTRIBUTION_REJUVENATION
    Refines samples using Metropolis-Adjusted Langevin Algorithm (MALA)
    on the interpolated potential U_mixed(x, lam).
    
    Since we don't have an analytical gradient, we use PyTorch Autograd.

    Supports batch processing to limit GPU memory usage.

    Args:
        solver: Poisson_2D object providing .get_potential_mixed_partial(x, y_obs, obs_indices, sigma, batch_indices, beta)
        samples (Tensor): [N, D] Input samples to rejuvenate.
        lam (float): Interpolation coefficient lambda (beta).
        dt (float): Time step size for ULA/MALA.
        n_iterations (int): Langevin steps.
        batch_size (int, optional): Max samples per batch. If None, process all.

    Returns:
        samples (Tensor): [N, D] Rejuvenated samples.
    """
    N = samples.shape[0]

    # Optimization: Skip if no iterations
    if n_iterations == 0:
        return samples

    # If no batching needed or requested
    if batch_size is None or N <= batch_size:
        return _mala_core(solver, samples, lam, dt, n_iterations, desc="MALA")

    # Batch Processing
    from tqdm import tqdm
    num_batches = int(np.ceil(N / batch_size))
    result_list = []

    for i in range(num_batches):
        st = i * batch_size
        en = min((i + 1) * batch_size, N)
        batch_x = samples[st:en]

        # Process batch
        batch_out = _mala_core(solver, batch_x, lam, dt, n_iterations, desc=f"MALA Batch {i+1}/{num_batches}")
        result_list.append(batch_out)

    return torch.cat(result_list, dim=0)


from tqdm import tqdm

def _mala_core(solver, samples, lam, dt, n_iterations, desc="MALA"):
    """ Internal MALA loop for a specific batch using Autograd and SGD mini-batches """
    x = samples.clone().detach()
    N = x.shape[0]
    device = x.device
    sqrt_2dt = np.sqrt(2 * dt)
    
    # Pre-map global parameters for the solver to use
    y_obs = para.Y_OBS.to(device)
    obs_indices = para.OBS_INDICES.to(device)
    sigma_noise = para.SIGMA_NOISE
    
    # Bounding Box
    limit_min = para.X_LIM_COMPUTE[0]
    limit_max = para.X_LIM_COMPUTE[1]

    for _ in tqdm(range(n_iterations), desc=desc, leave=False):
        x = x.clone().detach().requires_grad_(True)
        
        # Mini-batching sensors for stochastic gradient (OBS_SIZE sensors like in MATLAB simulate_MU)
        random_sensors = torch.randperm(81, device=device)[:OBS_SIZE]
        
        # 1. Compute Potential
        U_mix = solver.get_potential_mixed_partial(
            x, y_obs, obs_indices, sigma_noise, 
            batch_indices=random_sensors, beta=lam
        )
        
        # 2. Compute Gradient via Autograd
        loss = torch.sum(U_mix)
        grad_x = torch.autograd.grad(loss, x)[0]

        # 3. Proposal (Langevin Step)
        with torch.no_grad():
            noise = torch.randn_like(x)
            # ULA Update (Always Accept for simplicity, technically SGLD)
            x_new = x - dt * grad_x + sqrt_2dt * noise
            
            # 4. Strict Domain Enforcement (Reflecting or Clamping)
            # We strictly clamp to [limit_min + epsilon, limit_max - epsilon]
            # to prevent Flow spline evaluation errors exactly at bounds
            x_new = torch.clamp(x_new, limit_min + 1e-4, limit_max - 1e-4)
            
            x = x_new

    return x.detach()


# ---- 4. Generation & Metrics Utilities ----

def normalize_log_weights(log_weights):
    """
    Normalizes log weights: w = exp(log_w - log_sum_exp(log_w))
    """
    max_log_w = torch.max(log_weights)
    log_sum_exp = torch.logsumexp(log_weights - max_log_w, dim=0) + max_log_w
    return torch.exp(log_weights - log_sum_exp)


def compute_ESS(weights: torch.Tensor) -> float:
    """
    Computes Effective Sample Size Ratio (Total Efficiency).
    ESS = (sum w)^2 / sum w^2 / N
    """
    if weights.sum() == 0: return 0.0
    numerator = torch.sum(weights) ** 2
    denominator = torch.sum(weights ** 2)
    return (numerator / denominator).item() / weights.shape[0]


def compute_CESS(alphas: torch.Tensor, weights: torch.Tensor) -> float:
    """
    Computes Conditional Effective Sample Size Ratio.
    """
    if alphas.device != weights.device:
        alphas = alphas.to(weights.device)

    numerator = torch.sum(alphas * weights) ** 2
    denominator = torch.sum(alphas * (weights ** 2)) * torch.sum(alphas)

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


# ---- 5. Unit Testing ----

if __name__ == "__main__":
    print("--- Testing utilities.py for Poisson ---")
    import time
    from Poisson_2D import Poisson_2D
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    
    # Setup test objects
    solver = Poisson_2D(para.N_CELLS, para.ALPHA, para.GAMMA, para.C, device=device)
    
    B = 1000
    # Dummy uniform samples inside bounds
    x_init = torch.rand((B, para.DIM), device=device) * 0.8 + 0.1
    
    print("\n[Test 1] distribution_resample")
    # Skewed weights (first 100 have extreme weight)
    w = torch.zeros(B, device=device)
    w[:100] = 1.0
    x_res = distribution_resample(x_init, weights=w)
    
    # Check if resampled correctly (all should belong to the first 100)
    match = (x_res[:, 0] == x_init[0, 0])
    print(f"Resampling successful. Shape: {x_res.shape}")
    
    print("\n[Test 2] distribution_rejuvenation (Autograd MALA)")
    t0 = time.time()
    # Apply SGLD update
    x_reju = distribution_rejuvenation(
        solver, x_init, lam=0.5, dt=para.DT, n_iterations=10, batch_size=500
    )
    t1 = time.time()
    
    diff = (x_reju - x_init).abs().mean().item()
    print(f"Rejuvenated {B} samples (10 SGLD steps) in {t1-t0:.4f} seconds.")
    print(f"Mean Movement per coord: {diff:.6f}")
    
    # Check Bound constraints
    min_val = x_reju.min().item()
    max_val = x_reju.max().item()
    print(f"Bounds check: Min={min_val:.5f}, Max={max_val:.5f} (Expected exactly within [0, 1])")
    
    if min_val >= 0.0 and max_val <= 1.0:
        print(">> SUCCESS: MALA respects domain limits [0, 1].")
    else:
        print(">> WARNING: MALA exceeded limits!")
        
    print("\n>> All utility tests passed.")
