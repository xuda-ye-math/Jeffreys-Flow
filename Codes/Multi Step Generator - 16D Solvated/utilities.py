import torch
import numpy as np

# ---- 1. Global Configuration ----
SEED = 29
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---- 2. Base Distribution Utilities ----

def generate_mixed_base_samples(para, n_samples: int, device=None) -> torch.Tensor:
    """
    GENERATE_MIXED_BASE_SAMPLES
    Generates samples from the Solvated 16D Base Distribution.
    
    Structure:
      Dims 0-1 (Periodic): Uniform on [-pi, pi]
      Dims 2-15 (Bath):    Gaussian(MEAN, SIGMA)

    Args:
        para: Potential object. Must contain:
              - MEAN, SIGMA (for Bath)
              - X_PERI_LIM (for Periodic)
        n_samples (int): Number of samples to generate.
        device (torch.device, optional): Device to store samples on.

    Returns:
        samples (Tensor): [n_samples, dim]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim = para.DIM
    samples = torch.zeros(n_samples, dim, device=device)
    
    # 1. Periodic Dimensions (0, 1): Uniform [-pi, pi]
    # Check if para has X_PERI_LIM, else default
    if hasattr(para, 'X_PERI_LIM'):
        low, high = para.X_PERI_LIM
    else:
        low, high = -np.pi, np.pi
        
    width = high - low
    samples[:, 0:2] = torch.rand(n_samples, 2, device=device) * width + low
    
    # 2. Bath Dimensions (2 to end): Gaussian
    # Retrieve Parameters
    if hasattr(para, 'MEAN') and hasattr(para, 'SIGMA'):
        mu = para.MEAN
        sigma = para.SIGMA
    else:
        # Fallback defaults
        mu = 0.0
        sigma = 1.0
        
    # Generate Gaussian Noise for Bath
    dim_bath = dim - 2
    if dim_bath > 0:
        epsilon = torch.randn(n_samples, dim_bath, device=device)
        
        # Move params to device if they are tensors
        if isinstance(mu, torch.Tensor) and mu.device != device: mu = mu.to(device)
        if isinstance(sigma, torch.Tensor) and sigma.device != device: sigma = sigma.to(device)
            
        samples[:, 2:] = mu + sigma * epsilon

    return samples


# ---- 3. Resampling Utilities ----

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


# ---- 4. Rejuvenation Utilities ----

def distribution_rejuvenation(para,
                              samples: torch.Tensor,
                              lam: float,
                              dt: float,
                              n_iterations: int = 500,
                              batch_size: int = None) -> torch.Tensor:
    """
    DISTRIBUTION_REJUVENATION
    Refines samples using Metropolis-Adjusted Langevin Algorithm (MALA)
    on the interpolated potential U_mixed(x, lam).

    Supports batch processing to limit GPU memory usage.

    Args:
        para: Potential object providing .compute_potential_mixed(x, lam)
              and .compute_gradient_mixed(x, lam).
        samples (Tensor): [N, D] Input samples to rejuvenate.
        lam (float): Interpolation coefficient lambda.
        dt (float): Time step size.
        n_iterations (int): MCMC steps.
        batch_size (int, optional): Max samples per batch. If None, process all.

    Returns:
        samples (Tensor): [N, D] Rejuvenated samples.
    """
    N = samples.shape[0]

    # If no batching needed or requested
    if batch_size is None or N <= batch_size:
        return _mala_core(para, samples, lam, dt, n_iterations)

    # Batch Processing
    # print(f"    [Rejuvenation] Batching {N} samples with size {batch_size}...")
    num_batches = int(np.ceil(N / batch_size))
    result_list = []

    for i in range(num_batches):
        st = i * batch_size
        en = min((i + 1) * batch_size, N)
        batch_x = samples[st:en]

        # Process batch
        batch_out = _mala_core(para, batch_x, lam, dt, n_iterations)
        result_list.append(batch_out)

    return torch.cat(result_list, dim=0)


def _mala_core(para, samples, lam, dt, n_iterations):
    """ Internal MALA loop for a specific batch """
    x = samples.clone()
    N = x.shape[0]
    device = x.device
    sqrt_2dt = np.sqrt(2 * dt)

    for _ in range(n_iterations):
        with torch.no_grad():
            # 1. Current State
            u_x = para.compute_potential_mixed(x, lam)
            grad_x = para.compute_gradient_mixed(x, lam)

            # 2. Proposal
            noise = torch.randn_like(x)
            mean_y = x - dt * grad_x
            y = mean_y + sqrt_2dt * noise

            # 3. Proposed State
            u_y = para.compute_potential_mixed(y, lam)
            grad_y = para.compute_gradient_mixed(y, lam)

            # 4. Acceptance Prob
            # q(y|x)
            diff_y_x = y - mean_y
            log_q_y_x = -torch.sum(diff_y_x ** 2, dim=1) / (4 * dt)

            # q(x|y)
            mean_x_from_y = y - dt * grad_y
            diff_x_y = x - mean_x_from_y
            log_q_x_y = -torch.sum(diff_x_y ** 2, dim=1) / (4 * dt)

            # log alpha
            log_alpha = -u_y + u_x + log_q_x_y - log_q_y_x

            # Accept/Reject
            accept_log_prob = torch.log(torch.rand(N, device=device))
            accept_mask = accept_log_prob < log_alpha

            x[accept_mask] = y[accept_mask]

    return x


# ---- 5. Generation & Metrics Utilities ----

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
    Computes Conditional Effective Sample Size Ratio (Step Efficiency).

    Uses Log-Sum-Exp trick for numerical stability to avoid overflow when weights are large.
    CESS = (E[w])^2 / E[w^2] where expectation is over alpha.

    Formula:
      log(CESS) = 2*log(sum(alpha*w)) - log(sum(alpha*w^2))

    Args:
        alphas (Tensor): [N] Log-weights of the previous distribution (log_alphas).
                         Normally uniform (-log N), but function accepts any log-distribution.
        weights (Tensor): [N] Incremental Log-Weights (log_w).

    Returns:
        Ratio (0.0 to 1.0).
    """
    if alphas.device != weights.device:
        alphas = alphas.to(weights.device)

    # Let A = log_alphas, W = log_weights
    # Term 1: log(sum(exp(A + W)))
    term1 = torch.logsumexp(alphas + weights, dim=0)

    # Term 2: log(sum(exp(A + 2*W)))
    term2 = torch.logsumexp(alphas + 2 * weights, dim=0)

    # log(CESS) = 2 * Term1 - Term2
    log_cess = 2 * term1 - term2

    return torch.exp(log_cess).item()


# ---- 6. Unit Testing ----

if __name__ == "__main__":
    print("--- Testing utilities.py (Base Dist & Metrics Update) ---")

    # ---- Test 1: Base Distribution Generation (Mock) ----
    print("\n[Test 1] Base Distribution Generation (Mock)")


    class MockPara:
        DIM = 2
        # Anisotropic parameters
        MEAN = torch.tensor([10.0, -5.0])
        SIGMA = torch.tensor([0.1, 2.0])


    mock_para = MockPara()
    
    # MOCK adjustment for Mixed Base
    mock_para.X_PERI_LIM = [-np.pi, np.pi]
    mock_para.MEAN = 0.0 # Scalar mean for bath
    mock_para.SIGMA = 1.0
    mock_para.DIM = 16
    
    samples = generate_mixed_base_samples(mock_para, n_samples=50000, device=torch.device('cpu'))

    print(f"  Sample Shape: {samples.shape}")

    # Check Periodic (Dims 0-1) -> Uniform [-pi, pi]
    # Mean should be ~0, Std should be width/sqrt(12) = 2pi/sqrt(12) = pi/sqrt(3) ~ 1.8138
    mean_peri = samples[:, 0:2].mean()
    std_peri = samples[:, 0:2].std()
    expected_std_peri = np.pi / np.sqrt(3)
    
    print(f"  [Periodic] Mean: {mean_peri:.4f} (Exp: 0.0)")
    print(f"  [Periodic] Std:  {std_peri:.4f} (Exp: {expected_std_peri:.4f})")
    
    # Check Bath (Dims 2+) -> Gaussian(0,1)
    mean_bath = samples[:, 2:].mean()
    std_bath = samples[:, 2:].std()
    
    print(f"  [Bath]     Mean: {mean_bath:.4f} (Exp: 0.0)")
    print(f"  [Bath]     Std:  {std_bath:.4f} (Exp: 1.0)")

    # Basic checks
    assert abs(mean_peri) < 0.1
    assert abs(std_peri - expected_std_peri) < 0.1
    assert abs(mean_bath) < 0.1
    assert abs(std_bath - 1.0) < 0.1

    # Basic checks
    print("  >> Base Sample Generation Passed.")

    # ---- Test 2: CESS Log Stability ----
    print("\n[Test 2] CESS Log Stability")
    N = 1000
    # Simulate Log Inputs
    log_alphas = torch.zeros(N) - np.log(N)  # Uniform log weights
    log_weights = torch.randn(N) + 50.0  # Large log weights

    cess_val = compute_CESS(log_alphas, log_weights)

    print(f"  Stable CESS: {cess_val:.4f}")
    assert not np.isnan(cess_val)
    assert 0.0 <= cess_val <= 1.0

    print("\n>> All utility tests passed.")