import torch
import numpy as np

# ---- 1. Global Configuration ----
SEED = 29
np.random.seed(SEED)
torch.manual_seed(SEED)


import math

# ---- 2. Base Distribution Utilities ----

def generate_base_samples(pt, n_samples: int, device=None) -> torch.Tensor:
    """
    GENERATE_BASE_SAMPLES
    Generates samples from the Base Distribution (Gaussian) in Fourier space.
    Automatically distributes parameters mapping intrinsic Path_Integral properties.

    Args:
        pt: Path_Integral_Potential object instance.
        n_samples (int): Number of samples to generate.
        device (torch.device, optional): Device to store samples on.

    Returns:
        samples (Tensor): [n_samples, dim] Samples from the base Gaussian.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim = pt.N
    samples = torch.zeros(n_samples, dim, device=device)

    # 1. Mode 0 applies the base scalar scaling
    samples[:, 0] = torch.randn(n_samples, device=device) * (math.sqrt(pt.beta) * pt.SIGMA)

    # 2. Mode k>0 applies intrinsic dynamic modes inverse to OMEGA[k]
    omega = pt.OMEGA
    for k in range(1, dim):
        samples[:, k] = torch.randn(n_samples, device=device) / omega[k]

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

def distribution_rejuvenation(pt,
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
        pt: Path_Integral_Potential instance.
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
        return _mala_core(pt, samples, lam, dt, n_iterations)

    # Batch Processing
    num_batches = int(np.ceil(N / batch_size))
    result_list = []

    for i in range(num_batches):
        st = i * batch_size
        en = min((i + 1) * batch_size, N)
        batch_x = samples[st:en]

        # Process batch
        batch_out = _mala_core(pt, batch_x, lam, dt, n_iterations)
        result_list.append(batch_out)

    return torch.cat(result_list, dim=0)


def _mixed_energy(pt, x, lam):
    return (1.0 - lam) * pt.base_Uf(x) + lam * pt.target_Uf(x)


def _mala_core(pt, samples, lam, dt, n_iterations):
    """ Internal MALA loop for a specific batch explicitly tracking energy gradients. """
    x = samples.clone().detach()
    x.requires_grad_(True)
    N = x.shape[0]
    device = x.device
    sqrt_2dt = math.sqrt(2 * dt)

    # Initial state properties
    u_x = _mixed_energy(pt, x, lam)
    grad_x = torch.autograd.grad(u_x.sum(), x)[0]

    for _ in range(n_iterations):
        with torch.no_grad():
            # 2. Proposal
            noise = torch.randn_like(x)
            mean_y = x - dt * grad_x
            y = mean_y + sqrt_2dt * noise

        y.requires_grad_(True)

        # 3. Proposed State
        u_y = _mixed_energy(pt, y, lam)
        grad_y = torch.autograd.grad(u_y.sum(), y)[0]

        with torch.no_grad():
            # 4. Acceptance Prob
            # q(y|x)
            diff_y_x = y - mean_y
            log_q_y_x = -torch.sum(diff_y_x ** 2, dim=1) / (4 * dt)

            # q(x|y)
            mean_x_from_y = y - dt * grad_y
            diff_x_y = x - mean_x_from_y
            log_q_x_y = -torch.sum(diff_x_y ** 2, dim=1) / (4 * dt)

            # log alpha
            log_alpha = -u_y.detach() + u_x.detach() + log_q_x_y - log_q_y_x

            # Accept/Reject
            accept_log_prob = torch.log(torch.rand(N, device=device))
            accept_mask = accept_log_prob < log_alpha

            # Caching successful coordinates exclusively tracking persistent parameters locally
            x_new = x.detach().clone()
            x_new[accept_mask] = y.detach()[accept_mask]
            
            u_x_new = u_x.detach().clone()
            u_x_new[accept_mask] = u_y.detach()[accept_mask]
            
            grad_x_new = grad_x.detach().clone()
            grad_x_new[accept_mask] = grad_y.detach()[accept_mask]

            x = x_new
            x.requires_grad_(True)
            u_x = u_x_new
            grad_x = grad_x_new

    return x.detach()


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

    from potential import Path_Integral_Potential
    import math

    mock_pt = Path_Integral_Potential(N=8, beta=1.0)
    samples = generate_base_samples(mock_pt, n_samples=100000, device=torch.device('cpu'))

    mean_est = samples.mean(dim=0)
    std_est = samples.std(dim=0)

    target_std_0 = math.sqrt(mock_pt.beta) * mock_pt.SIGMA
    omega = mock_pt.OMEGA
    
    print(f"  Target Mode 0 Std: {target_std_0:.4f}")
    print(f"  Est Mode 0 Std:    {std_est[0]:.4f}")
    
    print(f"  Target Mode 1 Std: {(1.0 / omega[1]):.4f}")
    print(f"  Est Mode 1 Std:    {std_est[1]:.4f}")

    # Basic checks validating base frequencies
    assert torch.abs(std_est[0] - target_std_0) < 0.1
    assert torch.abs(std_est[1] - (1.0 / omega[1])) < 0.1
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