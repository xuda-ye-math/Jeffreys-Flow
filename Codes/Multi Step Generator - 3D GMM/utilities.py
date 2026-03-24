import torch
import numpy as np


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
    Computes Conditional Effective Sample Size Ratio (Step Efficiency).

    Formula:
        CESS = (sum(alpha * w))^2 / (sum(alpha * w^2) * sum(alpha))

    Properties:
        - If w is constant (perfect step), CESS = 1.0 (regardless of alpha quality).
        - If w is degenerate (e.g., 1 particle), CESS -> 0.

    Args:
        alphas (Tensor): [N] Weights of the previous distribution (can be unnormalized).
        weights (Tensor): [N] Incremental importance weights (likelihood ratio).

    Returns:
        Ratio (0.0 to 1.0).
    """
    if alphas.device != weights.device:
        alphas = alphas.to(weights.device)

    # CESS Calculation
    # Num = (E_alpha[w] * sum(alpha))^2 = (sum(alpha * w))^2
    numerator = torch.sum(alphas * weights) ** 2

    # Den = E_alpha[w^2] * sum(alpha) * sum(alpha) = sum(alpha * w^2) * sum(alpha)
    denominator = torch.sum(alphas * (weights ** 2)) * torch.sum(alphas)

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


# ---- 5. Unit Testing ----

if __name__ == "__main__":
    print("--- Testing utilities.py (Metrics Update) ---")

    # ---- Test 1: CESS vs ESS distinction ----
    print("\n[Test 1] CESS Logic Check")
    N = 100

    # Case A: Poor history (alpha), but Perfect Step (w constant)
    # ESS should be low (inherited from alpha), but CESS should be 1.0 (step is perfect)
    alphas_skew = torch.zeros(N)
    alphas_skew[0] = 1.0  # Degenerate history
    w_const = torch.ones(N)  # Perfect step

    cess_val = compute_CESS(alphas_skew, w_const)
    # Total weights = alpha * w = skew * const = skew
    ess_val = compute_ESS(alphas_skew * w_const)

    print(f"  Case A (Skewed History, Perfect Step):")
    print(f"    CESS (Step):  {cess_val:.4f} (Expected 1.0)")
    print(f"    ESS  (Total): {ess_val:.4f} (Expected ~0.01)")

    assert abs(cess_val - 1.0) < 1e-5
    assert abs(ess_val - 1.0 / N) < 1e-5

    # Case B: Uniform history, Noisy Step
    # CESS and ESS should be equal because history is perfect
    alphas_uni = torch.ones(N) / N
    w_skew = torch.zeros(N);
    w_skew[0] = 1.0

    cess_val_b = compute_CESS(alphas_uni, w_skew)
    ess_val_b = compute_ESS(alphas_uni * w_skew)

    print(f"  Case B (Uniform History, Skewed Step):")
    print(f"    CESS (Step):  {cess_val_b:.4f}")
    print(f"    ESS  (Total): {ess_val_b:.4f}")

    assert abs(cess_val_b - ess_val_b) < 1e-5

    print("\n>> All utility tests passed.")