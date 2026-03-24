# ---- 1. Header and Imports ----
from parameters import *  # Imports para, DATA_DIR, etc.
from model import Normalizing_Flow
import torch
import numpy as np
import os
import scipy.io
import math

# Seed
SEED = 30
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")

# Sync Potential Helper with current device
para.device = str(device)
print(f"[Config] Potential System: {para.NAME} ({para.ABBR})")


# ---- 2. Helper Functions ----

def base_potential(x):
    """
    Computes U_0(x) for the Base Distribution.
    For Periodic Well, the base is Uniform on [-pi, pi]^d.

    p(x) = 1 / Volume = 1 / (2*pi)^d
    U_0(x) = -log(p(x)) = log((2*pi)^d) = d * log(2*pi)
    """
    dim = para.DIM
    # U_0 = d * log(2*pi)
    u0_val = dim * math.log(2 * math.pi)

    # Return tensor of shape [Batch_Size]
    return torch.full((x.shape[0],), u0_val, device=x.device)


def normalize_log_weights(log_weights):
    """
    Normalizes log weights: w = exp(log_w - log_sum_exp(log_w))
    """
    max_log_w = torch.max(log_weights)
    # log(sum(exp(log_w - max))) + max
    log_sum_exp = torch.logsumexp(log_weights - max_log_w, dim=0) + max_log_w

    norm_log_weights = log_weights - log_sum_exp
    norm_weights = torch.exp(norm_log_weights)

    return norm_weights


def generate_for_size(exponent, model):
    """
    Generates samples for a specific size N = 4^exponent.

    Args:
        exponent (int): Power of 4 (e.g., 8, 9, 10...)
        model: Loaded PyTorch model
    """
    sample_size = 4 ** exponent

    print(f"\n" + "-" * 60)
    print(f"  PROCESSING SIZE: 4^{exponent} = {sample_size}")
    print("-" * 60)

    # 1. Generate Base Samples (NU_0) - Uniform on [-pi, pi]
    # x ~ Uniform(-pi, pi)
    # torch.rand gives [0, 1] -> scale to [-pi, pi]
    print(f"  [Step 1] Generating Uniform Base Samples...")

    # We generate on the fly to save memory, or strictly following batching
    # But to compute ESS accurately over the full ensemble, we need to accumulate weights.

    # 2. Push Forward to Target (NU_1) and Compute Weights
    print(f"  [Step 2] Pushing samples through flow...")

    # Process in batches to avoid OOM
    batch_size = 5000
    num_batches = int(np.ceil(sample_size / batch_size))

    nu_1_list = []
    weights_list = []

    with torch.no_grad():
        for i in range(num_batches):
            # Generate batch of NU_0 on the fly
            current_batch_size = min(batch_size, sample_size - i * batch_size)

            # Uniform Sampling [-pi, pi]
            batch_x = (torch.rand(current_batch_size, para.DIM, device=device) * 2 * math.pi) - math.pi

            # Forward Pass: x -> y
            # returns y, log|det J_F|
            batch_y, log_det = model(batch_x, inverse=False)

            # Compute Potentials
            # U_0(x): Uniform Base (Constant)
            u0 = base_potential(batch_x)

            # U_1(y): Target Potential
            u1 = para.compute_potential(batch_y)

            # Compute Log Importance Weights
            # w(y) = pi_1(y) / q(y)
            # log w = -U_1(y) + U_0(x) + log|det|
            log_w = -u1 + u0 + log_det

            nu_1_list.append(batch_y)
            weights_list.append(log_w)

    # Concatenate results
    nu_1 = torch.cat(nu_1_list, dim=0)
    log_weights_all = torch.cat(weights_list, dim=0)

    # Normalize weights
    final_weights = normalize_log_weights(log_weights_all)

    # Check normalization and ESS
    weight_sum = final_weights.sum().item()
    ess = 1.0 / (final_weights ** 2).sum().item()

    print(f"       Weights Sum: {weight_sum:.6f}")
    print(f"       Effective Sample Size (ESS): {ess:.2f} / {sample_size} ({ess / sample_size:.2%})")

    # 3. Save Outputs (MAT format)
    # Suffix is the exponent (e.g., _8, _9)
    # Samples: {ABBR}_samples_{exponent}.mat
    path_samples = os.path.join(DATA_DIR, f'{para.ABBR}_samples_{exponent}.mat')
    scipy.io.savemat(path_samples, {f'samples_{exponent}': nu_1.cpu().numpy()})
    print(f"  [Step 3] Saved Samples to: {path_samples}")

    # Weights: {ABBR}_weights_{exponent}.mat
    path_weights = os.path.join(DATA_DIR, f'{para.ABBR}_weights_{exponent}.mat')
    scipy.io.savemat(path_weights, {f'weights_{exponent}': final_weights.cpu().numpy()})
    print(f"           Saved Weights to: {path_weights}")


# ---- 3. Main Generation Execution ----

if __name__ == "__main__":
    print(f"--- Generating NU Samples (Convergence Test) ---")
    print(f"System: {para.NAME}")

    # 1. Load Trained Flow Model (Single Model)
    # Filename format: {ABBR}_flow.pth (from previous update)
    model_filename = f'{para.ABBR}_flow.pth'
    flow_path = os.path.join(DATA_DIR, model_filename)

    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Model file not found: {flow_path}. Please train first.")

    print(f"\n[Initialization] Loading Model: {model_filename}")
    model = Normalizing_Flow().to(device)
    model.load_state_dict(torch.load(flow_path, map_location=device))
    model.eval()

    # 2. Iterate through exponents 8 to 12
    # Sample sizes: 4^8 ... 4^12
    exponents = [8, 9, 10, 11, 12]

    for k in exponents:
        generate_for_size(k, model)

    print(f"\n[Main] All generation tasks complete.")