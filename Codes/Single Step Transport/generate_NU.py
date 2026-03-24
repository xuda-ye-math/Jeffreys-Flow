# ---- 1. Header and Imports ----
from parameters import *  # Imports para, NU_SIZE, DATA_DIR, etc.
from model import Normalizing_Flow
import torch
import numpy as np
import os
import scipy.io

# Seed
SEED = 30
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")

# Sync Potential Helper with current device
# 'para' is initialized in parameters.py based on ABBR
para.device = str(device)
print(f"[Config] Potential System: {para.NAME} ({para.ABBR})")


# ---- 2. Helper Functions ----

def gaussian_potential(x):
    """
    Computes U_0(x) for the Gaussian Base Distribution.
    U_0(x) = 0.5 * ||(x - mu)/sigma||^2

    Uses global 'para' for Mean and Sigma.
    """
    mean = para.BASE_MEAN
    sigma = para.BASE_SIGMA
    return 0.5 * torch.sum(((x - mean) / sigma) ** 2, dim=1)


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


def generate_for_index(index, theta, nu_0):
    """
    Loads a specific model by index, generates samples, computes weights, and saves to .mat.
    """
    print(f"\n" + "-" * 60)
    print(f"  PROCESSING INDEX: {index} (Theta={theta:.2f})")
    print("-" * 60)

    # 1. Load Trained Flow Model
    # Filename format: {ABBR}_flow_{index}.pth
    model_filename = f'{para.ABBR}_flow_{index}.pth'
    flow_path = os.path.join(DATA_DIR, model_filename)

    if not os.path.exists(flow_path):
        print(f"  [Warning] Model file not found: {flow_path}. Skipping.")
        return

    print(f"  [Step 1] Loading Model: {model_filename}")
    model = Normalizing_Flow().to(device)
    model.load_state_dict(torch.load(flow_path, map_location=device))
    model.eval()

    # 2. Push Forward to Target (NU_1) and Compute Weights
    print(f"  [Step 2] Pushing {NU_SIZE} samples through flow...")

    # Process in batches to avoid OOM
    batch_size = 5000
    num_batches = int(np.ceil(NU_SIZE / batch_size))

    nu_1_list = []
    weights_list = []

    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, NU_SIZE)
            batch_x = nu_0[start:end]

            # Forward Pass: x -> y
            # returns y, log|det J_F|
            batch_y, log_det = model(batch_x, inverse=False)

            # Compute Potentials
            # U_0(x): Gaussian (Base)
            u0 = gaussian_potential(batch_x)

            # U_1(y): Target Potential (Generic)
            u1 = para.compute_potential(batch_y)

            # Compute Log Importance Weights
            # w(y) = pi_1(y) / q(y)
            # log w = -U_1(y) + U_0(x) + log|det|
            # Note: log_det from model(inverse=False) is log|det dy/dx|
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
    print(f"       Effective Sample Size (ESS): {ess:.2f} / {NU_SIZE} ({ess / NU_SIZE:.2%})")

    # 3. Save Outputs (MAT format for MATLAB compatibility)
    # Samples: {ABBR}_samples_{index}.mat
    # Key: samples_{index}
    path_samples = os.path.join(DATA_DIR, f'{para.ABBR}_samples_{index}.mat')
    scipy.io.savemat(path_samples, {f'samples_{index}': nu_1.cpu().numpy()})
    print(f"  [Step 3] Saved Samples to: {path_samples}")

    # Weights: {ABBR}_weights_{index}.mat
    # Key: weights_{index}
    path_weights = os.path.join(DATA_DIR, f'{para.ABBR}_weights_{index}.mat')
    scipy.io.savemat(path_weights, {f'weights_{index}': final_weights.cpu().numpy()})
    print(f"           Saved Weights to: {path_weights}")


# ---- 3. Main Generation Execution ----

if __name__ == "__main__":
    print(f"--- Generating NU Samples using Trained Flows ---")
    print(f"Target Size: {NU_SIZE}")
    print(f"System: {para.NAME}")

    # 1. Generate Base Samples (NU_0) - Common for all strategies
    print(f"\n[Initialization] Generating Independent Gaussian Samples (NU_0)...")
    # x ~ N(mean, sigma^2)
    nu_0 = para.BASE_MEAN + para.BASE_SIGMA * torch.randn(NU_SIZE, para.DIM, device=device)

    # Save NU_0
    path_nu_0 = os.path.join(DATA_DIR, f'{para.ABBR}_samples_NU_0.mat')
    scipy.io.savemat(path_nu_0, {'samples_NU_0': nu_0.cpu().numpy()})
    print(f"       Saved Base Samples: {path_nu_0}")

    # 2. Iterate through indices 0 to 4
    # Thetas: [0.0, 0.25, 0.50, 0.75, 1.00]
    thetas = [0.0, 0.25, 0.50, 0.75, 1.00]

    for i, theta in enumerate(thetas):
        generate_for_index(i, theta, nu_0)

    print(f"\n[Main] All generation tasks complete.")