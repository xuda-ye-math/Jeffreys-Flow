import torch
import torch.optim as optim
import numpy as np
import os

from parameters import *
from utilities import *
from model import Normalizing_Flow

# ---- 2. Configuration ----
# Seed
SEED = 30
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")

# Sync Potential Helper with current device
pt_global = pt.to(device)

# MALA config
MALA_DT = 1e-3
MALA_ITERS = 1000
REJU_MAX_SIZE = 100000

# ---- 3. Loss & Potential Functions ----

def get_potential_func(beta_val):
    """
    Returns a callable function for the potential energy at a specific lambda (beta).
    """
    return lambda x: (1.0 - beta_val) * pt_global.base_Uf(x) + beta_val * pt_global.target_Uf(x)

# ---- 5. Main Execution ----

if __name__ == "__main__":
    
    print(f"\n[Initialization] Setting up {N}-dimensional sampling using pre-trained {N0}-dimensional flow.")

    # 1. Load Flows
    flows_path = os.path.join(DATA_DIR, "flows.pth")
    if not os.path.exists(flows_path):
        raise RuntimeError(f"Missing required dataset: {flows_path}")
        
    global_flows = torch.load(flows_path, map_location=device)
    M = len(global_flows)

    BETA_LIST = torch.linspace(0, 1, M+1, device=device)
    
    print(f"[Config] Ladder Steps (M): {M}")
    beta_list_print = [f"{b.item():.3f}" for b in BETA_LIST]
    print(f"[Config] Beta Sequence: {beta_list_print}")

    # Prepare Global Memory Repositories
    samples_all = torch.zeros(M+1, NU_SIZE, N)
    weights_all = torch.zeros(M+1, NU_SIZE)
    ess_all = torch.zeros(M+1)
    cess_all = torch.zeros(M+1)

    print(f"\n[Initialization] Generating NU_0 (Base Dist, N_samples={NU_SIZE}, Dim={N})...")
    nu_samples = generate_base_samples(pt_global, n_samples=NU_SIZE, device=device)
    nu_log_weights = torch.zeros(NU_SIZE, device=device)

    # Save Initial State Memory locally
    samples_all[0] = nu_samples.cpu()
    weights_all[0] = nu_log_weights.cpu()
    ess_all[0] = 1.0
    cess_all[0] = 0.0
    
    # Pre-allocate the N0-dimensional potential for the flow definition
    from potential import Path_Integral_Potential
    pt_N0 = Path_Integral_Potential(N=N0, beta=1.0).to(device)

    # 3. Sequential Evaluation Loop
    for k in range(1, M + 1):
        # Get Beta Values
        beta_prev = BETA_LIST[k - 1].item()
        beta_curr = BETA_LIST[k].item()

        print(f"\n>>> PROCESSING STAGE {k} / {M} (Beta {beta_prev:.3f} -> {beta_curr:.3f})")

        # =========================================================
        # A. Load Flow k
        # =========================================================
        model_k = Normalizing_Flow(pt=pt_N0).to(device)
        model_k.load_state_dict(global_flows[f'flow_{k}'])
        model_k.eval()
        
        u_prev_func = get_potential_func(beta_prev)
        u_curr_func = get_potential_func(beta_curr)

        # =========================================================
        # B. Propagate Ensemble (nu_{k-1} -> nu_k)
        # =========================================================
        print(f"  [Propagation] Applying Flow_{k} (Dim={N0}) to NU_{k - 1} (Dim={N})...")
        new_samples_list = []
        incremental_logw_list = []

        num_gen_batches = int(np.ceil(NU_SIZE / 5000))

        with torch.no_grad():
            for i in range(num_gen_batches):
                st, en = i * 5000, min((i + 1) * 5000, NU_SIZE)
                x_prev = nu_samples[st:en] # Shape: [Batch, N]
                
                # Split into low frequencies and high frequencies
                x_prev_low = x_prev[:, :N0]
                x_prev_high = x_prev[:, N0:]

                # Apply flow only on the low frequencies
                x_curr_low, log_det = model_k(x_prev_low, inverse=False)
                
                # Recombine
                x_curr = torch.cat([x_curr_low, x_prev_high], dim=1)

                # Compute full-dimensional energies
                u_prev = u_prev_func(x_prev)
                u_curr = u_curr_func(x_curr)

                # Incremental Weight: log w = -U_k(x_k) + U_{k-1}(x_{k-1}) + log|det J|
                new_samples_list.append(x_curr)
                incremental_logw_list.append(-u_curr + u_prev + log_det)

        # Update Ensemble State
        nu_samples = torch.cat(new_samples_list, dim=0)
        incremental_log_w = torch.cat(incremental_logw_list, dim=0)
        
        # Calculate log_alphas (normalized log-weights of nu_{k-1})
        max_log_w = torch.max(nu_log_weights)
        log_sum = torch.logsumexp(nu_log_weights - max_log_w, dim=0) + max_log_w
        log_alphas = nu_log_weights - log_sum
        
        # Now update nu_log_weights for the next step
        nu_log_weights = nu_log_weights + incremental_log_w

        # =========================================================
        # C. Evaluate Metrics (Log-Space Stable)
        # =========================================================

        # Using compute_CESS which is now log-stable from utilities.py
        cess_ratio = compute_CESS(log_alphas, incremental_log_w)

        # Calculate ESS
        ess_ratio = compute_ESS(normalize_log_weights(nu_log_weights))

        print(f"  [Metrics] CESS (Step): {cess_ratio:.2%} | ESS (Total): {ess_ratio:.2%}")

        # =========================================================
        # D. Adaptive Resampling & Rejuvenation
        # =========================================================

        ess_to_save = ess_ratio

        # 1. Conditional Resampling
        if ess_ratio < 0.5:
            print(f"  [Adaptive] ESS < 50%. Performing Resampling (Weights reset)...")

            # Resample the FULL ensemble based on current weights
            w_norm = normalize_log_weights(nu_log_weights)
            nu_samples = distribution_resample(
                nu_samples,
                weights=w_norm,
                n_resamples=NU_SIZE
            )

            # Reset Weights to Uniform
            nu_log_weights = torch.zeros(NU_SIZE, device=device)

            # Update ESS to 1.0 (100%) for saving, as we have just resampled
            ess_to_save = 1.0
        else:
            print(f"  [Adaptive] ESS >= 50%. Skipping Resampling (Weights preserved).")

        # 2. Unconditional Rejuvenation (Target: full U_k)
        print(f"  [Adaptive] Performing Full Dimensional Rejuvenation (Target: Beta {beta_curr:.3f}, Dim={N})...")
        nu_samples = distribution_rejuvenation(
            pt_global, nu_samples, lam=beta_curr,
            dt=MALA_DT, n_iterations=MALA_ITERS,
            batch_size=REJU_MAX_SIZE
        )

        # =========================================================
        # E. Push Layer to History Repositories
        # =========================================================
        samples_all[k] = nu_samples.cpu()
        weights_all[k] = nu_log_weights.cpu()
        cess_all[k] = cess_ratio
        ess_all[k] = ess_to_save
        print(f"  [IO] Tracked NU_{k} variables structurally...")

    # Final Export Logic unifying multi-layer history tracking across all targets
    torch.save(samples_all, os.path.join(DATA_DIR, f'samples_NU_N{N}.pth'))
    torch.save(
        {'weights': weights_all, 'ess': ess_all, 'cess': cess_all},
        os.path.join(DATA_DIR, f'weights_NU_N{N}.pth')
    )
    
    print(f"\n[Main] Model evaluation completed. Global outputs saved for N={N}!")