import torch
import numpy as np
import os

from parameters import *
from utilities import *

# ---- 2. Configuration ----
# Seed
SEED = 31
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameter for Jeffreys Loss
THETA = 0.75

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")

# Sync Potential Helper with current device
pt_global = pt.to(device)
print(f"[Config] Theta: {THETA}")

# MALA config
MALA_DT = 1e-3
MALA_ITERS = 1000
REJU_MAX_SIZE = 20000

# ---- 3. Loss & Potential Functions ----

from train_flow_S1 import get_potential_func, train_single_step

# [DELETED CHECKPOINT LOGIC]


# ---- 5. Main Execution ----

if __name__ == "__main__":
    # 1. Load MU Dataset and Schedule
    mu_path = os.path.join(DATA_DIR, "samples_MU_S2.pth")
    if not os.path.exists(mu_path):
        raise RuntimeError(f"Missing required dataset: {mu_path}")
        
    pt_data = torch.load(mu_path).to(device) # Shape [M+1, MU_SIZE, N]
    M = pt_data.shape[0] - 1

    BETA_LIST = torch.linspace(0, 1, M+1, device=device)
    
    print(f"[Config] Ladder Steps (M): {M}")
    beta_list_print = [f"{b.item():.3f}" for b in BETA_LIST]
    print(f"[Config] Beta Sequence: {beta_list_print}")

    # Prepare Global Memory Repositories
    samples_all = torch.zeros(M+1, NU_SIZE, N)
    weights_all = torch.zeros(M+1, NU_SIZE)
    ess_all = torch.zeros(M+1)
    cess_all = torch.zeros(M+1)

    print(f"\n[Initialization] Generating NU_0 (Base Dist, N={NU_SIZE})...")
    nu_samples = generate_base_samples(pt_global, n_samples=NU_SIZE, device=device)
    nu_log_weights = torch.zeros(NU_SIZE, device=device)

    # Save Initial State Memory locally
    samples_all[0] = nu_samples.cpu()
    weights_all[0] = nu_log_weights.cpu()
    ess_all[0] = 1.0
    cess_all[0] = 0.0

    # 3. Sequential Training Loop
    for k in range(1, M + 1):
        # Get Beta Values
        beta_prev = BETA_LIST[k - 1].item()
        beta_curr = BETA_LIST[k].item()

        print(f"\n>>> PROCESSING STAGE {k} / {M} (Beta {beta_prev:.3f} -> {beta_curr:.3f})")

        # =========================================================
        # A. Prepare Training Source (mu_source)
        # =========================================================
        mu_target_pt = pt_data[k]  # Target for training (Data KL)

        print("  [Data Prep] Resampling nu_{k-1} to create training source...")
        current_weights = normalize_log_weights(nu_log_weights)

        # Resample to create unweighted training set
        mu_source = distribution_resample(
            samples=nu_samples,
            weights=current_weights,
            n_resamples=mu_target_pt.shape[0]  # Match size of PT target
        )

        # Rejuvenate Training Source (Target: U_{k-1}) to decouple from history
        print("  [Data Prep] Rejuvenating training source (MALA)...")
        mu_source = distribution_rejuvenation(
            pt_global, mu_source, lam=beta_prev,
            dt=MALA_DT, n_iterations=MALA_ITERS,
            batch_size=REJU_MAX_SIZE
        )

        # =========================================================
        # B. Train Flow k
        # =========================================================
        u_prev_func = get_potential_func(beta_prev)
        u_curr_func = get_potential_func(beta_curr)

        # [FIX] Arguments passed positionally to avoid "positional follows keyword" syntax error
        model_k = train_single_step(
            k, mu_source, mu_target_pt,
            u_prev_func, u_curr_func
        )
        # Save model checkpoint explicitly isolating models without Abbr constants
        torch.save(model_k.state_dict(), os.path.join(DATA_DIR, f'flow_S2_{k}.pth'))

        # =========================================================
        # C. Propagate Ensemble (nu_{k-1} -> nu_k)
        # =========================================================
        print(f"  [Propagation] Applying Flow_{k} to NU_{k - 1}...")
        model_k.eval()
        new_samples_list = []
        incremental_logw_list = []

        num_gen_batches = int(np.ceil(NU_SIZE / 5000))

        with torch.no_grad():
            for i in range(num_gen_batches):
                st, en = i * 5000, min((i + 1) * 5000, NU_SIZE)
                x_prev = nu_samples[st:en]

                x_curr, log_det = model_k(x_prev, inverse=False)
                u_prev = u_prev_func(x_prev)
                u_curr = u_curr_func(x_curr)

                # Incremental Weight: log w = -U_k(x_k) + U_{k-1}(x_{k-1}) + log|det J|
                new_samples_list.append(x_curr)
                incremental_logw_list.append(-u_curr + u_prev + log_det)

        # Update Ensemble State
        nu_samples = torch.cat(new_samples_list, dim=0)
        incremental_log_w = torch.cat(incremental_logw_list, dim=0)
        # [UPDATED] CESS Calculation using weights from nu_{k-1} as alpha
        
        # Calculate log_alphas (normalized log-weights of nu_{k-1})
        # Note: normalize_log_weights returns probabilities (linear), but compute_CESS in 8D util expects log-weights
        max_log_w = torch.max(nu_log_weights)
        log_sum = torch.logsumexp(nu_log_weights - max_log_w, dim=0) + max_log_w
        log_alphas = nu_log_weights - log_sum
        
        # Now update nu_log_weights for the next step
        nu_log_weights = nu_log_weights + incremental_log_w

        # =========================================================
        # D. Evaluate Metrics (Log-Space Stable)
        # =========================================================

        # [FIX] Using compute_CESS which is now log-stable from utilities.py
        # alpha = nu_{k-1} log_weights (normalized)
        # weights = incremental log_weights
        cess_ratio = compute_CESS(log_alphas, incremental_log_w)

        # Calculate ESS
        ess_ratio = compute_ESS(normalize_log_weights(nu_log_weights))

        print(f"  [Metrics] CESS (Step): {cess_ratio:.2%} | ESS (Total): {ess_ratio:.2%}")

        # =========================================================
        # E. Adaptive Resampling & Rejuvenation
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

        # 2. Unconditional Rejuvenation (Target: U_k)
        print(f"  [Adaptive] Performing Rejuvenation (Target: Beta {beta_curr:.3f})...")
        nu_samples = distribution_rejuvenation(
            pt_global, nu_samples, lam=beta_curr,
            dt=MALA_DT, n_iterations=MALA_ITERS,
            batch_size=REJU_MAX_SIZE
        )

        # =========================================================
        # F. Push Layer to History Repositories
        # =========================================================
        samples_all[k] = nu_samples.cpu()
        weights_all[k] = nu_log_weights.cpu()
        cess_all[k] = cess_ratio
        ess_all[k] = ess_to_save
        print(f"  [IO] Tracked NU_{k} variables structurally...")

    # Final Export Logic unifying multi-layer history tracking across all targets
    torch.save(samples_all, os.path.join(DATA_DIR, 'samples_NU_S2.pth'))
    torch.save(
        {'weights': weights_all, 'ess': ess_all, 'cess': cess_all},
        os.path.join(DATA_DIR, 'weights_NU_S2.pth')
    )
    
    # Merge flows_S2.pth entirely and delete residual components
    global_flows = {}
    for k in range(1, M+1):
        temp_path = os.path.join(DATA_DIR, f'flow_S2_{k}.pth')
        global_flows[f'flow_{k}'] = torch.load(temp_path)
        os.remove(temp_path)
    
    torch.save(global_flows, os.path.join(DATA_DIR, 'flows_S2.pth'))
    
    print("\n[Main] Model training completed continuously. Global outputs uniformly merged!")