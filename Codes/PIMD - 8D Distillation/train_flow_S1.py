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

def get_potential_func(beta_val):
    """
    Returns a callable function for the potential energy at a specific lambda (beta).
    """
    return lambda x: (1.0 - beta_val) * pt_global.base_Uf(x) + beta_val * pt_global.target_Uf(x)

def compute_jeffreys_loss_generalized(model, batch_source, batch_target, u_source_func, u_target_func, lambda_0,
                                      lambda_1):
    # Energy KL (Forward: Source -> Target Energy)
    z_fwd, log_det_fwd = model(batch_source, inverse=False)
    loss_energy = torch.mean(u_target_func(z_fwd) - u_source_func(batch_source) - log_det_fwd)

    # Data KL (Reverse: Target Samples -> Source Energy)
    x_rec, log_det_bwd = model(batch_target, inverse=True)
    loss_data = torch.mean(u_source_func(x_rec) - u_target_func(batch_target) - log_det_bwd)

    return lambda_0 * loss_energy + lambda_1 * loss_data, loss_energy, loss_data

# ---- 4. Training Routine ----

def train_single_step(step_k, mu_source, mu_target, u_source_func, u_target_func):
    print(f"\n" + "=" * 60)
    print(f"  TRAINING STEP {step_k}: Flow_{step_k} (Theta={THETA})")
    print("=" * 60)

    # Calculate Variances for Loss Scaling
    var_0 = torch.var(mu_source, dim=0).sum().item()
    var_1 = torch.var(mu_target, dim=0).sum().item()
    lambda_0 = THETA * var_0
    lambda_1 = (1.0 - THETA) * var_1

    print(f"  > Var(Source): {var_0:.4f} | Var(Target): {var_1:.4f}")

    # Initialize Flow Model
    model = Normalizing_Flow(pt=pt_global).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    data_size = mu_source.shape[0]
    num_batches = data_size // BATCH_SIZE
    model.train()

    for epoch in range(EPOCHS):
        perm_0 = torch.randperm(data_size)
        perm_1 = torch.randperm(data_size)
        avg_loss, avg_l_e, avg_l_d = 0.0, 0.0, 0.0

        for i in range(num_batches):
            idx_0 = perm_0[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            idx_1 = perm_1[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

            optimizer.zero_grad()
            loss, l_e, l_d = compute_jeffreys_loss_generalized(
                model, mu_source[idx_0], mu_target[idx_1],
                u_source_func, u_target_func, lambda_0, lambda_1
            )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            avg_l_e += l_e.item()
            avg_l_d += l_d.item()

        print(
            f"  [Epoch {epoch + 1}/{EPOCHS}] Loss: {avg_loss / num_batches:.4f} | E_KL: {avg_l_e / num_batches:.4f} | D_KL: {avg_l_d / num_batches:.4f}")

    return model


# [DELETED CHECKPOINT LOGIC]


# ---- 5. Main Execution ----

if __name__ == "__main__":
    # 1. Load MU Dataset and Schedule
    mu_path = os.path.join(DATA_DIR, "samples_MU_S1.pth")
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
        torch.save(model_k.state_dict(), os.path.join(DATA_DIR, f'flow_S1_{k}.pth'))

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
    torch.save(samples_all, os.path.join(DATA_DIR, 'samples_NU_S1.pth'))
    torch.save(
        {'weights': weights_all, 'ess': ess_all, 'cess': cess_all},
        os.path.join(DATA_DIR, 'weights_NU_S1.pth')
    )
    
    # Merge flows_S1.pth entirely and delete residual components
    global_flows = {}
    for k in range(1, M+1):
        temp_path = os.path.join(DATA_DIR, f'flow_S1_{k}.pth')
        global_flows[f'flow_{k}'] = torch.load(temp_path)
        os.remove(temp_path)
    
    torch.save(global_flows, os.path.join(DATA_DIR, 'flows_S1.pth'))
    
    print("\n[Main] Model training completed continuously. Global outputs uniformly merged!")