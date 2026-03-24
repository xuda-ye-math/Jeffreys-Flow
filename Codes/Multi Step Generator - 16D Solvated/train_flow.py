# ---- 1. Header and Imports ----
from parameters import *  # Imports para, EPOCHS, load_data, NU_SIZE, DATA_DIR, REJU_MAX_SIZE
from model import Normalizing_Flow
from utilities import compute_CESS, compute_ESS, normalize_log_weights, distribution_resample, \
    distribution_rejuvenation, generate_mixed_base_samples
import torch
import torch.optim as optim
import numpy as np
import os
import scipy.io
import re
from tqdm import tqdm

# ---- 2. Configuration ----
# Seed
SEED = 30
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameter for Jeffreys Loss
THETA = 0.75
ALPHA = 1.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")

# Sync Potential Helper with current device
para.to(device)
print(f"[Config] System Name: {para.NAME} ({para.ABBR})")
print(f"[Config] Theta: {THETA}")

# ---- 3. Loss & Potential Functions ----

def get_potential_func(beta_val):
    """
    Returns a callable function for the potential energy at a specific lambda (beta).
    """
    return lambda x: para.compute_potential_mixed(x, lam=beta_val)


def compute_jeffreys_loss_generalized(model, batch_source, batch_target, u_source_func, u_target_func, lambda_0,
                                      lambda_1):
    # Energy KL -> Renyi Divergence (Forward: Source -> Target Energy)
    z_fwd, log_det_fwd = model(batch_source, inverse=False)
    
    log_ratios_fwd = u_target_func(z_fwd) - u_source_func(batch_source) - log_det_fwd
    
    # Renyi Aggregation (Alpha-Divergence)
    # D_alpha(q||p) = 1/(alpha-1) * log E_q [ (q/p)^(alpha-1) ]
    # log_ratios_fwd = log(q/p)
    
    if ALPHA == 1.0:
        loss_energy = torch.mean(log_ratios_fwd)
    else:
        # Default Aggregation: LogSumExp trick
        # log E[exp( (alpha-1) * log(q/p) )]
        log_sum_exp_fwd = torch.logsumexp((ALPHA - 1) * log_ratios_fwd, dim=0)
        loss_energy = (log_sum_exp_fwd - np.log(log_ratios_fwd.size(0))) / (ALPHA - 1)

    # Data KL -> Renyi Divergence (Reverse: Target Samples -> Source Energy)
    x_rec, log_det_bwd = model(batch_target, inverse=True)
    
    log_ratios_bwd = u_source_func(x_rec) - u_target_func(batch_target) - log_det_bwd
    # log_ratios_bwd = log(p/q)
    
    if ALPHA == 1.0:
        loss_data = torch.mean(log_ratios_bwd)
    else:
        log_sum_exp_bwd = torch.logsumexp((ALPHA - 1) * log_ratios_bwd, dim=0)
        loss_data = (log_sum_exp_bwd - np.log(log_ratios_bwd.size(0))) / (ALPHA - 1)

    return lambda_0 * loss_energy + lambda_1 * loss_data, loss_energy, loss_data


# ---- 4. Training Routine ----

def train_single_step(step_k, mu_source, mu_target, u_source_func, u_target_func):
    print(f"\n" + "=" * 60)
    print(f"  TRAINING STEP {step_k}: Flow_{step_k} (Theta={THETA})")
    print("=" * 60)

    # [Update] Removed forced projection of mu_source/mu_target to [-pi, pi].
    # The Flow model should learn the periodicity naturally from the data distribution.
    # The Variances below are calculated on the raw data (which is fine for Bath dims).

    # Calculate Variances for Loss Scaling
    # Calculate Variances for Loss Scaling (Bath dims only: 2->16)
    var_0 = torch.var(mu_source[:, 2:], dim=0).sum().item()
    var_1 = torch.var(mu_target[:, 2:], dim=0).sum().item()
    lambda_0 = THETA * var_0
    lambda_1 = (1.0 - THETA) * var_1

    print(f"  > Var(Source): {var_0:.4f} | Var(Target): {var_1:.4f}")

    # Initialize Flow Model
    model = Normalizing_Flow().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    data_size = mu_source.shape[0]
    num_batches = data_size // BATCH_SIZE
    model.train()

    for epoch in range(EPOCHS):
        perm_0 = torch.randperm(data_size)
        perm_1 = torch.randperm(data_size)
        avg_loss, avg_l_e, avg_l_d = 0.0, 0.0, 0.0

        # Inner Loop Progress Bar (Batches)
        pbar = tqdm(range(num_batches), desc=f"Step {step_k} [Ep {epoch+1}/{EPOCHS}]", leave=False)
        
        for i in pbar:
            idx_0 = perm_0[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            idx_1 = perm_1[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

            optimizer.zero_grad()
            loss, l_e, l_d = compute_jeffreys_loss_generalized(
                model, mu_source[idx_0], mu_target[idx_1],
                u_source_func, u_target_func, lambda_0, lambda_1
            )
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            avg_loss += current_loss
            avg_l_e += l_e.item()
            avg_l_d += l_d.item()
            
            # Update description with current batch loss
            pbar.set_postfix({
                "L": f"{current_loss:.4f}",
                "E_KL": f"{l_e.item():.4f}",
                "D_KL": f"{l_d.item():.4f}",
            })

        # End of Epoch Summary
        print(f"  [Epoch {epoch + 1}/{EPOCHS}] Avg Loss: {avg_loss / num_batches:.4f} | E_KL: {avg_l_e / num_batches:.4f} | D_KL: {avg_l_d / num_batches:.4f}")

    return model


# ---- 5. Checkpoint & Resume Logic ----

def find_latest_checkpoint(data_dir, abbr):
    """
    Scans for the latest completed step k.
    Criteria: Must have flow_{k}.pth AND samples_NU_{k}.mat AND weights_NU_{k}.mat
    """
    pattern_flow = rf"{abbr}_flow_(\d+)\.pth"
    files = os.listdir(data_dir)

    completed_steps = []

    for f in files:
        match = re.match(pattern_flow, f)
        if match:
            k = int(match.group(1))
            # Check if corresponding sample/weight files exist
            f_samp = os.path.join(data_dir, f"{abbr}_samples_NU_{k}.mat")
            f_wght = os.path.join(data_dir, f"{abbr}_weights_NU_{k}.mat")
            if os.path.exists(f_samp) and os.path.exists(f_wght):
                completed_steps.append(k)

    if not completed_steps:
        return 0

    return max(completed_steps)


def load_checkpoint_state(data_dir, abbr, k):
    """
    Loads NU samples and weights from step k.
    """
    print(f"[Resume] Loading state from Step {k}...")
    f_samp = os.path.join(data_dir, f"{abbr}_samples_NU_{k}.mat")
    f_wght = os.path.join(data_dir, f"{abbr}_weights_NU_{k}.mat")

    mat_samp = scipy.io.loadmat(f_samp)
    mat_wght = scipy.io.loadmat(f_wght)

    samples = torch.from_numpy(mat_samp['samples']).float().to(device)
    weights = torch.from_numpy(mat_wght['weights']).float().to(device).view(-1)

    # Try to load ESS/CESS if available, just for log info
    cess = mat_wght.get('cess', [[0.0]])[0][0]
    ess = mat_wght.get('ess', [[0.0]])[0][0]

    print(f"  > Loaded {samples.shape[0]} samples.")
    print(f"  > Metrics at Step {k}: CESS={cess:.2f}, ESS={ess:.2f}")

    return samples, weights


# ---- 6. Main Execution ----

if __name__ == "__main__":
    # 1. Load Targets and Beta List from Data Directory
    pt_data, loaded_beta_list = load_data(device=device)

    # Validate Beta List
    if loaded_beta_list is None:
        raise RuntimeError(
            f"BETA_LIST not found in {DATA_DIR}. Please run the simulation first to generate '{para.ABBR}_BETA_LIST.mat'.")

    BETA_LIST = loaded_beta_list
    M = len(BETA_LIST) - 1

    print(f"[Config] Ladder Steps (M): {M}")
    beta_list_print = [f"{b.item():.3f}" for b in BETA_LIST]
    print(f"[Config] Beta Sequence: {beta_list_print}")

    # 2. Determine Start Step (Resume Logic)
    latest_k = find_latest_checkpoint(DATA_DIR, para.ABBR)

    if latest_k == 0:
        # --- Start from Scratch ---
        print(f"\n[Initialization] No valid checkpoint found. Starting from Step 0.")
        print(f"[Initialization] Generating NU_0 (Base Dist, N={NU_SIZE})...")

        # Use generic utility to generate samples
        nu_samples = generate_mixed_base_samples(para, n_samples=NU_SIZE, device=device)
        nu_log_weights = torch.zeros(NU_SIZE, device=device)

        # Save Initial State (Step 0)
        scipy.io.savemat(
            os.path.join(DATA_DIR, f'{para.ABBR}_samples_NU_0.mat'),
            {'samples': nu_samples.cpu().numpy()}
        )
        scipy.io.savemat(
            os.path.join(DATA_DIR, f'{para.ABBR}_weights_NU_0.mat'),
            {
                'weights': nu_log_weights.cpu().numpy(),
                'cess': 0.0,
                'ess': 1.0
            }
        )
        print(f"  [IO] Saved NU_0 samples/weights to .mat")
        start_step = 1

    elif latest_k >= M:
        print(f"\n[Resume] Training already completed (Found Step {latest_k} >= M={M}). Exiting.")
        exit(0)

    else:
        # --- Resume ---
        print(f"\n[Resume] Found checkpoint at Step {latest_k}. Resuming from Step {latest_k + 1}...")
        # Load state from latest_k (samples become the 'prev' samples for step k+1)
        nu_samples, nu_log_weights = load_checkpoint_state(DATA_DIR, para.ABBR, latest_k)
        start_step = latest_k + 1

    # 3. Sequential Training Loop
    for k in range(start_step, M + 1):
        curr_idx = str(k)

        # Get Beta Values
        beta_prev = BETA_LIST[k - 1].item()
        beta_curr = BETA_LIST[k].item()

        print(f"\n>>> PROCESSING STAGE {k} / {M} (Beta {beta_prev:.3f} -> {beta_curr:.3f})")

        # =========================================================
        # A. Prepare Training Source (mu_source)
        # =========================================================

        if curr_idx not in pt_data:
            raise ValueError(f"Target data MU_{curr_idx} not found in loaded PT data.")
        mu_target_pt = pt_data[curr_idx]  # Target for training (Data KL)

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
            para, mu_source, lam=beta_prev,
            dt=para.DT, n_iterations=para.N_ITER,
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
        # Save model checkpoint
        torch.save(model_k.state_dict(), os.path.join(DATA_DIR, f'{para.ABBR}_flow_{k}.pth'))

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
        nu_log_weights = nu_log_weights + incremental_log_w

        # =========================================================
        # D. Evaluate Metrics (Log-Space Stable)
        # =========================================================

        # We assume uniform previous alphas (1/N) because we either start fresh or just resampled
        log_uniform_alphas = torch.full_like(incremental_log_w, -np.log(NU_SIZE))

        # [FIX] Using compute_CESS which is now log-stable from utilities.py
        cess_ratio = compute_CESS(log_uniform_alphas, incremental_log_w)

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
            para, nu_samples, lam=beta_curr,
            dt=para.DT, n_iterations=para.N_ITER,
            batch_size=REJU_MAX_SIZE
        )

        # =========================================================
        # F. Save
        # =========================================================
        scipy.io.savemat(
            os.path.join(DATA_DIR, f'{para.ABBR}_samples_NU_{k}.mat'),
            {'samples': nu_samples.cpu().numpy()}
        )

        # Save Weights AND Metrics (ESS will be 1.0 if resampled, otherwise actual value)
        scipy.io.savemat(
            os.path.join(DATA_DIR, f'{para.ABBR}_weights_NU_{k}.mat'),
            {
                'weights': nu_log_weights.cpu().numpy(),
                'cess': cess_ratio,
                'ess': ess_to_save
            }
        )
        print(f"  [IO] Saved NU_{k} samples/weights (CESS={cess_ratio:.2f}, ESS={ess_to_save:.2f}) to .mat")

    print("\n[Main] Sequential training completed.")