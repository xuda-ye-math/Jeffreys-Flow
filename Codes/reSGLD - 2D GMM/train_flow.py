
# ---- 1. Header and Imports ----
from parameters import * # Imports para, EPOCHS, load_data, NU_SIZE, DATA_DIR, REJU_MAX_SIZE
from model import Normalizing_Flow
from utilities import compute_ESS, normalize_log_weights, distribution_resample, distribution_rejuvenation
import torch
import torch.optim as optim
import numpy as np
import os
import scipy.io  # [NEW] Added for MATLAB compatibility

# ---- 2. Configuration ----
# Seed
SEED = 30
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameter for Jeffreys Loss
THETA = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")

# Sync Potential Helper with current device
para.to(device)
print(f"[Config] Potential System: {para.NAME} ({para.ABBR})")
print(f"[Config] Theta: {THETA}")

# ---- 3. Loss & Potential Functions ----

def get_potential_func(beta_val):
    return lambda x, i_vec=None: para.compute_potential_mixed(x, lam=beta_val, i_vec=i_vec)


def compute_jeffreys_loss_generalized(model, batch_source, batch_target, u_source_func, u_target_func, lambda_0,
                                      lambda_1):
    # Initialize i_vec [Batch, 2]
    batch_size = batch_source.shape[0]
    signs = torch.randint(0, 2, (batch_size, 2), device=batch_source.device).float() * 2 - 1
    i_vec = signs * para.NOISE

    # Energy KL
    z_fwd, log_det_fwd = model(batch_source, inverse=False)
    # Pass i_vec to potentials
    loss_energy = torch.mean(u_target_func(z_fwd, i_vec) - u_source_func(batch_source, i_vec) - log_det_fwd)

    # Data KL
    x_rec, log_det_bwd = model(batch_target, inverse=True)
    loss_data = torch.mean(u_source_func(x_rec, i_vec) - u_target_func(batch_target, i_vec) - log_det_bwd)

    return lambda_0 * loss_energy + lambda_1 * loss_data, loss_energy, loss_data


# ---- 4. Training Routine ----

def train_single_step(step_k, mu_source, mu_target, u_source_func, u_target_func):
    print(f"\n" + "=" * 60)
    print(f"  TRAINING STEP {step_k}: Flow_{step_k} (Theta={THETA})")
    print("=" * 60)

    # Variances for Scaling
    var_0 = torch.var(mu_source, dim=0).sum().item()
    var_1 = torch.var(mu_target, dim=0).sum().item()
    lambda_0 = THETA * var_0
    lambda_1 = (1.0 - THETA) * var_1

    print(f"  > Var(Source): {var_0:.4f} | Var(Target): {var_1:.4f}")

    model = Normalizing_Flow().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_F, weight_decay=WEIGHT_DECAY)

    data_size = mu_source.shape[0]
    num_batches = data_size // BATCH_SIZE
    model.train()

    for epoch in range(EPOCHS_F):
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
            f"  [Epoch {epoch + 1}/{EPOCHS_F}] Loss: {avg_loss / num_batches:.4f} | E_KL: {avg_l_e / num_batches:.4f} | D_KL: {avg_l_d / num_batches:.4f}")

    return model


def train_weight_function(step_k, model, mu_source, u_source_func, u_target_func):
    """
    Trains the phi and psi networks to minimize the L2 variance of the weight function.

    Args:
        step_k: Current step index.
        model: Trained Flow model F_k.
        mu_source: Samples from the source distribution (mu'_{k-1}).
        u_source_func: Function to compute U_{k-1}(x).
        u_target_func: Function to compute U_k(x).

    Returns:
        phi, psi: Trained WeightNet instances.
    """
    from model import WeightNet
    print(f"\n" + "=" * 60)
    print(f"  TRAINING WEIGHTS (Phi, Psi) for STEP {step_k}")
    print("=" * 60)

    # Initialize Networks
    phi = WeightNet(para.DIM).to(device)
    psi = WeightNet(para.DIM).to(device)

    # Optimizer
    optimizer = optim.Adam(
        list(phi.parameters()) + list(psi.parameters()),
        lr=LR_W, weight_decay=WEIGHT_DECAY
    )

    data_size = mu_source.shape[0]
    num_batches = data_size // BATCH_SIZE

    # Ensure Flow is in Eval mode (fixed)
    model.eval()
    phi.train()
    psi.train()

    print(f"  [Weight Training] Data Size: {data_size} | Batches: {num_batches}")

    for epoch in range(EPOCHS_W):
        perm = torch.randperm(data_size)
        avg_loss = 0.0

        for i in range(num_batches):
            idx = perm[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            x0 = mu_source[idx]

            optimizer.zero_grad()

            # 1. Compute Flow Transform (No Grad for Flow)
            with torch.no_grad():
                z, _ = model(x0, inverse=False)
                
                # Generate i_vec for Weight Training (Shared)
                batch_size_w = x0.shape[0]
                signs = torch.randint(0, 2, (batch_size_w, 2), device=x0.device).float() * 2 - 1
                i_vec = signs * para.NOISE

                # U_k(F_k(x0))
                u_k_z = u_target_func(z, i_vec)
                # U_{k-1}(x0)
                u_km1_x0 = u_source_func(x0, i_vec)

            # 2. Compute Phi(x0) and Psi(z)
            phi_val = phi(x0).squeeze()
            psi_val = psi(z).squeeze()

            # 3. L2 Loss
            # Loss = E[ | U_k(z) - U_{k-1}(x) + phi(x) - psi(z) |^2 ]
            term = u_k_z - u_km1_x0 + phi_val - psi_val
            loss = torch.mean(term ** 2)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        print(f"  [Weight Epoch {epoch + 1}/{EPOCHS_W}] L2 Loss: {avg_loss / num_batches:.6f}")

    return phi, psi


# ---- 5. Main Execution ----

def run_training_pipeline(mode='correct', use_mala=True):
    """
    Executes the full Jeffreys Flow training pipeline for the specified mode.
    Modes: 'correct' | 'naive'
    MALA: True | False (Controls rejuvenation)
    """
    mala_suffix = "mala" if use_mala else "no_mala"
    mode_full = f"{mode}_{mala_suffix}"
    n_iter = para.N_ITER if use_mala else 0

    print(f"\n{'='*80}")
    print(f"  STARTING TRAINING PIPELINE: {mode.upper()} ({mala_suffix.upper()})")
    print(f"{'='*80}")
    
    # 1. Load Targets and Beta List
    # Note: Target MU samples depend only on 'mode' (naive/correct), not MALA usage here.
    pt_data, loaded_beta_list = load_data(device=device, mode=mode)

    # Validate Beta List
    if loaded_beta_list is None:
        raise RuntimeError(
            f"BETA_LIST not found in {DATA_DIR}. Please run the MATLAB simulation first to generate '{para.ABBR}_BETA_LIST.mat'.")

    BETA_LIST = loaded_beta_list
    M = len(BETA_LIST) - 1

    print(f"[Config] Ladder Steps (M): {M}")
    beta_list_print = [f"{b.item():.3f}" for b in BETA_LIST]
    print(f"[Config] Beta Sequence: {beta_list_print}")
    print(f"[Config] MALA Rejuvenation: {'Enabled' if use_mala else 'Disabled'} (Iter={n_iter})")

    # 2. Initialize NU_0 (Base Distribution)
    print(f"\n[Initialization] Generating NU_0 (Gaussian, N={NU_SIZE})...")
    nu_samples = para.BASE_MEAN + para.BASE_SIGMA * torch.randn(NU_SIZE, para.DIM, device=device)
    nu_log_weights = torch.zeros(NU_SIZE, device=device)

    # [UPDATED] Save Initial State as .mat (Full Mode Specific)
    scipy.io.savemat(
        os.path.join(DATA_DIR, f'{para.ABBR}_samples_{mode_full}_NU_0.mat'),
        {'samples': nu_samples.cpu().numpy()}
    )
    scipy.io.savemat(
        os.path.join(DATA_DIR, f'{para.ABBR}_weights_{mode_full}_NU_0.mat'),
        {'weights': nu_log_weights.cpu().numpy()}
    )
    print(f"  [IO] Saved NU_0 ({mode_full}) samples/weights to .mat")

    # 3. Sequential Training Loop
    for k in range(1, M + 1):
        curr_idx = str(k)

        # Get Beta Values (Convert to float for GMM Mixed interface)
        beta_prev = BETA_LIST[k - 1].item()
        beta_curr = BETA_LIST[k].item()

        print(f"\n>>> PROCESSING STAGE {k} / {M} (Beta {beta_prev:.3f} -> {beta_curr:.3f})")

        # =========================================================
        # A. Prepare Training Source (mu_source)
        # =========================================================

        if curr_idx not in pt_data:
            print(f"Warning: Target data MU_{curr_idx} not found in loaded PT data ({mode}). Skipping step.")
            continue
            
        mu_target_pt = pt_data[curr_idx]  # Target for training (Data KL)

        print("  [Data Prep] Resampling nu_{k-1} to create training source...")
        current_weights = normalize_log_weights(nu_log_weights)

        # [CRITICAL] This generates a NEW tensor 'mu_source'.
        # It does NOT modify 'nu_samples' (the ensemble) itself.
        mu_source = distribution_resample(
            samples=nu_samples,
            weights=current_weights,
            n_resamples=mu_target_pt.shape[0]  # Match size of PT target (e.g. 50k)
        )

        # Rejuvenate Training Source (Target: U_{k-1})
        print("  [Data Prep] Rejuvenating training source (MALA)...")
        
        mu_source = distribution_rejuvenation(
            para, mu_source, lam=beta_prev,
            dt=para.DT, n_iterations=n_iter,
            batch_size=REJU_MAX_SIZE
        )

        # =========================================================
        # B. Train Flow k
        # =========================================================
        u_prev_func = get_potential_func(beta_prev)
        u_curr_func = get_potential_func(beta_curr)

        model_k = train_single_step(
            step_k=k, mu_source=mu_source, mu_target=mu_target_pt,
            u_source_func=u_prev_func, u_target_func=u_curr_func
        )
        # =========================================================
        # C. Train Weight Function (Phi, Psi)
        # =========================================================
        phi, psi = train_weight_function(
            step_k=k, model=model_k, mu_source=nu_samples,
            u_source_func=u_prev_func, u_target_func=u_curr_func
        )

        # [UPDATED] Save All Models in One Checkpoint (Full Mode Specific)
        checkpoint = {
            'flow_state_dict': model_k.state_dict(),
            'phi_state_dict': phi.state_dict(),
            'psi_state_dict': psi.state_dict(),
            'step': k,
            'beta_prev': beta_prev,
            'beta_curr': beta_curr,
            'mode': mode,
            'use_mala': use_mala
        }
        timestamp = k  # using k as timestamp
        checkpoint_path = os.path.join(DATA_DIR, f'{para.ABBR}_flow_{mode_full}_{k}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"  [IO] Saved Flow, Phi, Psi to {checkpoint_path}")

        # =========================================================
        # D. Propagate Ensemble (nu_{k-1} -> nu_k)
        # =========================================================
        print(f"  [Propagation] Applying Flow_{k} to NU_{k - 1} using Learned Weights...")
        model_k.eval()
        phi.eval()
        psi.eval()

        new_samples_list = []
        incremental_logw_list = []

        num_gen_batches = int(np.ceil(NU_SIZE / 5000))

        with torch.no_grad():
            for i in range(num_gen_batches):
                st, en = i * 5000, min((i + 1) * 5000, NU_SIZE)
                x_prev = nu_samples[st:en]

                # 1. Flow Transport
                # x_curr = F_k(x_prev)
                x_curr, log_det = model_k(x_prev, inverse=False)

                # 2. Compute Learned Weights
                # log w = -phi(x_prev) - psi(x_curr)
                # Note: These return shape [Batch, 1], squeeze to [Batch]
                phi_val = phi(x_prev).squeeze()
                psi_val = psi(x_curr).squeeze()

                # Calculate Incremental Log Weight
                # The target was: log w = -U_k(z) + U_{k-1}(x) + log_det (from note.tex)
                # The training minimized: | U_k(z) - U_{k-1}(x) + phi(x) - psi(z) |^2
                # So phi(x) - psi(z) approx -(U_k(z) - U_{k-1}(x))
                # Thus log w approx phi(x) - psi(z) + log_det
                log_w = phi_val - psi_val + log_det

                new_samples_list.append(x_curr)
                incremental_logw_list.append(log_w)

        # Update Ensemble State
        nu_samples = torch.cat(new_samples_list, dim=0)
        incremental_log_w = torch.cat(incremental_logw_list, dim=0)
        
        # [UPDATED] CESS Calculation (Log-Stable)
        # Using previous log weights (nu_log_weights) as alpha
        log_alpha = nu_log_weights
        log_w = incremental_log_w
        
        # Formula: 2*log(sum(alpha*w)) - log(sum(alpha*w^2)) - log(sum(alpha))
        term1 = torch.logsumexp(log_alpha + log_w, dim=0)
        term2 = torch.logsumexp(log_alpha + 2 * log_w, dim=0)
        term3 = torch.logsumexp(log_alpha, dim=0)
        
        log_cess = 2 * term1 - term2 - term3
        cess_ratio = torch.exp(log_cess).item()
        
        # Now update nu_log_weights for the next step
        nu_log_weights = nu_log_weights + incremental_log_w

        # =========================================================
        # D. Evaluate Metrics
        # =========================================================

        # ESS (Total Ensemble Efficiency)
        ess_ratio = compute_ESS(normalize_log_weights(nu_log_weights))

        print(f"  [Metrics] CESS (Step): {cess_ratio:.2%} | ESS (Total): {ess_ratio:.2%}")

        # =========================================================
        # E. Adaptive Resampling & Rejuvenation
        # =========================================================

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

            # Update ESS for logging (conceptually it's now 100%)
            ess_ratio = 1.0
        else:
            print(f"  [Adaptive] ESS >= 50%. Skipping Resampling (Weights preserved).")

        # 2. Unconditional Rejuvenation (Target: U_k)
        # Always use MALA to disperse particles into the current target landscape (U_k).
        print(f"  [Adaptive] Performing Rejuvenation (Target: Beta {beta_curr:.3f}) Iter={n_iter}...")
        nu_samples = distribution_rejuvenation(
            para, nu_samples, lam=beta_curr,
            dt=para.DT, n_iterations=n_iter,
            batch_size=REJU_MAX_SIZE
        )

        print(f"  [Adaptive] Step Completed. Final ESS Status: {ess_ratio:.2%}")

        # =========================================================
        # F. Save
        # =========================================================
        # [UPDATED] Save samples and weights as .mat for MATLAB plotting (Full Mode Specific)
        scipy.io.savemat(
            os.path.join(DATA_DIR, f'{para.ABBR}_samples_{mode_full}_NU_{k}.mat'),
            {'samples': nu_samples.cpu().numpy()}
        )
        scipy.io.savemat(
            os.path.join(DATA_DIR, f'{para.ABBR}_weights_{mode_full}_NU_{k}.mat'),
            {'weights': nu_log_weights.cpu().numpy(),
             'cess': cess_ratio,
             'ess': ess_ratio}
        )
        print(f"  [IO] Saved NU_{k} ({mode_full}) samples/weights to .mat")

    print(f"\n[Pipeline] Training completed for mode: {mode_full}")


if __name__ == "__main__":
    # Run All 4 Combinations
    
    # 1. Correct + MALA
    run_training_pipeline(mode='correct', use_mala=True)
    
    # 2. Correct + No MALA
    run_training_pipeline(mode='correct', use_mala=False)
    
    # 3. Naive + MALA
    run_training_pipeline(mode='naive', use_mala=True)
    
    # 4. Naive + No MALA
    run_training_pipeline(mode='naive', use_mala=False)