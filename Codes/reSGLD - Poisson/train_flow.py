# ---- 1. Header and Imports ----
from parameters import * # Imports para, EPOCHS, load_data, NU_SIZE, DATA_DIR, REJU_MAX_SIZE
from model import Normalizing_Flow, WeightNet
from utilities import compute_ESS, normalize_log_weights, distribution_resample, distribution_rejuvenation
from Poisson_2D import Poisson_2D

import torch
import torch.optim as optim
import numpy as np
import os
import scipy.io
import time

# ---- 2. Configuration ----
# Seed
SEED = 30
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameter for Jeffreys Loss
THETA = 0.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")

# Sync Potential Helper with current device
para.to(device)
print(f"[Config] Potential System: {para.NAME}")
print(f"[Config] Theta: {THETA}")

# Limit NU_SIZE for initial testing to keep memory bounded, can be overridden by parameters.py
ensemble_size = NU_SIZE

# Pre-instantiate the Poisson batched GPU solver
solver = Poisson_2D(para.N_CELLS, para.ALPHA, para.GAMMA, para.C, device=device)

# Monitor max memory allocated
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# ---- 3. Loss & Potential Functions ----

def get_potential_func(beta_val):
    return lambda x, batch_indices=None: solver.get_potential_mixed_partial(
        x, y_obs=para.Y_OBS, obs_indices=para.OBS_INDICES, 
        sigma_noise=para.SIGMA_NOISE, batch_indices=batch_indices, beta=beta_val
    )

def compute_jeffreys_loss_generalized(model, batch_source, batch_target, u_source_func, u_target_func, lambda_0, lambda_1):
    # User requested exactly OBS_SIZE sensors must be used for all calculations
    random_sensors = torch.randperm(81, device=device)[:OBS_SIZE]

    # Energy KL
    z_fwd, log_det_fwd = model(batch_source, inverse=False)
    # Target(F(x)) - Source(x) - log|det|
    loss_energy = torch.mean(u_target_func(z_fwd, random_sensors) - u_source_func(batch_source, random_sensors) - log_det_fwd)

    # Data KL
    x_rec, log_det_bwd = model(batch_target, inverse=True)
    # Source(F^-1(y)) - Target(y) - log|det|
    loss_data = torch.mean(u_source_func(x_rec, random_sensors) - u_target_func(batch_target, random_sensors) - log_det_bwd)

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
    optimizer = optim.Adam(model.parameters(), lr=LR_F)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    data_size = mu_source.shape[0]
    num_batches = data_size // BATCH_SIZE
    model.train()

    t0 = time.time()
    for epoch in range(EPOCHS_F):
        perm_0 = torch.randperm(data_size, device=device)
        perm_1 = torch.randperm(data_size, device=device)
        avg_loss, avg_l_e, avg_l_d = 0.0, 0.0, 0.0

        for i in range(num_batches):
            idx_0 = perm_0[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            idx_1 = perm_1[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                loss, l_e, l_d = compute_jeffreys_loss_generalized(
                    model, mu_source[idx_0], mu_target[idx_1],
                    u_source_func, u_target_func, lambda_0, lambda_1
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            avg_loss += loss.item()
            avg_l_e += l_e.item()
            avg_l_d += l_d.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [Epoch {epoch + 1:2d}/{EPOCHS_F}] Loss: {avg_loss / num_batches:.4f} | E_KL: {avg_l_e / num_batches:.4f} | D_KL: {avg_l_d / num_batches:.4f}")
            
    t1 = time.time()
    print(f"  [Time] Flow trained in {t1-t0:.2f} seconds.")
    return model


def train_weight_function(step_k, model, mu_source, u_source_func, u_target_func):
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
        lr=LR_W
    )
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    data_size = mu_source.shape[0]
    num_batches = data_size // BATCH_SIZE

    # Ensure Flow is in Eval mode (fixed)
    model.eval()
    phi.train()
    psi.train()

    t0 = time.time()
    for epoch in range(EPOCHS_W):
        perm = torch.randperm(data_size, device=device)
        avg_loss = 0.0

        for i in range(num_batches):
            idx = perm[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            x0 = mu_source[idx]

            optimizer.zero_grad()

            # 1. Compute Flow Transform (No Grad for Flow)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                z, _ = model(x0, inverse=False)
                # Use MODE to determine if we use full sensors ('f') or stochastic sensors ('s') for weight function
                if MODE == 'f':
                    sensors = None
                else:
                    sensors = torch.randperm(81, device=device)[:OBS_SIZE]

                # U_k(F_k(x0))
                u_k_z = u_target_func(z, sensors)
                # U_{k-1}(x0)
                u_km1_x0 = u_source_func(x0, sensors)

            # 2. Compute Phi(x0) and Psi(z)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                phi_val = phi(x0).squeeze()
                psi_val = psi(z).squeeze()

                # 3. L2 Loss
                # Loss = E[ | U_k(z) - U_{k-1}(x) + phi(x) - psi(z) |^2 ]
                term = u_k_z - u_km1_x0 + phi_val - psi_val
                loss = torch.mean(term ** 2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            avg_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [Weight Epoch {epoch + 1:2d}/{EPOCHS_W}] L2 Loss: {avg_loss / num_batches:.6f}")

    t1 = time.time()
    print(f"  [Time] Weights trained in {t1-t0:.2f} seconds.")
    return phi, psi


# ---- 5. Main Execution ----

def run_training_pipeline():
    """
    Executes the full Jeffreys Flow training pipeline for Poisson data.
    """
    print(f"\n{'='*80}")
    print(f"  STARTING TRAINING PIPELINE")
    print(f"{'='*80}")
    
    # 1. Load MATLAB PT Targets and Beta List
    pt_data, BETA_LIST = load_data(device=device)

    # Validate Config
    if BETA_LIST is None:
        raise RuntimeError(f"BETA_LIST not found. Please ensure para.mat loaded correctly.")

    M = len(BETA_LIST) - 1
    # MALA configuration
    dt = MALA_DT
    n_iter = MALA_STEPS # Rejuvenation steps
    reju_batch_size = 100000 # Memory bounds limit

    print(f"[Config] Ladder Steps (M): {M}")
    beta_list_print = [f"{b.item():.3f}" for b in BETA_LIST]
    print(f"[Config] Beta Sequence: {beta_list_print}")
    print(f"[Config] MALA Rejuvenation: Enabled (Iter={n_iter}, dt={dt})")

    # 2. Initialize NU_0 (Base Distribution - Uniform Prior [0, 1]^8)
    print(f"\n[Initialization] Generating NU_0 (Uniform [0, 1], N={ensemble_size})...")
    # Using torch.rand for exact uniform generation over the domain
    nu_samples = torch.rand((ensemble_size, para.DIM), device=device)
    nu_log_weights = torch.zeros(ensemble_size, device=device)

    # Save Initial State as .mat
    scipy.io.savemat(
        os.path.join(DATA_DIR, f'samples_{MODE}_NU_0.mat'),
        {'samples': nu_samples.cpu().numpy()}
    )
    scipy.io.savemat(
        os.path.join(DATA_DIR, f'weights_{MODE}_NU_0.mat'),
        {'weights': nu_log_weights.cpu().numpy()}
    )
    print(f"  [IO] Saved NU_0 samples/weights to .mat")

    # 3. Sequential Training Loop
    for k in range(1, M + 1):
        curr_idx = str(k)

        beta_prev = BETA_LIST[k - 1].item()
        beta_curr = BETA_LIST[k].item()

        print(f"\n>>> PROCESSING STAGE {k} / {M} (Beta {beta_prev:.3f} -> {beta_curr:.3f})")

        # =========================================================
        # A. Prepare Training Source (mu_source)
        # =========================================================
        if curr_idx not in pt_data:
            print(f"Warning: Target data MU_{curr_idx} not found. Stopping pipeline.")
            break
            
        mu_target_pt = pt_data[curr_idx]  # Target for training (Data KL)

        print("  [Data Prep] Resampling nu_{k-1} to create training source...")
        current_weights = normalize_log_weights(nu_log_weights)

        mu_source = distribution_resample(
            samples=nu_samples,
            weights=current_weights,
            n_resamples=mu_target_pt.shape[0]  # Match size of PT target
        )

        # Rejuvenate Training Source (Target: U_{k-1})
        print(f"  [Data Prep] Rejuvenating training source (MALA, Iter={n_iter})...")
        mu_source = distribution_rejuvenation(
            solver, mu_source, lam=beta_prev,
            dt=dt, n_iterations=n_iter,
            batch_size=reju_batch_size
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
        # C. Train Weight Function (Phi, Psi) (Only for 'f' or 's')
        # =========================================================
        if MODE != 'e':
            phi, psi = train_weight_function(
                step_k=k, model=model_k, mu_source=nu_samples,
                u_source_func=u_prev_func, u_target_func=u_curr_func
            )
        else:
            print(f"\n" + "=" * 60)
            print(f"  SKIPPING WEIGHTS TRAINING for STEP {k} (MODE='e')")
            print("=" * 60)
            phi, psi = None, None

        # Save Checkpoint
        checkpoint = {
            'flow_state_dict': model_k.state_dict(),
            'phi_state_dict': phi.state_dict() if phi is not None else None,
            'psi_state_dict': psi.state_dict() if psi is not None else None,
            'step': k,
            'beta_prev': beta_prev,
            'beta_curr': beta_curr,
            'mode': MODE
        }
        checkpoint_path = os.path.join(DATA_DIR, f'flow_{MODE}_{k}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"  [IO] Saved Flow, Phi, Psi to {checkpoint_path}")

        # =========================================================
        # D. Propagate Ensemble (nu_{k-1} -> nu_k)
        # =========================================================
        print(f"  [Propagation] Applying Flow_{k} to NU_{k - 1}...")
        model_k.eval()
        if MODE != 'e':
            phi.eval()
            psi.eval()

        new_samples_list = []
        incremental_logw_list = []

        propag_batch_size = 10000
        num_gen_batches = int(np.ceil(ensemble_size / propag_batch_size))

        t0_prop = time.time()
        with torch.no_grad():
            for i in range(num_gen_batches):
                st, en = i * propag_batch_size, min((i + 1) * propag_batch_size, ensemble_size)
                x_prev = nu_samples[st:en]

                # 1. Flow Transport
                x_curr, log_det = model_k(x_prev, inverse=False)

                if MODE == 'e':
                    # EXACT Importance Weights Calculation (Uses full potential landscape)
                    u_k_z = u_curr_func(x_curr, None)
                    u_km1_x = u_prev_func(x_prev, None)
                    log_w = -u_k_z + u_km1_x + log_det
                else:
                    # 2. Compute Learned Weights
                    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                        phi_val = phi(x_prev).squeeze()
                        psi_val = psi(x_curr).squeeze()

                    # log w approx phi(x) - psi(z) + log_det
                    log_w = phi_val - psi_val + log_det

                new_samples_list.append(x_curr)
                incremental_logw_list.append(log_w)

        # Update Ensemble State
        nu_samples = torch.cat(new_samples_list, dim=0)
        incremental_log_w = torch.cat(incremental_logw_list, dim=0)
        
        # CESS Calculation (Log-Stable)
        log_alpha = nu_log_weights
        log_w = incremental_log_w
        
        term1 = torch.logsumexp(log_alpha + log_w, dim=0)
        term2 = torch.logsumexp(log_alpha + 2 * log_w, dim=0)
        term3 = torch.logsumexp(log_alpha, dim=0)
        
        log_cess = 2 * term1 - term2 - term3
        cess_ratio = torch.exp(log_cess).item()
        
        # Update weights
        nu_log_weights = nu_log_weights + incremental_log_w
        
        t1_prop = time.time()
        print(f"  [Propagation] Time: {t1_prop-t0_prop:.2f} seconds.")

        # =========================================================
        # E. Evaluate Metrics
        # =========================================================
        ess_ratio = compute_ESS(normalize_log_weights(nu_log_weights))
        print(f"  [Metrics] CESS (Step): {cess_ratio:.2%} | ESS (Total): {ess_ratio:.2%}")

        # =========================================================
        # F. Adaptive Resampling & Rejuvenation
        # =========================================================
        if ess_ratio < 0.5:
            print(f"  [Adaptive] ESS < 50%. Performing Resampling (Weights reset)...")
            w_norm = normalize_log_weights(nu_log_weights)
            nu_samples = distribution_resample(
                nu_samples,
                weights=w_norm,
                n_resamples=ensemble_size
            )
            nu_log_weights = torch.zeros(ensemble_size, device=device)
            ess_ratio = 1.0

        print(f"  [Adaptive] Performing Unconditional Rejuvenation (Target: Beta {beta_curr:.3f}) Iter={n_iter}...")
        t0_reju = time.time()
        nu_samples = distribution_rejuvenation(
            solver, nu_samples, lam=beta_curr,
            dt=dt, n_iterations=n_iter,
            batch_size=reju_batch_size
        )
        t1_reju = time.time()
        print(f"  [Adaptive] Rejuvenation time: {t1_reju-t0_reju:.2f} seconds. Final ESS Status: {ess_ratio:.2%}")

        # =========================================================
        # G. Save and Monitor Memory
        # =========================================================
        scipy.io.savemat(
            os.path.join(DATA_DIR, f'samples_{MODE}_NU_{k}.mat'),
            {'samples': nu_samples.cpu().numpy()}
        )
        scipy.io.savemat(
            os.path.join(DATA_DIR, f'weights_{MODE}_NU_{k}.mat'),
            {'weights': nu_log_weights.cpu().numpy(), 'cess': cess_ratio, 'ess': ess_ratio}
        )
        
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"  [System] GPU Max Memory allocated: {max_mem:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

    print(f"\n[Pipeline] Training completed successfully.")


if __name__ == "__main__":
    run_training_pipeline()
