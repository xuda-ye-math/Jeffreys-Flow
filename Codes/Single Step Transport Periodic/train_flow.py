# ---- 1. Header and Imports ----
from parameters import *  # Imports para, EPOCHS, LOAD_DATA, etc.
from model import Normalizing_Flow
import torch
import torch.optim as optim
import numpy as np
import os
import math

# ---- 2. Configuration ----
# Seed
SEED = 29
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")

# Sync Potential Helper with current device
# 'para' is initialized in parameters.py based on ABBR
para.device = str(device)
print(f"[Config] Potential System: {para.NAME} ({para.ABBR})")


# ---- 3. Helper Functions ----

def base_potential(x):
    """
    Computes U_0(x) for the Base Distribution.
    For Periodic Well, the base is Uniform on [-pi, pi]^d.

    p(x) = 1 / Volume = 1 / (2*pi)^d
    U_0(x) = -log(p(x)) = log((2*pi)^d) = d * log(2*pi)
    """
    dim = para.DIM
    # Volume of hypercube [-pi, pi]^d is (2*pi)^d
    # U_0 = log(Volume)
    u0_val = dim * math.log(2 * math.pi)

    # Return tensor of shape [Batch_Size]
    return torch.full((x.shape[0],), u0_val, device=x.device)


def compute_jeffreys_loss(model, batch_0, batch_1, lambda_0, lambda_1):
    """
    Computes the Jeffreys Divergence Loss:
    L_J = lambda_0 * KL(F#mu_0 || pi_1) + lambda_1 * KL(pi_1 || F#mu_0)
    """

    # --- 1. Energy KL (Backward): Minimize KL(F#mu_0 || pi_1) ---
    z_fwd, log_det_fwd = model(batch_0, inverse=False)

    u1_z = para.compute_potential(z_fwd)
    u0_x = base_potential(batch_0)

    loss_energy = torch.mean(u1_z - u0_x - log_det_fwd)

    # --- 2. Data KL (Forward): Minimize KL(pi_1 || F#mu_0) ---
    x_rec, log_det_bwd = model(batch_1, inverse=True)

    u0_rec = base_potential(x_rec)
    u1_y = para.compute_potential(batch_1)

    loss_data = torch.mean(u0_rec - u1_y - log_det_bwd)

    # --- 3. Weighted Sum ---
    loss_total = lambda_0 * loss_energy + lambda_1 * loss_data

    return loss_total, loss_energy, loss_data


def train_flow_single(theta, mu_0, mu_1, var_0, var_1):
    """
    Executes training for a single Theta value (0.5) and saves without index.

    Args:
        theta (float): Balancing parameter (fixed to 0.5).
        mu_0, mu_1 (Tensor): Training data.
        var_0, var_1 (float): Variances for scaling.
    """
    print(f"\n" + "=" * 60)
    print(f"  STARTING TRAINING: Jeffreys Flow (Theta={theta:.2f})")
    print("=" * 60)

    # 1. Calculate Hyperparameters based on Variance Scaling
    lambda_0 = theta * var_0
    lambda_1 = (1.0 - theta) * var_1

    print(f"  > Lambda_0: {lambda_0:.4f}")
    print(f"  > Lambda_1: {lambda_1:.4f}")

    # 2. Initialize Model
    model = Normalizing_Flow().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    data_size = mu_0.shape[0]
    num_batches = data_size // BATCH_SIZE

    # 3. Training Loop
    model.train()

    for epoch in range(EPOCHS):
        # Shuffle
        perm_0 = torch.randperm(data_size)
        perm_1 = torch.randperm(data_size)

        avg_loss = 0.0
        avg_l_e = 0.0
        avg_l_d = 0.0

        for i in range(num_batches):
            idx_0 = perm_0[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            idx_1 = perm_1[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]

            batch_0 = mu_0[idx_0]
            batch_1 = mu_1[idx_1]

            optimizer.zero_grad()

            loss, l_e, l_d = compute_jeffreys_loss(model, batch_0, batch_1, lambda_0, lambda_1)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            avg_l_e += l_e.item()
            avg_l_d += l_d.item()

        # Stats
        avg_loss /= num_batches
        avg_l_e /= num_batches
        avg_l_d /= num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  [Epoch {epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f} | Energy KL: {avg_l_e:.4f} | Data KL: {avg_l_d:.4f}")

    # 4. Save Model
    # Naming convention: {ABBR}_flow.pth (No index)
    save_name = f"{para.ABBR}_flow.pth"
    save_path = os.path.join(DATA_DIR, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"  >> Model saved to: {save_path}")


# ---- 4. Main Execution ----

if __name__ == "__main__":
    # 1. Load Data
    mu_0, mu_1 = load_data(device=device)

    # 2. Compute Variances (Trace of Covariance)
    var_0 = torch.var(mu_0, dim=0).sum().item()
    var_1 = torch.var(mu_1, dim=0).sum().item()

    print(f"[Stats] Var(mu_0): {var_0:.4f}")
    print(f"[Stats] Var(mu_1): {var_1:.4f}")

    # 3. Single Run with Theta = 0.5
    theta_fixed = 0.50
    train_flow_single(theta_fixed, mu_0, mu_1, var_0, var_1)

    print("\n[Main] Training completed successfully.")