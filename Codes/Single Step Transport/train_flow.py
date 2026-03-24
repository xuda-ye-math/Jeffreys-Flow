# ---- 1. Header and Imports ----
from parameters import *  # Imports para, EPOCHS, LOAD_DATA, etc.
from model import Normalizing_Flow
import torch
import torch.optim as optim
import numpy as np
import os

# ---- 2. Configuration ----
# Seed
SEED = 29
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {device}")

# Sync Potential Helper with current device
# 'para' is initialized in parameters.py based on ABBR (TW or HB)
para.device = str(device)
print(f"[Config] Potential System: {para.NAME} ({para.ABBR})")


# ---- 3. Helper Functions ----

def gaussian_potential(x):
    """
    Computes U_0(x) for the Gaussian Base Distribution.
    U_0(x) = 0.5 * ||(x - mu)/sigma||^2

    Uses global 'para' for Mean and Sigma.
    """
    mean = para.BASE_MEAN
    sigma = para.BASE_SIGMA
    return 0.5 * torch.sum(((x - mean) / sigma) ** 2, dim=1)


def compute_jeffreys_loss(model, batch_0, batch_1, lambda_0, lambda_1):
    """
    Computes the Jeffreys Divergence Loss:
    L_J = lambda_0 * KL(F#mu_0 || pi_1) + lambda_1 * KL(pi_1 || F#mu_0)
    """

    # --- 1. Energy KL (Backward): Minimize KL(F#mu_0 || pi_1) ---
    # Loss ~ E_{x~mu_0} [ U_1(F(x)) - U_0(x) - log|det J_F(x)| ]

    z_fwd, log_det_fwd = model(batch_0, inverse=False)

    # Use para.compute_potential for the generic target potential
    u1_z = para.compute_potential(z_fwd)
    u0_x = gaussian_potential(batch_0)

    loss_energy = torch.mean(u1_z - u0_x - log_det_fwd)

    # --- 2. Data KL (Forward): Minimize KL(pi_1 || F#mu_0) ---
    # Loss ~ E_{y~mu_1} [ U_0(F^-1(y)) - U_1(y) + log|det J_{F^-1}(y)| ]

    x_rec, log_det_bwd = model(batch_1, inverse=True)

    u0_rec = gaussian_potential(x_rec)
    u1_y = para.compute_potential(batch_1)

    loss_data = torch.mean(u0_rec - u1_y - log_det_bwd)

    # --- 3. Weighted Sum ---
    loss_total = lambda_0 * loss_energy + lambda_1 * loss_data

    return loss_total, loss_energy, loss_data


def train_flow_strategy(save_index, theta, mu_0, mu_1, var_0, var_1):
    """
    Executes training for a specific Theta value and saves with index.

    Args:
        save_index (int): Index for file naming (0-4).
        theta (float): Balancing parameter [0, 1].
        mu_0, mu_1 (Tensor): Training data.
        var_0, var_1 (float): Variances for scaling.
    """
    print(f"\n" + "=" * 60)
    print(f"  STARTING TRAINING SCHEME: Index {save_index} (Theta={theta:.2f})")
    print("=" * 60)

    # 1. Calculate Hyperparameters based on Note Eq 179-180
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
    # Naming convention: {ABBR}_flow_{index}.pth
    # Uses para.ABBR which comes from parameters.py
    save_name = f"{para.ABBR}_flow_{save_index}.pth"
    save_path = os.path.join(DATA_DIR, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"  >> Model saved to: {save_path}")


# ---- 4. Main Execution ----

if __name__ == "__main__":
    # 1. Load Data
    # Uses load_data from parameters.py
    mu_0, mu_1 = load_data(device=device)

    # 2. Compute Variances (Trace of Covariance)
    # Var(mu) = Sum of variances across dimensions
    var_0 = torch.var(mu_0, dim=0).sum().item()
    var_1 = torch.var(mu_1, dim=0).sum().item()

    print(f"[Stats] Var(mu_0): {var_0:.4f}")
    print(f"[Stats] Var(mu_1): {var_1:.4f}")

    # 3. Define Thetas
    # 0: Pure Data KL, 1: Pure Energy KL
    thetas = [0.0, 0.25, 0.50, 0.75, 1.00]

    # 4. Run Experiments
    for i, theta in enumerate(thetas):
        train_flow_strategy(i, theta, mu_0, mu_1, var_0, var_1)

    print("\n[Main] All training schemes completed successfully.")