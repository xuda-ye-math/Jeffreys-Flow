import numpy as np
import torch
import os
import re
import scipy.io

# ---- 1. Header and Seed ----
SEED = 27
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- 2. General Configuration ----
ABBR = 'GM'
NU_SIZE = 400000

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

FIG_DIR = 'figures'
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

# ---- 3. Method Hyperparameters ----
EPOCHS = 20
BATCH_SIZE = 2000
LR = 5e-4
WEIGHT_DECAY = 0.0

HIDDEN_DIM = 256
NUM_LAYERS = 6
NUM_BINS = 16
DROPOUT = 0.0

REJU_MAX_SIZE = 50000


# ---- 4. Data Loading (Automated Interface) ----

def load_data(device='cpu'):
    """
    Scans the DATA_DIR for all relevant .mat files and loads them into memory.
    Also loads the BETA_LIST if available.

    Returns:
        data_dict: Dictionary {str: Tensor [N, 3]} containing sample data.
        beta_list: Tensor [M+1] containing the lambda/beta schedule, or None if not found.
    """
    print(f"[Data] Scanning and loading all MATLAB data from {DATA_DIR}...")

    # 1. Load Samples
    # Pattern to match {ABBR}_samples_MU_{index}.mat
    pattern = rf"^{ABBR}_samples_MU_(.+)\.mat$"
    data_dict = {}

    # List all files in the directory
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]

    if not files:
        raise FileNotFoundError(f"No .mat files found in {DATA_DIR}")

    for filename in files:
        match = re.match(pattern, filename)
        if match:
            idx_label = match.group(1)  # Extract '0', '1', ..., 'M'
            file_path = os.path.join(DATA_DIR, filename)

            # Load MATLAB file
            mat_contents = scipy.io.loadmat(file_path)

            # The key inside .mat is typically 'samples_MU_{idx}'
            mat_key = f'samples_MU_{idx_label}'
            if mat_key in mat_contents:
                samples_np = mat_contents[mat_key]
                # Convert to Tensor [N, 3] and move to device
                samples_tensor = torch.from_numpy(samples_np).float().to(device)
                data_dict[idx_label] = samples_tensor
                print(f"       Loaded MU_{idx_label}: {samples_tensor.shape}")
            else:
                print(f"       Warning: Key '{mat_key}' not found in {filename}")

    # Sorting for log output
    sorted_keys = sorted(data_dict.keys(), key=lambda x: (int(x) if x.isdigit() else float('inf')))
    print(f"[Data] Successfully loaded {len(data_dict)} stages: {sorted_keys}")

    # 2. Load Beta List
    beta_filename = f"{ABBR}_BETA_LIST.mat"
    beta_path = os.path.join(DATA_DIR, beta_filename)
    beta_list = None

    if os.path.exists(beta_path):
        try:
            mat_contents = scipy.io.loadmat(beta_path)
            if 'BETA_LIST' in mat_contents:
                beta_np = mat_contents['BETA_LIST']
                # Ensure it's a 1D flat tensor
                beta_list = torch.from_numpy(beta_np).float().view(-1).to(device)
                print(f"[Data] Loaded BETA_LIST from {beta_filename}: {beta_list.shape} elements")
            else:
                print(f"[Data] Warning: 'BETA_LIST' variable not found in {beta_filename}")
        except Exception as e:
            print(f"[Data] Error loading {beta_filename}: {e}")
    else:
        print(f"[Data] Warning: Beta list file {beta_filename} not found.")

    return data_dict, beta_list


# ---- 5. Initialize Parameters ----
from potential.GMM_3D import GMM_3D

# Initialize Parameter Object (Configuration is now separate or loaded dynamically)
# Note: Beta schedule is no longer initialized here; it should be loaded via load_data()
# and assigned to para.BETA_LIST in the main script if needed.
para = GMM_3D()