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

# Resolve absolute paths based on the location of parameters.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

FIG_DIR = os.path.join(BASE_DIR, 'figures')
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

# ---- 3. Method Hyperparameters ----
MU_SIZE = 100000 # number of samples in MU
NU_SIZE = 400000 # number of samples in NU

EPOCHS_F = 100  # epochs for flow training
EPOCHS_W = 1000  # epochs for weight training
BATCH_SIZE = 20000
LR_F = 5e-4
LR_W = 2e-3

HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_BINS = 32

MALA_DT = 1e-5
MALA_STEPS = 100
OBS_SIZE = 9
MODE = 's'  # 'f' for full potential, 's' for stochastic, 'e' for exact weights (no Network)

# ---- 4. Loading MATLAB Configuration ----
class ParameterConfig:
    def __init__(self):
        para_path = os.path.join(DATA_DIR, 'para.mat')
        if os.path.exists(para_path):
            print(f"[Config] Loading MATLAB parameters from {para_path}...")
            # Use squeeze_me to simplify scalars
            mat_contents = scipy.io.loadmat(para_path, squeeze_me=True, struct_as_record=False)
            para_mat = mat_contents['para']
            
            # Extract common parameters
            self.NAME = "2D Screened Poisson"
            self.DIM = int(para_mat.dim)
            self.BETA_LIST = torch.tensor(para_mat.beta_list).float()
            self.M = int(para_mat.M)
            self.TOTAL_SAMPLES = int(para_mat.total_samples)
            self.N_CELLS = int(para_mat.N_cells)
            
            # Physics Constraints
            self.ALPHA = float(para_mat.alpha)
            self.C = float(para_mat.c)
            self.GAMMA = float(para_mat.gamma)
            
            # Observations
            self.SIGMA_NOISE = float(para_mat.sigma_noise)
            self.Y_OBS = torch.tensor(para_mat.y_obs).float()
            self.OBS_INDICES = torch.tensor(para_mat.obs_indices).long() # 1-indexed from MATLAB
            self.THETA_TRUE = torch.tensor(para_mat.theta_true).float()
            
            # Rejuvenation parameters (MALA)
            self.DT = float(para_mat.dt)
            self.N_ITER = 100 # Default MALA Steps Local configuration
            self.NOISE = 1.0
            
            # Limits and Bounds
            self.X_LIM_COMPUTE = [0.0, 1.0]
            self.Y_LIM_COMPUTE = [0.0, 1.0]
            
            # Base Configuration (for Uninformed Prior initialization)
            self.BASE_MEAN = 0.5
            self.BASE_SIGMA = 0.5 / 3.0 # Approx boundaries 99% within [0, 1]
            
        else:
            print(f"[Config] Warning: {para_path} not found. Please run simulated parameters in MATLAB first.")
            self.DIM = 8
            self.BETA_LIST = None
            self.TOTAL_SAMPLES = 100000
            
        # Bind device dynamically
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        if hasattr(self, 'BETA_LIST') and self.BETA_LIST is not None:
            self.BETA_LIST = self.BETA_LIST.to(device)
        if hasattr(self, 'Y_OBS'):
            self.Y_OBS = self.Y_OBS.to(device)
        if hasattr(self, 'OBS_INDICES'):
            self.OBS_INDICES = self.OBS_INDICES.to(device)
        if hasattr(self, 'THETA_TRUE'):
            self.THETA_TRUE = self.THETA_TRUE.to(device)

# Instantiate Global Config
para = ParameterConfig()

# Setup Data Sizes
if hasattr(para, 'TOTAL_SAMPLES'):
    NU_SIZE = para.TOTAL_SAMPLES
    REJU_MAX_SIZE = NU_SIZE
else:
    NU_SIZE = 100000
    REJU_MAX_SIZE = NU_SIZE


# ---- 5. Data Loading (Automated Interface) ----
def load_data(device='cpu'):
    '''
    Scans the DATA_DIR for all relevant simulated .mat files from MATLAB and loads them into memory.
    Supports MATLAB v7.3 .mat files via h5py.
    
    Args:
        device: Torch device ('cpu', 'cuda').
        
    Returns:
        data_dict: Dictionary {str: Tensor [N, DIM]} containing sample data for each Beta.
        beta_list: Tensor [M+1] containing the lambda/beta schedule.
    '''
    import h5py
    print(f"[Data] Scanning and loading MATLAB simulated data from {DATA_DIR} ...")
    
    # regex rule for sample file names
    pattern = r"^samples_MU_(.+)\.mat$"
    data_dict = {}
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"{DATA_DIR} directory does not exist.")
        
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]
    if not files:
        raise FileNotFoundError(f"No .mat files found in {DATA_DIR}")
        
    for filename in files:
        match = re.match(pattern, filename)
        if match:
            idx_label = match.group(1)
            file_path = os.path.join(DATA_DIR, filename)
            
            try:
                # MATLAB v7.3 files require h5py
                with h5py.File(file_path, 'r') as f:
                    mat_key = f'samples_MU_{idx_label}'
                    
                    if mat_key in f:
                        # h5py reads MATLAB arrays transposed compared to scipy.io
                        # MATLAB [N, DIM] -> h5py [DIM, N]
                        samples_np = np.array(f[mat_key]).T 
                        samples_tensor = torch.from_numpy(samples_np).float().to(device)
                        data_dict[idx_label] = samples_tensor
                        print(f"       Loaded MU_{idx_label}: {samples_tensor.shape}")
                    else:
                        print(f"       Warning: Key '{mat_key}' not found in {filename}")
            except OSError as e:
                # Fallback to scipy if it happens to be not v7.3
                try:
                    mat_contents = scipy.io.loadmat(file_path)
                    mat_key = f'samples_MU_{idx_label}'
                    if mat_key in mat_contents:
                        samples_np = mat_contents[mat_key]
                        samples_tensor = torch.from_numpy(samples_np).float().to(device)
                        data_dict[idx_label] = samples_tensor
                        print(f"       Loaded MU_{idx_label} (scipy fallback): {samples_tensor.shape}")
                except Exception as sc_e:
                    print(f"       Error loading {filename} via both h5py and scipy. ({e} | {sc_e})")
            except Exception as e:
                print(f"       Error loading {filename}: {e}")
                
    sorted_keys = sorted(data_dict.keys(), key=lambda x: (int(x) if x.isdigit() else float('inf')))
    print(f"[Data] Successfully loaded {len(data_dict)} stages: {sorted_keys}")
    
    if hasattr(para, 'BETA_LIST') and para.BETA_LIST is not None:
        beta_list = para.BETA_LIST.to(device)
    else:
        beta_list = None
        
    return data_dict, beta_list

if __name__ == "__main__":
    print(f"--- Verification: testing parameters.py Loading Protocol ---")
    para.to('cpu')
    print(f"Loaded Dimension: {para.DIM}")
    print(f"Loaded Total Samples expected: {para.TOTAL_SAMPLES}")
    if para.BETA_LIST is not None:
        print(f"Beta List: {para.BETA_LIST.cpu().numpy()}")
    
    try:
        data_dict, beta_list = load_data('cpu')
        print("Data Loading successful.")
    except Exception as e:
        print(f"Skipping loading test because data might not be generated yet. Error: {e}")
