import torch
import os

from potential import Path_Integral_Potential

torch.manual_seed(27)

# Data Set Parameters
N0 = 8  # number of Fourier modes (low-frequenccy)
N  = 12 # number of Fourier modes (full)
BETA_TARGET = 1.0
M = 10 # temperature ladder size
MU_SIZE = 100000 # MU data set size
NU_SIZE = 500000 # NU data set size 

# Flow Architecture
HIDDEN_DIM = 256
NUM_LAYERS = 6
NUM_BINS = 32

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

FIG_DIR = 'figures'
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

pt = Path_Integral_Potential(N=N)