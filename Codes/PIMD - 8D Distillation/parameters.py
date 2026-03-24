import torch
import os

from potential import Path_Integral_Potential

torch.manual_seed(27)

# Data Set Parameters
N = 8 # number of Fourier modes
BETA_TARGET = 1.0
M = 10 # temperature ladder size
MU_SIZE = 100000 # MU data set size
NU_SIZE = 200000 # NU data set size 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

FIG_DIR = os.path.join(BASE_DIR, 'figures')
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

# Flow Training Parameters
EPOCHS = 20
BATCH_SIZE = 2000
LR = 1e-3

HIDDEN_DIM = 256
NUM_LAYERS = 6
NUM_BINS = 32

pt = Path_Integral_Potential(N=N)