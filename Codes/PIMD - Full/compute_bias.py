import numpy as np
import scipy.linalg as la
from parameters import N, BETA_TARGET

def V(x):
    return 20.0 * ((x - 1.0)**2) * (x + 0.9) * (x + 1.1)

# Discretization parameters
L = 6.0
K = 5000
x = np.linspace(-L, L, K)
dx = x[1] - x[0]

# Kinetic energy operator (-1/2 d^2/dx^2) using 2nd order finite diff
diag = np.ones(K) / (dx**2)
off_diag = -0.5 * np.ones(K-1) / (dx**2)
H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

# Potential energy operator
H += np.diag(V(x))

# Eigen decomposition
print(f"Performing eigendecomposition on {K}x{K} Hamiltonian...", flush=True)
evals, evecs = la.eigh(H)

# Check stability using minimum eigenvalue
evals_shifted = evals - np.min(evals)

# Compute thermal partition weights
beta = BETA_TARGET
weights = np.exp(-beta * evals_shifted)
Z = np.sum(weights)
probs = weights / Z

# Define observable functions
def O1(x): return np.sin(x)
def O2(x): return np.cos(x)
def O3(x): return np.exp(-x**2 / 2.0)
def O4(x): return np.exp(-(x - 1.0)**2 / 2.0)
def O5(x): return np.exp(-(x + 1.0)**2)

observables = [O1, O2, O3, O4, O5]

averages = []
for O in observables:
    # Evaluate observable on grid
    O_x = O(x)
    
    # Compute <psi_n | O | psi_n> for each eigenvalue n
    O_n = np.sum(O_x[:, None] * (evecs ** 2), axis=0) # Shape: [K]
    
    # Sum over all states: sum_n p_n * <psi_n | O | psi_n>
    avg = np.sum(probs * O_n)
    averages.append(float(avg))

print("Quantum thermal averages (Exact Analytical):")
print(averages)

import os
import torch
import json
from parameters import pt, DATA_DIR

print("\nEvaluating observables from Flow Samples...")

# Load flow samples and weights
samples_path = os.path.join(DATA_DIR, f'samples_NU_N{N}.pth')
weights_path = os.path.join(DATA_DIR, f'weights_NU_N{N}.pth')

if not os.path.exists(samples_path) or not os.path.exists(weights_path):
    print(f"Flow samples for N={N} not found.")
else:
    # [M+1, NU_SIZE, N]
    samples_all = torch.load(samples_path, map_location='cpu')
    weights_all = torch.load(weights_path, map_location='cpu')['weights']
    
    # We evaluate the target distribution samples (at the last ladder index M)
    M_idx = samples_all.shape[0] - 1
    
    xi_samples = samples_all[M_idx] # [NU_SIZE, N]
    log_weights = weights_all[M_idx] # [NU_SIZE]
    
    # Convert weights to normalized probabilities
    max_log_w = torch.max(log_weights)
    log_sum_exp = torch.logsumexp(log_weights - max_log_w, dim=0) + max_log_w
    norm_weights = torch.exp(log_weights - log_sum_exp) # [NU_SIZE]
    
    # Transform Fourier modes Space -> Position Space
    # pt.f2x outputs [NU_SIZE, N] positions
    x_samples = pt.cpu().f2x(xi_samples)
    
    flow_averages = []
    
    for O in observables:
        # Evaluate O(x) on all beads [NU_SIZE, N]
        O_val_beads = O(x_samples.numpy())
        
        # Average over all beads of the ring polymer [NU_SIZE]
        O_val_samples = np.mean(O_val_beads, axis=1)
        
        # Weighted expectation over the ensemble
        O_val_ensemble = np.sum(O_val_samples * norm_weights.numpy())
        flow_averages.append(float(O_val_ensemble))
        
    print(f"Quantum thermal averages (Flow Samples N={N}):")
    print(flow_averages)
    
    # Compute and calculate biases
    biases = [abs(ex - fl) for ex, fl in zip(averages, flow_averages)]
    print(f"Absolute Biases for N={N}:")
    print(biases)
    
    # Save to JSON record
    record_file = os.path.join(DATA_DIR, 'bias_record.json')
    record_data = {}
    
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            try:
                record_data = json.load(f)
            except:
                pass
                
    # Update record for current N
    record_data[str(N)] = biases
    
    with open(record_file, 'w') as f:
        json.dump(record_data, f, indent=4)
    print(f"\nBias record for N={N} successfully written to {record_file}.")
