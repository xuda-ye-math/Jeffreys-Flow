import torch
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from parameters import *
from potential import Path_Integral_Potential

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Globally establish pt object to link potential formulas parametrically
pt = Path_Integral_Potential(N=N, beta=BETA_TARGET).to(device)

def main():
    save_path = os.path.join(DATA_DIR, "samples_MU_S1.pth")
    if os.path.exists(save_path):
        print(f"Data file {save_path} already exists. Skipping generation.")
    else:
        # Lambda schedule: shape [M+1, 1] for broadcasting against [M+1, MU_SIZE]
        lam = torch.linspace(0, 1, M+1, device=device).view(M+1, 1)
        
        # Initialize y = xi_0 / sqrt(BETA). This matches the marginal coordinate we are simulating.
        # Start with standard normal for MU_SIZE independent samples across M+1 interpolations.
        y = torch.randn(M+1, MU_SIZE, device=device)
        y.requires_grad_(True)
    
        dt = 5e-3
        steps = 20000
        swap_interval = 20
        
        print(f"Running Tamed Euler with Parallel Tempering on {device}...")
        for step in tqdm(range(steps)):
            # Calculate Energies and Forces
            energy = (1.0 - lam) * pt.base_V(y) + lam * pt.target_V(y)
            force = -torch.autograd.grad(energy.sum(), y)[0]
            
            # Tamed Euler integration step
            tamed_force = force * dt / (1.0 + dt * torch.abs(force))
            noise = math.sqrt(2.0 * dt) * torch.randn_like(y)
            
            with torch.no_grad():
                y_new = y + tamed_force + noise
                
                # Parallel Tempering (Replica Exchange)
                if step % swap_interval == 0:
                    # Perform swaps for even, then odd adjacent pairs
                    for parity in [0, 1]:
                        for i in range(parity, M, 2):
                            # Energies for current positions and swapped positions
                            E1 = (1.0 - lam[i]) * pt.base_V(y_new[i]) + lam[i] * pt.target_V(y_new[i])
                            E2 = (1.0 - lam[i+1]) * pt.base_V(y_new[i+1]) + lam[i+1] * pt.target_V(y_new[i+1])
                            E1_swap = (1.0 - lam[i]) * pt.base_V(y_new[i+1]) + lam[i] * pt.target_V(y_new[i+1])
                            E2_swap = (1.0 - lam[i+1]) * pt.base_V(y_new[i]) + lam[i+1] * pt.target_V(y_new[i])
                            
                            # Swap probability criterion: min(1, exp(-Delta))
                            delta = E1_swap + E2_swap - E1 - E2
                            rand_val = torch.rand_like(delta)
                            swap_mask = torch.log(rand_val) < -delta
                            
                            # Execute swap where accepted
                            tmp = y_new[i, swap_mask].clone()
                            y_new[i, swap_mask] = y_new[i+1, swap_mask]
                            y_new[i+1, swap_mask] = tmp

            # Setup for next iteration
            y = y_new.detach().clone()
            y.requires_grad_(True)

        print("Simulation complete. Extending data to N Fourier modes...")
        with torch.no_grad():
            xi = torch.zeros(M+1, MU_SIZE, N, device=device)
            
            # Extended data: xi_0 = y * sqrt(BETA)
            xi[:, :, 0] = y * math.sqrt(BETA_TARGET)
            
            # Remaining modes xi_k ~ N(0, 1/OMEGA[k]^2) 
            omega = pt.OMEGA
            for k in range(1, N):
                xi[:, :, k] = torch.randn(M+1, MU_SIZE, device=device) / omega[k]
            
            # Output and save
            torch.save(xi.cpu(), save_path)
            print(f"Data saved to {save_path} with shape {xi.shape}")

    # Plot the distributions of all 8 modes for the target temperature (lam = 1)
    print("Plotting distributions for the target temperature...")
    data = torch.load(save_path)
    target_data = data[-1].numpy()  # Extract the last temperature index [MU_SIZE, N]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Marginal Distributions of Fourier Modes (Target Temperature)", fontsize=16)
    
    for i in range(N):
        ax = axes[i // 4, i % 4]
        ax.hist(target_data[:, i], bins=100, density=True, alpha=0.7, color='blue')
        ax.set_title(f"Mode $\\xi_{i}$")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "samples_MU_S1.png")
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")

if __name__ == '__main__':
    main()