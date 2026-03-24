import torch
import os
import matplotlib.pyplot as plt
from parameters import *
from utilities import distribution_resample, normalize_log_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    nu_s1_path = os.path.join(DATA_DIR, "samples_NU_S1.pth")
    weights_s1_path = os.path.join(DATA_DIR, "weights_NU_S1.pth")
    mu_s1_path = os.path.join(DATA_DIR, "samples_MU_S1.pth")

    if not (os.path.exists(nu_s1_path) and os.path.exists(weights_s1_path)):
        raise RuntimeError(f"Missing NU_S1 samples or weights at {DATA_DIR}")

    print("[Config] Loading NU_S1 generated data...")
    nu_samples_all = torch.load(nu_s1_path, map_location=device) # [M+1, NU_SIZE, N]
    weights_data = torch.load(weights_s1_path, map_location=device)
    nu_weights_all = weights_data['weights'] # [M+1, NU_SIZE]

    M_layers = nu_samples_all.shape[0]
    
    # 1. Uniformly resample NU_S1 distributions to MU arrays
    mu_s2_samples = torch.zeros(M_layers, MU_SIZE, N, device=device)
    
    for k in range(M_layers):
        print(f"  > Uniform Resampling stage {k} / {M_layers-1}...")
        w_norm = normalize_log_weights(nu_weights_all[k])
        
        mu_s2_samples[k] = distribution_resample(
            samples=nu_samples_all[k],
            weights=w_norm,
            n_resamples=MU_SIZE
        )

    # Export directly to data/samples_MU_S2.pth
    save_path = os.path.join(DATA_DIR, "samples_MU_S2.pth")
    torch.save(mu_s2_samples.cpu(), save_path)
    print(f"[Export] Saved MU_S2 samples to {save_path} with shape {mu_s2_samples.shape}")
    
    # 2. Extract final target distributions (k=M) exclusively showing Mode 0 (xi_0)
    print("[Analysis] Comparing quantum vs classical deviations...")
    if not os.path.exists(mu_s1_path):
        print(f"  > MU_S1 reference missing at {mu_s1_path}. Skipping overlay plot.")
        return

    mu_s1_samples = torch.load(mu_s1_path, map_location=device)
    
    xi0_s1 = mu_s1_samples[-1, :, 0].detach().cpu().numpy()
    xi0_s2 = mu_s2_samples[-1, :, 0].detach().cpu().numpy()
    
    plt.figure(figsize=(6, 4))
    plt.hist(xi0_s1, bins=150, density=True, alpha=0.5, label='Classical', color='blue')
    plt.hist(xi0_s2, bins=150, density=True, alpha=0.5, label='Quantum', color='orange')
    plt.title('Distribution of Centroid Mode ($\\xi_0$)')
    plt.xlabel('$\\xi_0$')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fig_path = os.path.join(FIG_DIR, "compare_MU_S1_S2.pdf")
    plt.savefig(fig_path, dpi=600)
    plt.close()
    
    print(f"[Visual] Plot saved to {fig_path}")

if __name__ == "__main__":
    main()
