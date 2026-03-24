import torch
import os
import matplotlib.pyplot as plt
from parameters import DATA_DIR, FIG_DIR, M

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    weights_s1_path = os.path.join(DATA_DIR, "weights_NU_S1.pth")
    weights_s2_path = os.path.join(DATA_DIR, "weights_NU_S2.pth")

    if not (os.path.exists(weights_s1_path) and os.path.exists(weights_s2_path)):
        raise RuntimeError(f"Missing weights_NU_S1.pth or weights_NU_S2.pth at {DATA_DIR}")

    print("[Config] Loading NU_S1 and NU_S2 weights data...")
    weights_data_s1 = torch.load(weights_s1_path, map_location=device)
    weights_data_s2 = torch.load(weights_s2_path, map_location=device)

    # CESS arrays: shape [M+1] (index 0 is base dist, 1 to M are flow steps)
    cess_s1 = weights_data_s1['cess'].cpu().numpy()
    cess_s2 = weights_data_s2['cess'].cpu().numpy()

    # We only care about the M steps (k=1 to M)
    steps = range(1, M + 1)
    cess_s1_steps = cess_s1[1:M+1] * 100.0  # Convert to percentage
    cess_s2_steps = cess_s2[1:M+1] * 100.0

    print("\n[Analysis] CESS Evaluation across Flow Layers:")
    print("-" * 45)
    print(f"{'Step (k)':<15} | {'Flow S1 (%)':<12} | {'Flow S2 (%)':<12}")
    print("-" * 45)
    for k in steps:
        print(f"{k:<15} | {cess_s1_steps[k-1]:<12.2f} | {cess_s2_steps[k-1]:<12.2f}")
    print("-" * 45)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(steps, cess_s1_steps, marker='o', linestyle='-', color='blue', label='Flow (Classical)')
    plt.plot(steps, cess_s2_steps, marker='s', linestyle='-', color='orange', label='Flow (Quantum)')
    
    plt.axhline(y=100.0, color='gray', linestyle='--', alpha=0.5, label='100% CESS')
    
    plt.title('CESS per Flow Step')
    plt.xlabel('Ladder Step (k)')
    plt.ylabel('CESS (%)')
    plt.xticks(steps)
    plt.ylim(85, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fig_path = os.path.join(FIG_DIR, "compare_CESS_S1_S2.pdf")
    plt.savefig(fig_path, dpi=600)
    plt.close()
    
    print(f"\n[Visual] Comparison Plot saved successfully to {fig_path}")

if __name__ == "__main__":
    main()
