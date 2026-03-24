import os
import sys
import scipy.io
import torch
import numpy as np
import csv

SCRIPT_DIR = '/media/xuda/Data/Rare Event Sampling/Codes/reSGLD - Poisson'
sys.path.insert(0, SCRIPT_DIR)
from utilities import compute_ESS, compute_CESS
from model import Normalizing_Flow
from parameters import para
from Poisson_2D import Poisson_2D

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    para.to(device)
    
    # Initialize Solver for Potentials
    solver = Poisson_2D(para.N_CELLS, para.ALPHA, para.GAMMA, para.C, device=device)
    
    def get_potential_func(beta_val):
        # We ALWAYS use the FULL potential for importance weights, regardless of the training mode
        return lambda x: solver.get_potential_mixed_partial(
            x, para.Y_OBS.to(device), para.OBS_INDICES.to(device), para.SIGMA_NOISE, 
            batch_indices=None, beta=beta_val
        )
    
    if para.BETA_LIST is None:
        print("BETA_LIST not found!")
        return
        
    BETA_LIST = para.BETA_LIST
    M = len(BETA_LIST) - 1
    print(f"Number of steps (M): {M}")
    
    MODES = ['e', 'f', 's']
    
    all_ess_cess = {k: {} for k in range(0, M+1)}
    
    for mode in MODES:
        print(f"\n{'='*70}")
        print(f"Processing Mode: '{mode}'")
        print(f"{'='*70}")
        print(f"{'Step':<6} | {'ESS':<15} | {'CESS':<15}")
        print("-" * 65)
        
        for k in range(0, M+1):
            if k == 0:
                w_0_path = os.path.join(DATA_DIR, f'weights_{mode}_NU_0.mat')
                if not os.path.exists(w_0_path):
                    continue
                w_0_data = scipy.io.loadmat(w_0_path)
                log_w_0 = torch.tensor(w_0_data['weights'], dtype=torch.float64, device=device).flatten()
                N = log_w_0.shape[0]
                max_log_w = torch.max(log_w_0)
                W_0 = torch.exp(log_w_0 - max_log_w)
                ess_0 = (torch.sum(W_0)**2) / (torch.sum(W_0**2)) / N
                print(f"{0:<6} | {ess_0.item():<15.2%} | {'N/A':<15}")
                all_ess_cess[0][mode] = (ess_0.item(), "N/A")
                continue

            # 1. Compute ESS_k directly from weights_{mode}_NU_k
            w_curr_path = os.path.join(DATA_DIR, f'weights_{mode}_NU_{k}.mat')
            if not os.path.exists(w_curr_path):
                break
            w_curr_data = scipy.io.loadmat(w_curr_path)
            log_w_curr = torch.tensor(w_curr_data['weights'], dtype=torch.float64, device=device).flatten()
            N = log_w_curr.shape[0]
            max_log_w_curr = torch.max(log_w_curr)
            W_curr = torch.exp(log_w_curr - max_log_w_curr)
            ess_k = (torch.sum(W_curr)**2) / (torch.sum(W_curr**2)) / N
            
            nu_prev_path = os.path.join(DATA_DIR, f'samples_{mode}_NU_{k-1}.mat')
            w_prev_path = os.path.join(DATA_DIR, f'weights_{mode}_NU_{k-1}.mat')
            model_path = os.path.join(DATA_DIR, f'flow_{mode}_{k}.pth')
            
            nu_prev_data = scipy.io.loadmat(nu_prev_path)
            x_prev = torch.tensor(nu_prev_data['samples'], dtype=torch.float32, device=device)
            
            w_prev_data = scipy.io.loadmat(w_prev_path)
            log_w_prev = torch.tensor(w_prev_data['weights'], dtype=torch.float64, device=device).flatten()
            
            beta_prev = BETA_LIST[k-1]
            beta_curr = BETA_LIST[k]
            u_prev_func = get_potential_func(beta_prev.item())
            u_curr_func = get_potential_func(beta_curr.item())
            
            has_flow = os.path.exists(model_path)
            if has_flow:
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                model = Normalizing_Flow().to(device)
                
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['flow_state_dict'])
                model.eval()
                
                if mode != 'e':
                    from model import WeightNet
                    phi = WeightNet(para.DIM).to(device)
                    psi = WeightNet(para.DIM).to(device)
                    phi.load_state_dict(checkpoint['phi_state_dict'])
                    psi.load_state_dict(checkpoint['psi_state_dict'])
                    phi.eval()
                    psi.eval()
                else:
                    phi, psi = None, None
                    
                sys.stdout = old_stdout
            else:
                model = None
                phi, psi = None, None

            # Process in batches
            incremental_log_ws = []
            batch_size = 5000
            with torch.no_grad():
                for i in range(int(np.ceil(N / batch_size))):
                    bx = x_prev[i*batch_size : (i+1)*batch_size]
                    
                    if has_flow:
                        fx, log_det = model(bx, inverse=False)
                    else:
                        fx = bx
                        log_det = 0.0
                        
                    if mode == 'e':
                        u_prev = u_prev_func(bx)
                        u_curr = u_curr_func(fx)
                        inc_log_w = -u_curr + u_prev + log_det
                    else:
                        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                            phi_val = phi(bx).squeeze()
                            psi_val = psi(fx).squeeze()
                        inc_log_w = phi_val.float() - psi_val.float() + log_det.float()

                    incremental_log_ws.append(inc_log_w)
                    
            inc_log_w = torch.cat(incremental_log_ws, dim=0).to(torch.float64)

            # CESS Calculation (Log-Stable)
            log_alpha = log_w_prev
            log_w = inc_log_w
            
            term1 = torch.logsumexp(log_alpha + log_w, dim=0)
            term2 = torch.logsumexp(log_alpha + 2 * log_w, dim=0)
            term3 = torch.logsumexp(log_alpha, dim=0)
            
            log_cess = 2 * term1 - term2 - term3
            cess_k = torch.exp(log_cess).item()

            scipy.io.savemat(
                w_curr_path,
                {
                    'weights': log_w_curr.cpu().numpy(),
                    'cess': cess_k,
                    'ess': ess_k.item()
                }
            )
            
            print(f"{k:<6} | {ess_k.item():<15.2%} | {cess_k:<15.2%}")
            all_ess_cess[k][mode] = (ess_k.item(), cess_k)
            
    # Write aggregated CSV securely
    csv_path = os.path.join(DATA_DIR, 'P_CESS.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'ESS_e', 'CESS_e', 'ESS_f', 'CESS_f', 'ESS_s', 'CESS_s'])
        for k in range(0, M+1):
            row = [k]
            for mode in ['e', 'f', 's']:
                if mode in all_ess_cess[k]:
                    ess, cess = all_ess_cess[k][mode]
                    ess_str = f"{ess:.2%}"
                    cess_str = f"{cess:.2%}" if isinstance(cess, float) else "N/A"
                    row.extend([ess_str, cess_str])
                else:
                    row.extend(["N/A", "N/A"])
            writer.writerow(row)

if __name__ == '__main__':
    main()
