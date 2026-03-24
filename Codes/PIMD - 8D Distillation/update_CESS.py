import os
import sys
import torch
import numpy as np
import csv

SCRIPT_DIR = '/media/xuda/Data/Rare Event Sampling/Codes/PIMD - 8D Distillation'
sys.path.insert(0, SCRIPT_DIR)
from utilities import compute_ESS, compute_CESS
from model import Normalizing_Flow
from parameters import pt, M
import parameters

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    pt_global = pt.to(device)
    
    BETA_LIST = torch.linspace(0, 1, M+1, device=device)
    print(f"Number of steps (M): {M}")
    
    def get_potential_func(beta_val):
        return lambda x: (1.0 - beta_val) * pt_global.base_Uf(x) + beta_val * pt_global.target_Uf(x)
    
    MODES = ['S1', 'S2']
    
    all_ess_cess = {k: {} for k in range(0, M+1)}
    
    for mode in MODES:
        print(f"\n{'='*70}")
        print(f"Processing Mode: '{mode}'")
        print(f"{'='*70}")
        print(f"{'Step':<6} | {'ESS':<15} | {'CESS':<15}")
        print("-" * 65)
        
        samples_path = os.path.join(DATA_DIR, f'samples_NU_{mode}.pth')
        weights_path = os.path.join(DATA_DIR, f'weights_NU_{mode}.pth')
        flows_path = os.path.join(DATA_DIR, f'flows_{mode}.pth')
        
        if not (os.path.exists(samples_path) and os.path.exists(weights_path) and os.path.exists(flows_path)):
            print(f"Data for mode {mode} not fully found. Skipping.")
            continue
            
        samples_all = torch.load(samples_path, map_location=device)
        weights_dict = torch.load(weights_path, map_location=device)
        global_flows = torch.load(flows_path, map_location=device)
        
        weights_all = weights_dict['weights']
        ess_all = weights_dict['ess']
        cess_all = weights_dict['cess']
        
        for k in range(0, M+1):
            if k == 0:
                log_w_0 = weights_all[0].to(device)
                N = log_w_0.shape[0]
                max_log_w = torch.max(log_w_0)
                W_0 = torch.exp(log_w_0 - max_log_w)
                ess_0 = (torch.sum(W_0)**2) / (torch.sum(W_0**2)) / N
                
                print(f"{0:<6} | {ess_0.item():<15.2%} | {'N/A':<15}")
                all_ess_cess[0][mode] = (ess_0.item(), "N/A")
                
                ess_all[0] = ess_0.item()
                cess_all[0] = 0.0
                continue

            log_w_curr = weights_all[k].to(device)
            N = log_w_curr.shape[0]
            max_log_w_curr = torch.max(log_w_curr)
            W_curr = torch.exp(log_w_curr - max_log_w_curr)
            ess_k = (torch.sum(W_curr)**2) / (torch.sum(W_curr**2)) / N
            
            x_prev = samples_all[k-1].to(device)
            log_w_prev = weights_all[k-1].to(device)
            
            beta_prev = BETA_LIST[k-1].item()
            beta_curr = BETA_LIST[k].item()
            
            u_prev_func = get_potential_func(beta_prev)
            u_curr_func = get_potential_func(beta_curr)
            
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            model = Normalizing_Flow(pt=pt_global).to(device)
            model.load_state_dict(global_flows[f'flow_{k}'])
            model.eval()
            sys.stdout = old_stdout

            # Process in batches
            incremental_log_ws = []
            batch_size = 5000
            with torch.no_grad():
                for i in range(int(np.ceil(N / batch_size))):
                    bx = x_prev[i*batch_size : (i+1)*batch_size]
                    
                    fx, log_det = model(bx, inverse=False)
                        
                    u_prev = u_prev_func(bx)
                    u_curr = u_curr_func(fx)
                    inc_log_w = -u_curr + u_prev + log_det
                    
                    incremental_log_ws.append(inc_log_w)
                    
            inc_log_w = torch.cat(incremental_log_ws, dim=0).to(torch.float64)

            # CESS Calculation (Log-Stable)
            log_alpha = log_w_prev.to(torch.float64)
            log_w = inc_log_w
            
            term1 = torch.logsumexp(log_alpha + log_w, dim=0)
            term2 = torch.logsumexp(log_alpha + 2 * log_w, dim=0)
            term3 = torch.logsumexp(log_alpha, dim=0)
            
            log_cess = 2 * term1 - term2 - term3
            cess_k = torch.exp(log_cess).item()

            print(f"{k:<6} | {ess_k.item():<15.2%} | {cess_k:<15.2%}")
            
            all_ess_cess[k][mode] = (ess_k.item(), cess_k)
            ess_all[k] = ess_k.item()
            cess_all[k] = cess_k
            
        # Overwrite the weights MAT file with corrected ESS and CESS
        weights_dict['ess'] = ess_all
        weights_dict['cess'] = cess_all
        torch.save(weights_dict, weights_path)
            
    # Write aggregated CSV securely
    csv_path = os.path.join(DATA_DIR, 'PIMD_CESS.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'ESS_S1', 'CESS_S1', 'ESS_S2', 'CESS_S2'])
        for k in range(0, M+1):
            row = [k]
            for mode in ['S1', 'S2']:
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
