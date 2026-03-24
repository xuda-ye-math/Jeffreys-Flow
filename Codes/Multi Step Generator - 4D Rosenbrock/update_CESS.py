import os
import sys
import scipy.io
import torch
import numpy as np

SCRIPT_DIR = '/media/xuda/Data/Rare Event Sampling/Codes/Multi Step Generator - 4D Rosenbrock'
sys.path.insert(0, SCRIPT_DIR)
from utilities import compute_ESS, compute_CESS
from model import Normalizing_Flow
from parameters import para

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
PREFIX = 'RB'

def get_potential_func(beta_val):
    return lambda x: para.compute_potential_mixed(x, lam=beta_val)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    para.to(device)
    
    beta_path = os.path.join(DATA_DIR, f'{PREFIX}_BETA_LIST.mat')
    if not os.path.exists(beta_path):
        print(f"BETA_LIST not found at {beta_path}")
        return
        
    beta_data = scipy.io.loadmat(beta_path)
    BETA_LIST = beta_data['BETA_LIST'].flatten()
    M = len(BETA_LIST) - 1
    
    print(f"Number of steps (M): {M}")
    print("-" * 65)
    print(f"{'Step':<6} | {'ESS':<15} | {'CESS':<15}")
    print("-" * 65)
    
    for k in range(0, M+1):
        if k == 0:
            w_0_path = os.path.join(DATA_DIR, f'{PREFIX}_weights_NU_0.mat')
            if not os.path.exists(w_0_path):
                continue
            w_0_data = scipy.io.loadmat(w_0_path)
            log_w_0 = torch.tensor(w_0_data['weights'], dtype=torch.float64, device=device).flatten()
            N = log_w_0.shape[0]
            max_log_w = torch.max(log_w_0)
            W_0 = torch.exp(log_w_0 - max_log_w)
            ess_0 = (torch.sum(W_0)**2) / (torch.sum(W_0**2)) / N
            print(f"{0:<6} | {ess_0.item():<15.2%} | {'N/A':<15}")
            continue

        # 1. Compute ESS_k directly from weights_NU_k
        w_curr_path = os.path.join(DATA_DIR, f'{PREFIX}_weights_NU_{k}.mat')
        if not os.path.exists(w_curr_path):
            break
        w_curr_data = scipy.io.loadmat(w_curr_path)
        log_w_curr = torch.tensor(w_curr_data['weights'], dtype=torch.float64, device=device).flatten()
        N = log_w_curr.shape[0]
        max_log_w_curr = torch.max(log_w_curr)
        W_curr = torch.exp(log_w_curr - max_log_w_curr)
        ess_k = (torch.sum(W_curr)**2) / (torch.sum(W_curr**2)) / N
        
        # 2. Compute CESS_k by applying F_k to nu_{k-1}
        nu_prev_path = os.path.join(DATA_DIR, f'{PREFIX}_samples_NU_{k-1}.mat')
        w_prev_path = os.path.join(DATA_DIR, f'{PREFIX}_weights_NU_{k-1}.mat')
        model_path = os.path.join(DATA_DIR, f'{PREFIX}_flow_{k}.pth')
        
        nu_prev_data = scipy.io.loadmat(nu_prev_path)
        x_prev = torch.tensor(nu_prev_data['samples'], dtype=torch.float32, device=device)
        
        w_prev_data = scipy.io.loadmat(w_prev_path)
        log_w_prev = torch.tensor(w_prev_data['weights'], dtype=torch.float64, device=device).flatten()
        
        # Load model with suppressed output
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        model = Normalizing_Flow().to(device)
        sys.stdout = old_stdout
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        beta_prev = BETA_LIST[k-1]
        beta_curr = BETA_LIST[k]
        u_prev_func = get_potential_func(beta_prev)
        u_curr_func = get_potential_func(beta_curr)
        
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

        # Compute CESS directly using utility function
        # We must normalize the previous log weights so they sum to 1 in probability space
        # otherwise E_w[f(w)] will not be properly evaluated in utilities
        stable_w_prev = log_w_prev - torch.max(log_w_prev)
        norm_log_w_prev = stable_w_prev - torch.logsumexp(stable_w_prev, dim=0)

        cess_k = compute_CESS(norm_log_w_prev, inc_log_w)

        # Overwrite the weights MAT file with corrected ESS and CESS
        scipy.io.savemat(
            w_curr_path,
            {
                'weights': log_w_curr.cpu().numpy(),
                'cess': cess_k,
                'ess': ess_k.item()
            }
        )
        
        print(f"{k:<6} | {ess_k.item():<15.2%} | {cess_k:<15.2%}")
        
if __name__ == '__main__':
    main()
