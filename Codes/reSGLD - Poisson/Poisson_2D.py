import torch

class Poisson_2D:
    """
    Batched 2D Screened Poisson Solver in PyTorch.
    Designed to exactly replicate the MATLAB Poisson_2D.m behavior while 
    supporting highly parallel batched inputs [Batch, DIM].
    """
    def __init__(self, N_cells, alpha, gamma, c, device='cuda'):
        self.N_cells = int(N_cells)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.c = float(c)
        self.device = device
        
        self.h = 1.0 / self.N_cells
        self.n_pts = self.N_cells + 1
        self.N = self.n_pts ** 2
        
        # Grid using MATLAB-like meshgrid (indexing='xy' ensures X differs by column)
        grid_vec = torch.linspace(0, 1, self.n_pts, device=device)
        self.X, self.Y = torch.meshgrid(grid_vec, grid_vec, indexing='xy')
        
        # In MATLAB, 'f(:)' flattens column-wise. We achieve this matching order:
        self.X_flat = self.X.T.flatten()
        self.Y_flat = self.Y.T.flatten()
        
        # Build the system matrix A and invert it for fast batched forward solves
        self.build_matrix()
        
    def build_matrix(self):
        """
        Builds the 5-point finite difference matrix A.
        Since N=1681 typically, we use a dense representation and pre-invert it ONCE.
        This changes an iterative/sparse solve to a single extreme-fast batched MatMul.
        """
        # Build A in float64 for highly exact inverse condition handling
        A = torch.zeros((self.N, self.N), dtype=torch.float64, device=self.device)
        h2 = self.h ** 2
        
        for j in range(1, self.n_pts + 1):
            for i in range(1, self.n_pts + 1):
                # Flattened index matching MATLAB's column-major `k = i + (j - 1) * n_pts`
                k = (i - 1) + (j - 1) * self.n_pts
                
                coeff_center = 4.0 / h2 + self.alpha
                c_west  = -1.0 / h2
                c_east  = -1.0 / h2
                c_south = -1.0 / h2
                c_north = -1.0 / h2
                
                # Apply pure Neumann Boundary Conditions
                if i == 1:
                    coeff_center += c_west
                    c_west = 0.0
                if i == self.n_pts:
                    coeff_center += c_east
                    c_east = 0.0
                if j == 1:
                    coeff_center += c_south
                    c_south = 0.0
                if j == self.n_pts:
                    coeff_center += c_north
                    c_north = 0.0
                    
                A[k, k] = coeff_center
                if c_west != 0: A[k, k - 1] = c_west
                if c_east != 0: A[k, k + 1] = c_east
                if c_south != 0: A[k, k - self.n_pts] = c_south
                if c_north != 0: A[k, k + self.n_pts] = c_north
                
        print(f"[Poisson_2D_GPU] Inverting {self.N}x{self.N} system matrix on {self.device}...")
        # Then cast it down to Float32 to optimize VRAM space and matmul speed
        self.A_inv = torch.linalg.inv(A).float()
        print("[Poisson_2D_GPU] Pre-computation completed. Ready for batch-inference.")

    def build_source(self, theta):
        """
        Constructs the batched source term f(x; theta).
        
        Args:
            theta: Tensor of shape [Batch, 8]
        Returns:
            f_mat: Tensor shape [Batch, N] representing the flattened RHS.
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
            
        Batch = theta.shape[0]
        num_sources = theta.shape[1] // 2
        
        X_b = self.X_flat.unsqueeze(0) # [1, N]
        Y_b = self.Y_flat.unsqueeze(0) # [1, N]
        
        f_vec = torch.zeros((Batch, self.N), dtype=torch.float32, device=self.device)
        
        for k in range(num_sources):
            mu_x = theta[:, 2*k].unsqueeze(1)     # [Batch, 1]
            mu_y = theta[:, 2*k + 1].unsqueeze(1) # [Batch, 1]
            
            # Using broadcasting to construct the sources
            gaussian = self.c * torch.exp(-((X_b - mu_x)**2 + (Y_b - mu_y)**2) / (2 * self.gamma**2))
            f_vec += gaussian
            
        return f_vec

    def solve_forward(self, theta):
        """
        Solves the forward Poisson problem for a batch of thetas.
        A * u = f  ==>  u = f * A_inv^T (batched)
        
        Args:
            theta: [Batch, 8] Tensor
        Returns:
            u: [Batch, N] State field vectors
        """
        f = self.build_source(theta)
        # Highly efficient Batched Matrix-Vector Multiplication mapping
        u = torch.matmul(f, self.A_inv.T)
        return u
        
    def solve_forward_mat(self, theta):
        """
        Solves the forward problem and returns structured 2D Grids.
        
        Args:
            theta: [Batch, 8] Tensor
        Returns:
            u_mat: [Batch, n_pts, n_pts] 
        """
        u = self.solve_forward(theta)
        
        # Reshape to match MATLAB's column-major matrices (transpose spatial dims)
        # PyTorch natively shapes as [Batch, Row, Col]. We transpose to emulate MATLAB.
        u_mat = u.reshape(-1, self.n_pts, self.n_pts).transpose(1, 2)
        return u_mat

    def get_potential_mixed_partial(self, theta, y_obs, obs_indices, sigma_noise, batch_indices=None, beta=1.0):
        """
        Computes the mixed potential estimator U_mix for reSGLD across a specific batch.
        Supports Batched theta evaluation.
        
        Args:
            theta: [Batch, 8] Tensor 
            y_obs: [81] Tensor of empirical observations
            obs_indices: [81] Tensor of 1-indexed MATLAB coordinates mapped to self.N
            sigma_noise: float, std of noise
            batch_indices: [SubBatch] Tensor of indices (0-indexed) specifying which sensors to use.
                           If None, uses all observations.
            beta: float or [Batch], inverse temperature (target concentration factor).
            
        Returns:
            U_mix: [Batch] Tensor containing the weighted potential loss per item.
        """
        total_obs = y_obs.shape[0]
        
        if batch_indices is None:
            batch_indices = torch.arange(total_obs, device=self.device)
            
        num_batch_obs = batch_indices.shape[0]
        
        # 1. Forward solve: [Batch, N]
        u_full = self.solve_forward(theta) 
        
        # 2. Extract specific observations (MATLAB uses 1-based indexing, map to 0-based)
        grid_indices = obs_indices[batch_indices] - 1
        
        # u_batch: [Batch, num_batch_obs]
        G_theta_batch = u_full[:, grid_indices]
        
        # y_obs_batch: [num_batch_obs]
        y_obs_batch = y_obs[batch_indices]
        
        # 3. Compute Residual
        residuals = G_theta_batch - y_obs_batch.unsqueeze(0) # [Batch, num_batch_obs]
        ssr_batch = torch.sum(residuals**2, dim=1) # [Batch]
        
        # 4. Compute Loss
        U_batch = ssr_batch / (2 * sigma_noise**2)
        
        # 5. Scale to unbiased total estimator
        scale_factor = total_obs / num_batch_obs
        U_target_partial = U_batch * scale_factor
        
        # 6. Temperature mixing
        U_mix = beta * U_target_partial
        
        return U_mix


if __name__ == "__main__":
    # Small automated unit test comparing shape compatibility with Parameters
    import time
    from parameters import para
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Poisson_2D on {device}...")
    solver = Poisson_2D(para.N_CELLS, para.ALPHA, para.GAMMA, para.C, device=device)
    
    batch_size = 1000
    # Add requires_grad=True to test PyTorch Autograd backwards pass
    theta_dummy = torch.rand((batch_size, 8), device=device, requires_grad=True)
    
    t0 = time.time()
    u_solution = solver.solve_forward(theta_dummy)
    t1 = time.time()
    
    print(f"Output shape of u: {u_solution.shape}")
    print(f"Computed {batch_size} forward passes in {t1-t0:.4f} seconds!")
    
    print("\nTesting Potential Mixed Computation...")
    # Map parameters onto same device
    para.to(device)
    
    # Test Full Batch (No Subsampling)
    U_mix_full = solver.get_potential_mixed_partial(
        theta_dummy, para.Y_OBS, para.OBS_INDICES, para.SIGMA_NOISE, 
        batch_indices=None, beta=0.5
    )
    
    # Test Sub-Batch (e.g. SGLD logic)
    random_sensors = torch.randperm(81, device=device)[:9]
    U_mix_sub = solver.get_potential_mixed_partial(
        theta_dummy, para.Y_OBS, para.OBS_INDICES, para.SIGMA_NOISE, 
        batch_indices=random_sensors, beta=0.5
    )
    
    print(f"U_mix Full shape: {U_mix_full.shape} | Sample Value: {U_mix_full[0].item():.4f}")
    print(f"U_mix Sub  shape: {U_mix_sub.shape} | Sample Value: {U_mix_sub[0].item():.4f}")
    
    print("\nTesting Autograd Backpropagation...")
    t2 = time.time()
    # Compute sum over batch (each item in batch has independent graph)
    loss = torch.sum(U_mix_sub) 
    loss.backward()
    t3 = time.time()
    
    print(f"Computed Backpropagation for {batch_size} items in {t3-t2:.4f} seconds!")
    print(f"Gradient Tensor Shape: {theta_dummy.grad.shape}")
    print(f"First Sample's Gradients:\n{theta_dummy.grad[0].cpu().numpy()}")
