import torch
import numpy as np

class Solvated_16D:
    """
    Solvated_16D Potential Energy System Class
    
    Target Distribution: Solvated Particle in a Periodic Potential
    Manifold: T^2 x R^14  (First 2 dims periodic, 14 dims Euclidean)
    
    U(x) = U_grid(x1, x2) + U_bath(x1, x2, x3:16)
    
    1. Periodic Grid (4x4 Wells):
      U_grid = A * [ cos(4*x1) + cos(4*x2) ]
      (x1, x2 in [-pi, pi])
    
    2. Harmonic Solvent Coupling:
      U_bath = sum_{k=3}^{16} 0.5 * k_s * (x_k - alpha * sin(w*x1)*sin(w*x2))^2
    """

    NAME = 'Solvated Periodic Grid'
    ABBR = 'SL'
    DIM = 16

    # ---- 1. Configurable Parameters ----
    # Base Distribution (Gaussian)
    MEAN        = 0.0
    SIGMA       = 1.0

    # MALA Dynamics
    DT = 1e-3
    N_ITER = 1000
    
    # Potential Specifics
    # Grid Parameters (4x4 in [-pi, pi])
    A_GRID      = 2.0   # Barrier height for grid
    FREQ_GRID   = 4.0   # Frequency 4 implies 4 wells in 2pi (if cos(4x))
    
    # Coupling Parameters
    K_SPRING    = 30.0  # Stiff solvent (Constant)
    
    ALPHA_MIN   = 2.0
    ALPHA_MAX   = 4.0
    
    FREQ_C      = 1.0   # Coupling frequency

    # Visualization Limits
    X_PERI_LIM = [-np.pi, np.pi]
    X_FREE_LIM_COMP  = [-6.0, 6.0]
    X_FREE_LIM_PLOT  = [-4.0, 4.0]

    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Initialize Variable ALPHA_C
        # Linearly spaced from ALPHA_MIN to ALPHA_MAX across the 14 bath dimensions
        # Corresponds to MATLAB: linspace(obj.ALPHA_MIN, obj.ALPHA_MAX, obj.DIM - 2)
        alpha_c_np = np.linspace(self.ALPHA_MIN, self.ALPHA_MAX, self.DIM - 2)
        self.ALPHA_C = torch.tensor(alpha_c_np, dtype=torch.float32).to(self.device)

    def to(self, device):
        self.device = device
        if hasattr(self, 'ALPHA_C') and isinstance(self.ALPHA_C, torch.Tensor):
             self.ALPHA_C = self.ALPHA_C.to(device)
        return self

    # ---- Potential Functions ----

    def compute_potential_base(self, x: torch.Tensor) -> torch.Tensor:
        """
        U_base: Isotropic Gaussian on bath dims, Flat on Torus dims (1-2)
        
        MATLAB:
        Dims 1-2: Uniform on Torus => Potential = 0 (Flat)
        Dims 3-16: Gaussian Bath => Potential = 0.5 * z^2
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        
        # Gaussian contribution from Bath Dims only (indices 2 to end)
        x_bath = x[:, 2:]
        z_bath = (x_bath - self.MEAN) / self.SIGMA
        u_base = 0.5 * torch.sum(z_bath**2, dim=1)
        
        return u_base
    
    def compute_potential_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        U_target: Solvated System
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        
        x1 = x[:, 0:1] # Keep dim for broadcasting
        x2 = x[:, 1:2]
        x_bath = x[:, 2:]
        
        # Part A: Periodic Grid
        # U_grid = A * ( cos(4*x1) + cos(4*x2) )
        u_grid = self.A_GRID * (torch.cos(self.FREQ_GRID * x1) + torch.cos(self.FREQ_GRID * x2))
        u_grid = u_grid.squeeze(1) # [N]
        
        # Part B: Bath Coupling
        # shift_k = alpha_k * sin(w*x1) * sin(w*x2)
        # ALPHA_C is [14], sin is [N, 1]
        
        term_coupling = torch.sin(self.FREQ_C * x1) * torch.sin(self.FREQ_C * x2) # [N, 1]
        shift_val = self.ALPHA_C * term_coupling # [N, 14] via broadcasting
        
        diff = x_bath - shift_val
        
        # Constant Stiffness: 0.5 * K * sum(diff^2)
        temp = torch.sum(diff**2, dim=1)
        u_bath = 0.5 * self.K_SPRING * temp
        
        u_target = u_grid + u_bath
        
        return u_target

    def compute_potential_mixed(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        U_mix = (1-lambda)U_base + lambda U_target
        """
        u0 = self.compute_potential_base(x)
        u1 = self.compute_potential_target(x)
        return (1.0 - lam) * u0 + lam * u1

    # ---- Gradient Functions ----

    def compute_gradient_base(self, x: torch.Tensor) -> torch.Tensor:
        """
        grad U_base(x)
        Dims 1-2: 0
        Dims 3-16: z / sigma
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        
        grad = torch.zeros_like(x)
        
        # Bath gradients
        grad[:, 2:] = (x[:, 2:] - self.MEAN) / (self.SIGMA ** 2)
        
        return grad

    def compute_gradient_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        grad U_target(x) w.r.t x
        Analytical gradient matching MATLAB implementation
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        
        grad_target = torch.zeros_like(x)
        
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x_bath = x[:, 2:]
        
        # Precompute common terms
        S1 = torch.sin(self.FREQ_C * x1)
        C1 = torch.cos(self.FREQ_C * x1)
        S2 = torch.sin(self.FREQ_C * x2)
        C2 = torch.cos(self.FREQ_C * x2)
        
        # shift = alpha * S1 * S2
        shift = self.ALPHA_C * (S1 * S2) # [N, 14]
        
        # Diff [N, 14]
        diff = x_bath - shift
        
        # A. Bath Gradients (x3...x16)
        # dU/dx_k = K * (x_k - shift_k)
        grad_target[:, 2:] = self.K_SPRING * diff
        
        # B. Coupling Gradient on x1, x2 (Chain rule)
        # common_grad_bath corresponds to dU/ds_k in MATLAB (-K * diff)
        common_grad_bath = -self.K_SPRING * diff # [N, 14]
        
        # Derivatives of shift w.r.t x1 and x2
        d_shift_d_x1 = self.ALPHA_C * (self.FREQ_C * C1 * S2) # [N, 14]
        d_shift_d_x2 = self.ALPHA_C * (self.FREQ_C * S1 * C2) # [N, 14]
        
        # Sum over k (bath dims)
        grad_c_x1 = torch.sum(common_grad_bath * d_shift_d_x1, dim=1, keepdim=True) # [N, 1]
        grad_c_x2 = torch.sum(common_grad_bath * d_shift_d_x2, dim=1, keepdim=True) # [N, 1]
        
        # C. Grid Gradients
        # U_grid = A(cos4x1 + cos4x2)
        # dU/dx1 = -4A sin(4x1)
        grad_g_x1 = -self.FREQ_GRID * self.A_GRID * torch.sin(self.FREQ_GRID * x1)
        grad_g_x2 = -self.FREQ_GRID * self.A_GRID * torch.sin(self.FREQ_GRID * x2)
        
        grad_target[:, 0:1] = grad_g_x1 + grad_c_x1
        grad_target[:, 1:2] = grad_g_x2 + grad_c_x2
        
        return grad_target

    def compute_gradient_mixed(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        grad U_mix = (1-lambda)grad U_base + lambda grad U_target
        """
        g0 = self.compute_gradient_base(x)
        g1 = self.compute_gradient_target(x)
        return (1.0 - lam) * g0 + lam * g1
