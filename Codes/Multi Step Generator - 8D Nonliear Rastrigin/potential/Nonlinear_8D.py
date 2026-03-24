import torch
import numpy as np
import scipy.io
import os


class Nonlinear_8D:
    """
    8D Nonlinear Rastrigin Potential Energy System.

    Target Distribution: Twisted Independent Rastrigin (No Inner Coupling)
    The potential is defined as a composition:
      U_total(x) = U_inner(T(x))

    1. Outer Twist (Diffeomorphism):
      z = T(x) = x + delta * tanh(x Q)
      where Q is a fixed orthogonal matrix loaded from data/DISTORT_Q.mat.

    2. Inner Potential (Decoupled Rastrigin):
      U_inner(z) = sum [ 0.5*z_i^2 + A*cos(w z_i) ]
    """

    NAME = 'Twisted Nonlinear Rastrigin'
    ABBR = 'NR'
    DIM = 8

    # ---- 1. Configurable Parameters ----
    # Base Distribution (Gaussian)
    MEAN = 0.0
    SIGMA = 1.0

    # Dynamics
    DT = 5e-3
    N_ITER = 1000

    # Potential Specifics
    A_VAL = 12.0     # Amplitude A
    W_VAL = 2.0      # Frequency w
    DISTORT_DELTA = 1.0

    # Visualization Limits (Projected)
    X_LIM_COMPUTE = [-6.0, 6.0]
    X_LIM_PLOT = [-4.0, 4.0]

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.load_distortion_matrix()

    def to(self, device):
        self.device = device
        if hasattr(self, 'DISTORT_Q') and isinstance(self.DISTORT_Q, torch.Tensor):
             self.DISTORT_Q = self.DISTORT_Q.to(device)
        return self

    def load_distortion_matrix(self):
        # Load Q from data/DISTORT_Q.mat
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mat_path = os.path.join(base_dir, 'data', 'DISTORT_Q.mat')
        if not os.path.exists(mat_path):
             # Try absolute path based on project structure if relative fails, or just raise
             # Just raise for now as per plan
             raise FileNotFoundError(f"Distortion matrix not found at {mat_path}. Please run Nonlinear_8D.m (disp_info) to generate it.")
        
        mat_data = scipy.io.loadmat(mat_path)
        Q_np = mat_data['Q']
        self.DISTORT_Q = torch.tensor(Q_np, dtype=torch.float32).to(self.device)

    # ---- Potential Functions ----

    def compute_potential_base(self, x: torch.Tensor) -> torch.Tensor:
        """
        U_base: Isotropic Gaussian
        U(x) = 0.5 * sum ((x - mean)/sigma)^2
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        z = (x - self.MEAN) / self.SIGMA
        return 0.5 * torch.sum(z ** 2, dim=1)
    
    def map_forward(self, x: torch.Tensor):
        """
        Applies the diffeomorphism T(x)
        z = x + delta * tanh(x Q)
        """
        pre_act = x @ self.DISTORT_Q
        act = torch.tanh(pre_act)
        z = x + self.DISTORT_DELTA * act
        
        # J_sech2 = 1 - tanh^2
        J_sech2 = 1.0 - act ** 2
        return z, J_sech2

    def map_gradient_backward(self, grad_z: torch.Tensor, J_sech2: torch.Tensor):
        """
        Applies chain rule:
        grad_x = grad_z + delta * (grad_z * J_sech2) @ Q.T
        """
        term = grad_z * J_sech2
        return grad_z + self.DISTORT_DELTA * (term @ self.DISTORT_Q.T)

    def compute_potential_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        U_target: Twisted Decoupled Rastrigin
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        
        # 1. Apply Twist
        z, _ = self.map_forward(x)
        
        # 2. Rastrigin on z
        term_rastrigin = torch.sum(0.5 * z**2 + self.A_VAL * torch.cos(self.W_VAL * z), dim=1)
        
        return term_rastrigin

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
        grad U_base(x) = (x - mean) / sigma^2
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        return (x - self.MEAN) / (self.SIGMA ** 2)

    def compute_gradient_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        grad U_target(x) w.r.t x
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        
        # A. Forward Map
        z, J_sech2 = self.map_forward(x)
        
        # B. Gradient w.r.t z
        grad_inner = z - self.A_VAL * self.W_VAL * torch.sin(self.W_VAL * z)
        
        # C. Backward Map
        grad = self.map_gradient_backward(grad_inner, J_sech2)
        
        return grad

    def compute_gradient_mixed(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        grad U_mix = (1-lambda)grad U_base + lambda grad U_target
        """
        g0 = self.compute_gradient_base(x)
        g1 = self.compute_gradient_target(x)
        return (1.0 - lam) * g0 + lam * g1
