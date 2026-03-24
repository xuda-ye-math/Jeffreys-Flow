import torch
import math


# =========================================================================
# 1. JIT Compiled Calculation Kernel (Potential)
# =========================================================================

@torch.jit.script
def _rosenbrock_pot_jit(x: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for 4D Rosenbrock Potential.
    Matches MATLAB logic:
      term = 200 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2
      Summed over i = 0 to D-2
    Input: x [N, 4]
    Output: V [N] (Potential energy)
    """
    # D = 4, so loop i from 0 to 2
    # Indices: 0->1, 1->2, 2->3

    val = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    for i in range(3):
        x_i = x[:, i]
        x_next = x[:, i + 1]

        diff = x_next - x_i ** 2
        term = 200.0 * diff ** 2 + (1.0 - x_i) ** 2
        val += term

    return val  # Scale factor is 1.0 in MATLAB code


@torch.jit.script
def _rosenbrock_grad_jit(x: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for 4D Rosenbrock Gradient.
    Strictly follows MATLAB 'compute_gradient_mixed' arithmetic:
      d_xi    = -400 * x_i * diff - 2 * (1 - x_i)
      d_xnext = 200 * diff
      grad    = accumulate * 4.0
    """
    grad = torch.zeros_like(x)
    scale_factor = 4.0

    for i in range(3):
        x_i = x[:, i]
        x_next = x[:, i + 1]

        diff = x_next - x_i ** 2

        # Derivatives as defined in MATLAB source
        d_xi = -400.0 * x_i * diff - 2.0 * (1.0 - x_i)
        d_xnext = 200.0 * diff

        # Accumulate gradients (Note: PyTorch in-place add)
        grad[:, i] += d_xi
        grad[:, i + 1] += d_xnext

    return grad * scale_factor


# =========================================================================
# 2. Potential Energy Class
# =========================================================================

class Rosenbrock_4D:
    """
    4D ROSENBROCK Potential Energy System.
    Target Distribution:
      U(x) = sum_{i=1}^{3} [ 200(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]

    Note: Base Sigma and Mean are vectorized [4].
    """

    NAME = '4D Rosenbrock (Scaled)'
    ABBR = 'RB'
    DIM = 4

    # ---- 1. Configurable Parameters ----
    # Base Distribution (Gaussian) Parameters
    # Defined as lists here, converted to Tensor in __init__
    _MEAN_LIST = [0.0, 0.6, 1.0, 2.0]
    _SIGMA_LIST = [1.0, 1.0, 1.5, 2.0]

    # Rejuvenation Parameters
    DT = 5e-4
    N_ITER = 1000  # Corresponds to WARM_TIME in MATLAB (approx)

    # Visualization Limits (Projected)
    X1_LIM_COMPUTE = [-1.6, 2]
    X2_LIM_COMPUTE = [-0.5, 2]
    X3_LIM_COMPUTE = [-0.5, 3]
    X4_LIM_COMPUTE = [-0.5, 6.5]

    # Specific Plot Limits
    X1_LIM_PLOT = [-1.2, 1.3]
    X2_LIM_PLOT = [-0.1, 1.6]
    X3_LIM_PLOT = [-0.1, 2.5]
    X4_LIM_PLOT = [-0.1, 6.0]

    def __init__(self, device: str = 'cpu'):
        self.device = device

        # Initialize Base Parameters
        # Naming matches GMM_3D style (BASE_MEAN, BASE_SIGMA)
        # but uses Tensors for anisotropy.
        self.BASE_MEAN = torch.tensor(self._MEAN_LIST, dtype=torch.float32, device=device)
        self.BASE_SIGMA = torch.tensor(self._SIGMA_LIST, dtype=torch.float32, device=device)

        # Initialize target (empty for Rosenbrock)
        self._init_target_parameters()

    def to(self, device):
        """
        Moves all internal tensors to the specified device.
        """
        self.device = device
        self.BASE_MEAN = self.BASE_MEAN.to(device)
        self.BASE_SIGMA = self.BASE_SIGMA.to(device)
        return self

    def _init_target_parameters(self):
        # No GMM parameters to initialize
        pass

    # ---- Potential Functions ----

    def compute_potential_base(self, x: torch.Tensor) -> torch.Tensor:
        """
        U_base: Diagonal Gaussian Potential
        U(x) = 0.5 * sum ((x - mu)/sigma)^2
        """
        if x.device != torch.device(self.device): x = x.to(self.device)

        # Broadcasting: x [N,4] - MEAN [4] -> [N,4]
        z = (x - self.BASE_MEAN) / self.BASE_SIGMA
        return 0.5 * torch.sum(z ** 2, dim=1)

    def compute_potential_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        U_target: Rosenbrock Potential (JIT)
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        return _rosenbrock_pot_jit(x)

    def compute_potential_mixed(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        U_mix = (1-lambda)U_base + lambda U_target
        """
        u0 = self.compute_potential_base(x)
        u1 = self.compute_potential_target(x)
        return (1.0 - lam) * u0 + lam * u1

    # ---- Gradient Functions (for MALA) ----

    def compute_gradient_base(self, x: torch.Tensor) -> torch.Tensor:
        """
        grad U_base(x) = (x - mu) / sigma^2
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        return (x - self.BASE_MEAN) / (self.BASE_SIGMA ** 2)

    def compute_gradient_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        grad U_target(x) using JIT kernel
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        return _rosenbrock_grad_jit(x)

    def compute_gradient_mixed(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        grad U_mix = (1-lambda)grad U_base + lambda grad U_target
        """
        g0 = self.compute_gradient_base(x)
        g1 = self.compute_gradient_target(x)
        return (1.0 - lam) * g0 + lam * g1