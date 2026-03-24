import torch


# =========================================================================
# 1. JIT Compiled Calculation Kernel
# =========================================================================

@torch.jit.script
def _multiple_well_jit(x: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for Multiple Well Potential.
    Optimized for PyTorch Autograd (Required for Energy KL Loss).

    Formula:
    U(x,y) = 4*sin(1.1*x)*sin(y) + 0.25*x^2 + 0.25*y^2

    Input:
        x: Tensor [N, 2]
    Output:
        V: Tensor [N]
    """
    x1 = x[:, 0]
    x2 = x[:, 1]

    # Term 1: 4 * sin(1.1*x) * sin(y)
    term_sin = 4.0 * torch.sin(1.1 * x1) * torch.sin(x2)

    # Term 2: 0.25 * x^2 + 0.25 * y^2
    term_quad = 0.25 * (x1.square() + x2.square())

    return term_sin + term_quad


# =========================================================================
# 2. Potential Energy Class
# =========================================================================

class Multiple_Well:
    """
    MULTIPLE_WELL Potential Energy System

    Description:
        Defines the potential energy V(x) for the 2D Multiple Well function.
        This class is designed for PyTorch-based training (e.g., Normalizing Flows)
        and provides the energy surface required for KL divergence minimization.
    """

    # ---- System Identification ----
    NAME = 'Multiple Well'
    ABBR = 'MW'
    DIM = 2

    # ---- Base Distribution (Gaussian) ----
    # Used for initializing the flow or defining the prior
    # Matches SIGMA = 2.2 from Multiple_Well.m
    BASE_MEAN = 0.0
    BASE_SIGMA = 2.2

    # ---- Plotting/Computing Boundaries ----
    # Consistent with MATLAB Multiple_Well.m visualization settings
    X_LIM_COMPUTE = [-7, 7]
    Y_LIM_COMPUTE = [-7, 7]
    X_LIM_PLOT = [-5, 5]
    Y_LIM_PLOT = [-5, 5]

    def __init__(self, device: str = 'cpu'):
        """
        Constructor: Sets up the device for potential calculation.

        Input:
            device: 'cpu' or 'cuda'
        """
        self.device = device

    def compute_potential(self, x: torch.Tensor) -> torch.Tensor:
        """
        COMPUTE_POTENTIAL Calculates V(x)

        Input:
            x : Tensor [N, DIM] (Batch of coordinates)

        Output:
            V : Tensor [N] (Potential energy values)
        """
        # Ensure input is on the correct device
        if x.device != torch.device(self.device):
            x = x.to(self.device)

        return _multiple_well_jit(x)