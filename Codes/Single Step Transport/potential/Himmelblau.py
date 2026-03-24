import torch


# =========================================================================
# 1. JIT Compiled Calculation Kernel
# =========================================================================

@torch.jit.script
def _himmelblau_jit(x: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for Himmelblau Potential.
    Optimized for PyTorch Autograd (Required for Energy KL Loss).

    Formula:
    V(x) = 0.2 * [ (x^2 + y - 11)^2 + (x + y^2 - 7)^2 ]

    Input:
        x: Tensor [N, 2]
    Output:
        V: Tensor [N]
    """
    x1 = x[:, 0]
    x2 = x[:, 1]

    # V(x) calculation
    term1 = (x1.square() + x2 - 11).square()
    term2 = (x1 + x2.square() - 7).square()

    # Scaling factor from Himmelblau.m
    return 0.2 * (term1 + term2)


# =========================================================================
# 2. Potential Energy Class
# =========================================================================

class Himmelblau:
    """
    HIMMELBLAU Potential Energy System

    Description:
        Defines the potential energy V(x) for the 2D Himmelblau function.
        This class is designed for PyTorch-based training (e.g., Normalizing Flows)
        and provides the energy surface required for KL divergence minimization.
    """

    # ---- System Identification ----
    NAME = 'Himmelblau'
    ABBR = 'HB'
    DIM = 2

    # ---- Base Distribution (Gaussian) ----
    # Used for initializing the flow or defining the prior
    # Note: SIGMA is 2.0 here (wider) compared to 1.0 in Three_Well, matching Himmelblau.m
    BASE_MEAN = 0.0
    BASE_SIGMA = 2.0

    # ---- Plotting/Computing Boundaries ----
    # Consistent with MATLAB Himmelblau.m visualization settings
    X_LIM_COMPUTE = [-6, 6]
    Y_LIM_COMPUTE = [-6, 6]
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

        return _himmelblau_jit(x)