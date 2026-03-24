import torch


# =========================================================================
# 1. JIT Compiled Calculation Kernel
# =========================================================================

@torch.jit.script
def _annulus_jit(x: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for Annulus Potential.
    Optimized for PyTorch Autograd (Required for Energy KL Loss).

    Formula:
    V(x) = 10 * ( r^6 - 8r^4 + 16r^2 + 1 )^(1/3)
    where r^2 = x1^2 + x2^2

    Input:
        x: Tensor [N, 2]
    Output:
        V: Tensor [N]
    """
    # Calculate r^2
    r2 = x.square().sum(dim=1)

    # Inner term: r^6 - 8r^4 + 16r^2 + 1
    # Note: r^6 - 8r^4 + 16r^2 = r^2(r^2 - 4)^2 >= 0, so term >= 1
    # (Safe for cube root)
    inner_term = r2.pow(3) - 8 * r2.pow(2) + 16 * r2 + 1

    # Scaling factor matches MATLAB code (10)
    return 10.0 * inner_term.pow(1 / 3)


# =========================================================================
# 2. Potential Energy Class
# =========================================================================

class Annulus:
    """
    ANNULUS Potential Energy System

    Description:
        Defines the potential energy V(x) for the Annulus potential.
        This class is designed for PyTorch-based training (e.g., Normalizing Flows)
        and provides the energy surface required for KL divergence minimization.
    """

    # ---- System Identification ----
    NAME = 'Annulus'
    ABBR = 'AN'
    DIM = 2

    # ---- Base Distribution (Gaussian) ----
    # Used for initializing the flow or defining the prior
    # Matches SIGMA = 1 from Annulus.m
    BASE_MEAN = 0.0
    BASE_SIGMA = 1.0

    # ---- Plotting/Computing Boundaries ----
    # Consistent with MATLAB Annulus.m visualization settings
    X_LIM_COMPUTE = [-4, 4]
    Y_LIM_COMPUTE = [-4, 4]
    X_LIM_PLOT = [-3, 3]
    Y_LIM_PLOT = [-3, 3]

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

        return _annulus_jit(x)