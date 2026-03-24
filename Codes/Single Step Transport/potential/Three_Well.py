import torch


# =========================================================================
# 1. JIT Compiled Calculation Kernel
# =========================================================================

@torch.jit.script
def _three_well_jit(x: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for Three-Well Potential.
    Optimized for PyTorch Autograd (Required for Energy KL Loss).

    Formula:
    V(x) = 4 * [ (x1^2 - 1)^2 + (x2^2 - 1)^2 + sin(x1 + 2*x2) ]

    Input:
        x: Tensor [N, 2]
    Output:
        V: Tensor [N]
    """
    x1 = x[:, 0]
    x2 = x[:, 1]

    # V(x) calculation
    term1 = (x1.square() - 1).square()
    term2 = (x2.square() - 1).square()
    term3 = torch.sin(x1 + 2 * x2)

    return 3.0 * (term1 + term2 + term3)


# =========================================================================
# 2. Potential Energy Class
# =========================================================================

class Three_Well:
    """
    THREE_WELL Potential Energy System

    Description:
        Defines the potential energy V(x) for the 2D Three-Well system.
        This class is designed for PyTorch-based training (e.g., Normalizing Flows)
        and provides the energy surface required for KL divergence minimization.
    """

    # ---- System Identification ----
    NAME = 'Three Well'
    ABBR = 'TW'
    DIM = 2

    # ---- Base Distribution (Gaussian) ----
    # Used for initializing the flow or defining the prior
    BASE_MEAN = 0.0
    BASE_SIGMA = 1.0

    # ---- Plotting/Computing Boundaries ----
    # Consistent with MATLAB visualization settings
    X_LIM_COMPUTE = [-4, 4]
    Y_LIM_COMPUTE = [-4, 4]
    X_LIM_PLOT = [-2, 2]
    Y_LIM_PLOT = [-2, 2]

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

        return _three_well_jit(x)