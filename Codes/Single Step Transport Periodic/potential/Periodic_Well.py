import torch
import math


# =========================================================================
# 1. JIT Compiled Calculation Kernel
# =========================================================================

@torch.jit.script
def _periodic_well_jit(x: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for Periodic Well Potential.
    Optimized for PyTorch Autograd.

    Formula:
    V(x,y) = 4 * sin(2*x) * ((sin(2*y))^7)^(1/5)

    Implementation:
    V = 4 * sin(2x) * sign(sin(2y)) * |sin(2y)|^1.4
    This handles negative bases raised to fractional powers correctly on GPU/CPU.

    Input:
        x: Tensor [N, 2]
    Output:
        V: Tensor [N]
    """
    x1 = x[:, 0]
    x2 = x[:, 1]

    sin_2x1 = torch.sin(2 * x1)
    sin_2x2 = torch.sin(2 * x2)

    # Safe fractional power implementation:
    # 1.4 = 7/5. Ideally define on Real set.
    # torch.pow(negative, fractional) returns NaN, so we use abs() then re-apply sign.
    term_y = torch.sign(sin_2x2) * torch.abs(sin_2x2).pow(1.4)

    return 4.0 * sin_2x1 * term_y


# =========================================================================
# 2. Potential Energy Class
# =========================================================================

class Periodic_Well:
    """
    PERIODIC_WELL Potential Energy System

    Description:
        Defines the potential energy V(x) for the 2D Periodic system on [-pi, pi].
        Features a gap of approx 8.
    """

    # ---- System Identification ----
    NAME = 'Periodic Well'
    ABBR = 'PW'
    DIM = 2

    # ---- Plotting/Computing Boundaries ----
    # Consistent with MATLAB settings [-pi, pi]
    X_LIM_PLOT = [-math.pi, math.pi]
    Y_LIM_PLOT = [-math.pi, math.pi]

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

        return _periodic_well_jit(x)