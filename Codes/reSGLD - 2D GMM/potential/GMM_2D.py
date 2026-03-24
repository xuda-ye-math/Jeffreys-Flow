import torch
import math


# =========================================================================
# 1. JIT Compiled Calculation Kernel (Potential 2D)
# =========================================================================

@torch.jit.script
def _gmm_2d_jit(x: torch.Tensor,
                means: torch.Tensor,
                precs: torch.Tensor,
                log_dets: torch.Tensor,
                log_weights: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for 2D Anisotropic GMM Potential.
    Input: x [N, 2]
    Output: V [N] (Potential energy)
    """
    N = x.shape[0]
    K = means.shape[0]

    # Constants: log(2*pi) * 2 / 2
    const_term = -0.5 * 2.0 * math.log(2.0 * math.pi)

    # Store log probabilities for each component [N, K]
    log_probs = torch.zeros((N, K), dtype=x.dtype, device=x.device)

    for k in range(K):
        # 1. Diff: (x - mu_k) -> [N, 2]
        diff = x - means[k]

        # 2. Mahalanobis Term: -0.5 * (x-mu)^T * P * (x-mu)
        P = precs[k]  # [2, 2]
        term = torch.mm(diff, P)  # [N, 2]
        mahalanobis = -0.5 * torch.sum(term * diff, dim=1)

        # 3. Log Gaussian PDF
        log_gauss = const_term - 0.5 * log_dets[k] + mahalanobis

        # 4. Log Joint: log(w_k) + log N
        log_probs[:, k] = log_weights[k] + log_gauss

    # LogSumExp across components -> [N]
    log_p = torch.logsumexp(log_probs, dim=1)

    # U(x) = -log p(x)
    return -log_p


# =========================================================================
# 2. JIT Compiled Calculation Kernel (Gradient 2D)
# =========================================================================

@torch.jit.script
def _gmm_2d_grad_jit(x: torch.Tensor,
                     means: torch.Tensor,
                     precs: torch.Tensor,
                     log_dets: torch.Tensor,
                     log_weights: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for Gradient of 2D Anisotropic GMM Potential.
    """
    N = x.shape[0]
    K = means.shape[0]
    const_term = -0.5 * 2.0 * math.log(2.0 * math.pi)

    # 1. Compute Log Probabilities
    log_probs = torch.zeros((N, K), dtype=x.dtype, device=x.device)

    for k in range(K):
        diff = x - means[k]
        P = precs[k]
        term = torch.mm(diff, P)
        mahalanobis = -0.5 * torch.sum(term * diff, dim=1)
        log_gauss = const_term - 0.5 * log_dets[k] + mahalanobis
        log_probs[:, k] = log_weights[k] + log_gauss

    # 2. Compute Responsibilities
    log_p = torch.logsumexp(log_probs, dim=1, keepdim=True)
    responsibilities = torch.exp(log_probs - log_p)

    # 3. Compute Weighted Gradient
    grad = torch.zeros_like(x)

    for k in range(K):
        diff = x - means[k]
        P = precs[k]
        grad_comp = torch.mm(diff, P)
        w = responsibilities[:, k].unsqueeze(1)
        grad += w * grad_comp

    return grad


# =========================================================================
# 3. Potential Energy Class
# =========================================================================

class GMM_2D:
    """
    2D ANISOTROPIC GMM Potential Energy System.
    Strictly aligned with GMM_2D.m
    """

    NAME = '2D Anisotropic GMM'
    ABBR = 'GM'
    DIM = 2

    # Base Distribution Parameters
    BASE_MEAN = 0.0
    BASE_SIGMA = 1.5
    NOISE = 4.0

    # Rejuvenation Parameters (Default for this potential)
    DT = 1e-3
    N_ITER = 2000

    # Computing/Plotting Boundaries
    X_LIM_PLOT = [-5, 5]
    Y_LIM_PLOT = [-5, 5]
    
    X_LIM_COMPUTE = [-10, 10]
    Y_LIM_COMPUTE = [-10, 10]
    
    # 3D Limits Removed as requested

    def __init__(self, device: str = 'cpu'):
        self.device = device

        # Initialize Gaussian components
        self._init_target_parameters()

    def to(self, device):
        """
        Moves all internal tensors to the specified device.
        Mimics the behavior of nn.Module.to(device).
        """
        self.device = device

        # Move GMM Parameters
        if hasattr(self, 'means'):
            self.means = self.means.to(device)
            self.precs = self.precs.to(device)
            self.log_dets = self.log_dets.to(device)
            self.log_weights = self.log_weights.to(device)

        return self

    def _init_target_parameters(self):
        """
        Constructs 4 anisotropic Gaussian components (Irregular Wells).
        Matches GMM_2D.m exactly.
        """
        K = 4
        
        # Means (Centers)
        # 1: Bottom Left, 2: Bottom Right, 3: Top Left, 4: Top Right
        means_list = [
            [-1.2, -1.2], 
            [1.3, -0.8], 
            [-0.5, 1.4], 
            [1.2, 1.3]
        ]
        
        # Covariances (Shapes)
        covs_list = [
            [[0.06, 0.02], [0.02, 0.06]],
            [[0.08, -0.04], [-0.04, 0.12]],
            [[0.05, 0.0], [0.0, 0.03]],
            [[0.04, 0.01], [0.01, 0.04]]
        ]
        
        # Weights
        weights_vals = [0.25, 0.3, 0.15, 0.3]
        
        precs_list = []
        log_dets_list = []

        for k in range(K):
            Sigma = torch.tensor(covs_list[k], dtype=torch.float32)
            Prec = torch.inverse(Sigma)
            log_det = torch.log(torch.det(Sigma))

            precs_list.append(Prec)
            log_dets_list.append(log_det)

        # Move to initial device
        self.means = torch.tensor(means_list, dtype=torch.float32).to(self.device)
        self.precs = torch.stack(precs_list).to(self.device)
        self.log_dets = torch.stack(log_dets_list).to(self.device)

        weights = torch.tensor(weights_vals, dtype=torch.float32)
        # Normalize weights explicitly just in case, though they sum to 1.0
        weights = weights / weights.sum()
        self.log_weights = torch.log(weights).to(self.device)

    # ---- Potential Functions ----

    def compute_potential_base(self, x: torch.Tensor) -> torch.Tensor:
        """
        U_base: Gaussian Potential
        U(x) = 0.5 * ||(x - mu)/sigma||^2
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        z = (x - self.BASE_MEAN) / self.BASE_SIGMA
        return 0.5 * torch.sum(z ** 2, dim=1)

    def compute_potential_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        U_target: Pure GMM Potential
        """
        if x.device != torch.device(self.device): x = x.to(self.device)
        return _gmm_2d_jit(x, self.means, self.precs, self.log_dets, self.log_weights)

    def compute_potential_mixed(self, x: torch.Tensor, lam: float, i_vec: torch.Tensor = None) -> torch.Tensor:
        """
        U_mix = (1-lam)*U_base + lam*U_target + i1*cos(x1) + i2*cos(x2)
        [UPDATED] Injection of Random Noise (Stochastic Potential) - Independent per Particle
        """
        u0 = self.compute_potential_base(x)
        u1 = self.compute_potential_target(x)

        # 2. Compute Perturbation: i1*cos(x1) + i2*cos(x2)
        if i_vec is not None:
            # Using broadcasting: x is [N, 2], i_vec is [N, 2]
            perturbation = torch.sum(i_vec * torch.cos(x), dim=1)
        else:
            perturbation = 0.0

        return (1.0 - lam) * u0 + lam * u1 + perturbation

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
        return _gmm_2d_grad_jit(x, self.means, self.precs, self.log_dets, self.log_weights)

    def compute_gradient_mixed(self, x: torch.Tensor, lam: float, i_vec: torch.Tensor = None) -> torch.Tensor:
        """
        grad U_mix = (1-lam)*grad_U_base + lam*grad_U_target + grad_perturbation
        grad_perturbation: [-i1*sin(x1), -i2*sin(x2)]
        [UPDATED] Injection of Random Noise (Stochastic Gradient) - Independent per Particle
        """
        g0 = self.compute_gradient_base(x)
        g1 = self.compute_gradient_target(x)
        
        # Gradient of i*cos(x) is -i*sin(x)
        if i_vec is not None:
            grad_perturbation = -i_vec * torch.sin(x)
        else:
            grad_perturbation = 0.0
        
        return (1.0 - lam) * g0 + lam * g1 + grad_perturbation