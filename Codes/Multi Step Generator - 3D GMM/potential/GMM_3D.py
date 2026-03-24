import torch
import math


# =========================================================================
# 1. JIT Compiled Calculation Kernel (Potential)
# =========================================================================

@torch.jit.script
def _gmm_3d_jit(x: torch.Tensor,
                means: torch.Tensor,
                precs: torch.Tensor,
                log_dets: torch.Tensor,
                log_weights: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for 3D Anisotropic GMM Potential.
    Input: x [N, 3]
    Output: V [N] (Potential energy)
    """
    N = x.shape[0]
    K = means.shape[0]

    # Constants: log(2*pi) * 3 / 2
    const_term = -0.5 * 3.0 * math.log(2.0 * math.pi)

    # Store log probabilities for each component [N, K]
    log_probs = torch.zeros((N, K), dtype=x.dtype, device=x.device)

    for k in range(K):
        # 1. Diff: (x - mu_k) -> [N, 3]
        diff = x - means[k]

        # 2. Mahalanobis Term: -0.5 * (x-mu)^T * P * (x-mu)
        P = precs[k]  # [3, 3]
        term = torch.mm(diff, P)  # [N, 3]
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
# 2. JIT Compiled Calculation Kernel (Gradient)
# =========================================================================

@torch.jit.script
def _gmm_3d_grad_jit(x: torch.Tensor,
                     means: torch.Tensor,
                     precs: torch.Tensor,
                     log_dets: torch.Tensor,
                     log_weights: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled kernel for Gradient of 3D Anisotropic GMM Potential.
    """
    N = x.shape[0]
    K = means.shape[0]
    const_term = -0.5 * 3.0 * math.log(2.0 * math.pi)

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

class GMM_3D:
    """
    3D ANISOTROPIC GMM Potential Energy System.
    Note: Beta Ladder configuration is now handled externally (parameters.py).
    """

    NAME = '3D Anisotropic GMM'
    ABBR = 'GM'
    DIM = 3

    # Base Distribution Parameters
    BASE_MEAN = 0.0
    BASE_SIGMA = 2.5

    # Rejuvenation Parameters (Default for this potential)
    DT = 5e-3
    N_ITER = 400

    # Computing/Plotting Boundaries
    X_LIM_PLOT = [-8, 8]
    Y_LIM_PLOT = [-8, 8]
    Z_LIM_PLOT = [-8, 8]
    X_LIM_COMPUTE = [-12, 12]
    Y_LIM_COMPUTE = [-12, 12]
    Z_LIM_COMPUTE = [-12, 12]

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
        Constructs 6 anisotropic Gaussian components in an octahedral layout.
        """
        K = 6
        L = 6.0

        means_list = [
            [L, 0, 0], [-L, 0, 0], [0, L, 0], [0, -L, 0], [0, 0, L], [0, 0, -L]
        ]
        shapes = [
            [0.3, 0.3, 0.8], [0.3, 0.8, 0.3], [0.8, 0.3, 0.3],
            [0.3, 0.6, 0.6], [0.4, 0.4, 0.4], [0.25, 0.8, 0.25]
        ]
        angles_deg = [
            [0, 0, 0], [45, 45, 0], [30, 0, 30],
            [90, 45, 0], [15, 15, 15], [60, 0, -60]
        ]

        precs_list = []
        log_dets_list = []

        for k in range(K):
            lam_vals = [s ** 2 for s in shapes[k]]
            Lambda = torch.diag(torch.tensor(lam_vals, dtype=torch.float32))

            phi, theta, psi = [math.radians(a) for a in angles_deg[k]]

            Rz = torch.tensor([[math.cos(phi), -math.sin(phi), 0],
                               [math.sin(phi), math.cos(phi), 0],
                               [0, 0, 1]], dtype=torch.float32)
            Ry = torch.tensor([[math.cos(theta), 0, math.sin(theta)],
                               [0, 1, 0],
                               [-math.sin(theta), 0, math.cos(theta)]], dtype=torch.float32)
            Rx = torch.tensor([[1, 0, 0],
                               [0, math.cos(psi), -math.sin(psi)],
                               [0, math.sin(psi), math.cos(psi)]], dtype=torch.float32)

            R = torch.mm(Rz, torch.mm(Ry, Rx))
            Sigma = torch.mm(R, torch.mm(Lambda, R.t()))
            Prec = torch.inverse(Sigma)
            log_det = torch.log(torch.det(Sigma))

            precs_list.append(Prec)
            log_dets_list.append(log_det)

        # Move to initial device
        self.means = torch.tensor(means_list, dtype=torch.float32).to(self.device)
        self.precs = torch.stack(precs_list).to(self.device)
        self.log_dets = torch.stack(log_dets_list).to(self.device)

        weights = torch.ones(K, dtype=torch.float32) / K
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
        return _gmm_3d_jit(x, self.means, self.precs, self.log_dets, self.log_weights)

    def compute_potential_mixed(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        U_mix = (1-lam)*U_base + lam*U_target
        Used for linear interpolation training.
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
        return _gmm_3d_grad_jit(x, self.means, self.precs, self.log_dets, self.log_weights)

    def compute_gradient_mixed(self, x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        grad U_mix = (1-lam)*grad_U_base + lam*grad_U_target
        """
        g0 = self.compute_gradient_base(x)
        g1 = self.compute_gradient_target(x)
        return (1.0 - lam) * g0 + lam * g1