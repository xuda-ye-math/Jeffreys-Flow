import torch
import torch.nn as nn
import math

class Path_Integral_Potential(nn.Module):
    def __init__(self, N, beta=1.0):
        super().__init__()
        assert N % 2 == 0, "N must be an even number"
        self.N = N
        self.beta = beta
        self.SIGMA = 1.0
        
        # Build the transformation matrix C of shape [N, N]
        j = torch.arange(1, N + 1, dtype=torch.float32)
        C = torch.zeros(N, N, dtype=torch.float32)
        
        # k = 0
        C[:, 0] = 1.0 / math.sqrt(beta)
        
        # k = N - 1
        C[:, N - 1] = ((-1)**j) / math.sqrt(beta)
        
        # k = 1 ... N/2 - 1
        for k in range(1, N // 2):
            C[:, 2 * k - 1] = math.sqrt(2.0 / beta) * torch.sin(2 * math.pi * k * j / N)
            C[:, 2 * k] = math.sqrt(2.0 / beta) * torch.cos(2 * math.pi * k * j / N)
            
        self.register_buffer('C', C)
        self.register_buffer('C_T', C.transpose(0, 1))

        # Build the intrinsic frequencies OMEGA of shape [N]
        OMEGA = torch.zeros(N, dtype=torch.float32)
        # k = 0
        OMEGA[0] = 0.0
        # k = 1 ... N/2 - 1
        for k in range(1, N // 2):
            val = (2.0 * N / beta) * math.sin(k * math.pi / N)
            OMEGA[2 * k - 1] = val
            OMEGA[2 * k] = val
        # k = N/2 (only the N-1 mode)
        OMEGA[N - 1] = (2.0 * N / beta)
        
        self.register_buffer('OMEGA', OMEGA)
        
    def target_V(self, x):
        """
        V(x) = 20 * (x-1)^2 * (x+0.9) * (x+1.1)
        Input: x [Batch, N]
        Output: [Batch, N] (torch.float32)
        """
        return 20.0 * ((x - 1.0)**2) * (x + 0.9) * (x + 1.1)

    def target_Ux(self, x):
        """
        U(x_1...x_N) = N/(2*beta) * sum_{i=1}^N (x_i - x_{i+1})^2 + 1/N * sum_{i=1}^N V(x_i)
        Input: x [Batch, N]
        Output: [Batch] (torch.float32)
        """
        x_next = torch.roll(x, shifts=-1, dims=1)
        spring_potential = (self.N / (2.0 * self.beta)) * torch.sum((x - x_next)**2, dim=1)
        external_potential = (1.0 / self.N) * torch.sum(self.target_V(x), dim=1)
        return spring_potential + external_potential

    def target_Uf(self, xi):
        """
        U(xi_1...xi_N) = 1/2 * sum_{k=0}^{N-1} OMEGA_k^2 xi_k^2 + 1/N * sum_{i=1}^N V(x_i)
        Input: xi [Batch, N]
        Output: [Batch] (torch.float32)
        """
        x = self.f2x(xi)
        spring_potential = 0.5 * torch.sum((self.OMEGA ** 2) * (xi ** 2), dim=1)
        external_potential = (1.0 / self.N) * torch.sum(self.target_V(x), dim=1)
        return spring_potential + external_potential

    def base_V(self, x):
        """
        bV(x) = x^2 / (2 * SIGMA^2)
        Input: x [Batch, N]
        Output: [Batch, N] (torch.float32)
        """
        return (x ** 2) / (2.0 * self.SIGMA ** 2)

    def base_Ux(self, x):
        """
        base_U(x_1...x_N) = N/(2*beta) * sum_{i=1}^N (x_i - x_{i+1})^2 + 1/N * sum_{i=1}^N bV(x_i)
        Input: x [Batch, N]
        Output: [Batch] (torch.float32)
        """
        x_next = torch.roll(x, shifts=-1, dims=1)
        spring_potential = (self.N / (2.0 * self.beta)) * torch.sum((x - x_next)**2, dim=1)
        external_potential = (1.0 / self.N) * torch.sum(self.base_V(x), dim=1)
        return spring_potential + external_potential

    def base_Uf(self, xi):
        """
        base_U(xi_1...xi_N) = 1/2 * sum_{k=0}^{N-1} OMEGA_k^2 xi_k^2 + 1/N * sum_{i=1}^N bV(x_i)
        Input: xi [Batch, N]
        Output: [Batch] (torch.float32)
        """
        x = self.f2x(xi)
        spring_potential = 0.5 * torch.sum((self.OMEGA ** 2) * (xi ** 2), dim=1)
        external_potential = (1.0 / self.N) * torch.sum(self.base_V(x), dim=1)
        return spring_potential + external_potential

    def mixed_Uf(self, xi, lam):
        """
        mixed_U(xi_1...xi_N) = (1 - lam) * base_Uf + lam * target_Uf
        Input: xi [Batch, N], lam (float or Tensor)
        Output: [Batch] (torch.float32)
        """
        x = self.f2x(xi)
        spring_potential = 0.5 * torch.sum((self.OMEGA ** 2) * (xi ** 2), dim=1)
        mixed_V = (1.0 - lam) * self.base_V(x) + lam * self.target_V(x)
        external_potential = (1.0 / self.N) * torch.sum(mixed_V, dim=1)
        return spring_potential + external_potential

    def x2f(self, x):
        """
        position to Fourier coefficients
        Input: x [Batch, N]
        Output: xi [Batch, N] (torch.float32)
        """
        return (self.beta / self.N) * torch.matmul(x, self.C)

    def f2x(self, xi):
        """
        Fourier coefficients to position
        Input: xi [Batch, N]
        Output: x [Batch, N] (torch.float32)
        """
        return torch.matmul(xi, self.C_T)

if __name__ == "__main__":
    N = 8
    beta = 1.0
    model = Path_Integral_Potential(N=N, beta=beta)
    
    x = torch.randn(3, N)
    xi = model.x2f(x)
    x_rec = model.f2x(xi)
    
    print("Reconstruction error:", torch.max(torch.abs(x - x_rec)).item())
    
    target_Ux_val = model.target_Ux(x)
    target_Uf_val = model.target_Uf(xi)
    
    print("target_Ux(x) shape:", target_Ux_val.shape)
    print("target_Ux(x) values:", target_Ux_val)
    print("target_Uf(xi) values:", target_Uf_val)
    print("Max difference between target_Ux and target_Uf:", torch.max(torch.abs(target_Ux_val - target_Uf_val)).item())

    base_Ux_val = model.base_Ux(x)
    base_Uf_val = model.base_Uf(xi)
    print("Max difference between base_Ux and base_Uf:", torch.max(torch.abs(base_Ux_val - base_Uf_val)).item())

    # Test mixed_Uf
    lam = 0.3
    mixed_Uf_direct = (1.0 - lam) * base_Uf_val + lam * target_Uf_val
    mixed_Uf_method = model.mixed_Uf(xi, lam)
    print("Max difference for mixed_Uf equivalence:", torch.max(torch.abs(mixed_Uf_direct - mixed_Uf_method)).item())
