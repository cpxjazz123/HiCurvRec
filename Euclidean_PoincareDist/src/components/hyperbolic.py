import torch
import torch.nn as nn
from typing import Optional


class PoincareExpmap(nn.Module):
    """Map Euclidean latent vectors to the Poincare ball of curvature c (M1).

    expmap_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
    """

    def __init__(self, curvature: float = 1.0, learnable: bool = False):
        super().__init__()
        self.register_buffer("_c", torch.tensor(curvature))
        self.learnable = learnable
        if learnable:
            self.curvature = nn.Parameter(torch.tensor(curvature))

    def get_curvature(self) -> torch.Tensor:
        if self.learnable:
            return torch.clamp(self.curvature, min=1e-4, max=10.0)
        return self._c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.get_curvature()
        sqrt_c = torch.sqrt(c)
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.tanh(sqrt_c * x_norm) / (sqrt_c * x_norm)
        return scale * x


class PoincareLogmap(nn.Module):
    """Inverse of expmap_0: map Poincare ball points back to the tangent space (M1 inverse)."""

    def __init__(self, curvature: float = 1.0, learnable: bool = False):
        super().__init__()
        self.register_buffer("_c", torch.tensor(curvature))
        self.learnable = learnable
        if learnable:
            self.curvature = nn.Parameter(torch.tensor(curvature))

    def get_curvature(self) -> torch.Tensor:
        if self.learnable:
            return torch.clamp(self.curvature, min=1e-4, max=10.0)
        return self._c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.get_curvature()
        sqrt_c = torch.sqrt(c)
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8, max=1.0 - 1e-6)
        scale = torch.atanh(sqrt_c * x_norm) / (sqrt_c * x_norm)
        return scale * x


class PoincareDistance(nn.Module):
    """Poincare ball distance with curvature c (M2 + M3).

    d(x, y) = (2 / sqrt(c)) * atanh(sqrt(c) * ||(-x) ⊕_c y||)
    where the Mobius addition is
      x ⊕_c y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) / (1 + 2c<x,y> + c^2||x||^2||y||^2)
    """

    def __init__(self, curvature: float = 1.0, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.curvature = nn.Parameter(torch.tensor(curvature))
        else:
            self.register_buffer("_c", torch.tensor(curvature))

    def get_curvature(self) -> torch.Tensor:
        if self.learnable:
            return torch.clamp(self.curvature, min=1e-4, max=10.0)
        return self._c

    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c.pow(2) * x_sq * y_sq
        return num / denom.clamp(min=1e-12)

    def compute(self, x: torch.Tensor, y: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """Distances of shape (n1, n2). x: (n1, d), y: (n2, d)."""
        assert x.dim() == 2 and y.dim() == 2 and x.size(1) == y.size(1)
        c = self.get_curvature()
        sqrt_c = torch.sqrt(c)
        # Project both x and y into the Poincare ball of curvature c.
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        y_norm = torch.norm(y, dim=-1, keepdim=True).clamp(min=1e-8)
        x = torch.tanh(sqrt_c * x_norm) / (sqrt_c * x_norm) * x
        y = torch.tanh(sqrt_c * y_norm) / (sqrt_c * y_norm) * y
        add = self.mobius_add(-x, y, c)
        add_norm = torch.norm(add, dim=-1)  # (n1, n2)
        dist = (2.0 / sqrt_c) * torch.atanh((sqrt_c * add_norm).clamp(max=1.0 - 1e-6))
        return dist


class PoincareExpmapAndDistance:
    """Combined module factory: expmap before quantization + poincare distance."""

    @staticmethod
    def build(expmap: bool, poincare_dist: bool, learnable_curvature: bool, curvature: float = 1.0):
        modules = []
        if expmap:
            modules.append(("expmap", PoincareExpmap(curvature, learnable_curvature)))
        return modules
