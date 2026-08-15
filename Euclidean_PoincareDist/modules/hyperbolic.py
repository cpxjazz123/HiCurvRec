"""双曲机制 M2: PoincareDist — 用 Poincare 球距离替代 L2 做量化.

d(x, y) = (2 / sqrt(c)) * atanh(sqrt(c) * ||(-x) ⊕_c y||)
其中 Möbius 加法:
  x ⊕_c y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) / (1 + 2c<x,y> + c^2||x||^2||y||^2)

input 与 codebook 都先自动 project 到 Poincare ball 内部.
"""
import torch
from torch import nn


class PoincareDistance(nn.Module):
    """Poincare ball distance with curvature c (M2)."""

    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.register_buffer("c", torch.tensor(curvature))

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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Distances of shape (n1, n2). x: (n1, d), y: (n2, d)."""
        assert x.dim() == 2 and y.dim() == 2 and x.size(1) == y.size(1)
        sqrt_c = torch.sqrt(self.c)
        # 自动 project 到 Poincare ball 内部
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        y_norm = torch.norm(y, dim=-1, keepdim=True).clamp(min=1e-8)
        x_proj = torch.tanh(sqrt_c * x_norm) / (sqrt_c * x_norm) * x
        y_proj = torch.tanh(sqrt_c * y_norm) / (sqrt_c * y_norm) * y
        add = self.mobius_add(-x_proj, y_proj, self.c)
        add_norm = torch.norm(add, dim=-1)  # (n1, n2)
        dist = (2.0 / sqrt_c) * torch.atanh((sqrt_c * add_norm).clamp(max=1.0 - 1e-6))
        return dist
