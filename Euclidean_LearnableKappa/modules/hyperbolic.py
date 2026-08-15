"""双曲机制 M3: LearnableKappa — Poincare 距离 + 可学习曲率 κ.

与 M2 (PoincareDist) 同公式, 但曲率 κ 改成 nn.Parameter (learnable),
通过 sigmoid 限幅到 [1e-4, 10.0] 范围内, 由 Stage2 RQ-VAE 训练优化.

机制核心: 让 κ 适应数据分布, 在 Stage2 训练时学习最优曲率.
"""
import torch
from torch import nn


class LearnableKappaPoincareDistance(nn.Module):
    """Poincare distance with learnable curvature kappa (M3).

    kappa is a learnable parameter clamped to [1e-4, 10.0].
    """

    def __init__(self, init_curvature: float = 1.0):
        super().__init__()
        self.curvature = nn.Parameter(torch.tensor(init_curvature))

    def get_curvature(self) -> torch.Tensor:
        return torch.clamp(self.curvature, min=1e-4, max=10.0)

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
        c = self.get_curvature()
        sqrt_c = torch.sqrt(c)
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        y_norm = torch.norm(y, dim=-1, keepdim=True).clamp(min=1e-8)
        x_proj = torch.tanh(sqrt_c * x_norm) / (sqrt_c * x_norm) * x
        y_proj = torch.tanh(sqrt_c * y_norm) / (sqrt_c * y_norm) * y
        add = self.mobius_add(-x_proj, y_proj, c)
        add_norm = torch.norm(add, dim=-1)
        dist = (2.0 / sqrt_c) * torch.atanh((sqrt_c * add_norm).clamp(max=1.0 - 1e-6))
        return dist

    def get_kappa(self) -> float:
        return self.get_curvature().item()
