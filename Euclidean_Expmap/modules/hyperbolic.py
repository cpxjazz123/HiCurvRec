"""双曲机制 M1: Expmap — 把欧氏 latent 通过 expmap0 映射到 Poincare 球.

expmap_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)

之后用 L2 距离做量化 (与欧氏一致),但 input space 是 Poincare ball.
"""
import torch
from torch import nn


class PoincareExpmap(nn.Module):
    """Map Euclidean latent vectors to the Poincare ball of curvature c (M1)."""

    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.register_buffer("c", torch.tensor(curvature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sqrt_c = torch.sqrt(self.c)
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.tanh(sqrt_c * x_norm) / (sqrt_c * x_norm)
        return scale * x
