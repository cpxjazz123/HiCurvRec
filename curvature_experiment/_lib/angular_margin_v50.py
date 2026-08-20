"""v50 Inter-Layer Angular Margin Regularization (R36 新曲率正则项, 几何变换).

设计原则 (R36 严格化 v2, user memory feedback):
- 不引入 c_l schedule / weight decay / variance reg (v40-v49 全部 R37 FAIL)
- 走几何不变量: 强制 encoder 6 层 hidden states 之间的 Poincaré 距离 > margin
- margin = 0.1 (硬编码, R30/R43)
- reg_weight = 0.01 (硬编码, R30/R43)
- 完全几何变换, 不引入 schedule/weight, 不引入 valid 偏置

几何解释: encoder 不同层应有 geometric separation (避免 layer collapse),
这是 Poincaré 球面几何不变量, 与 HALC v2 σ_field (magnitude) 互补.
"""
import math
import torch
import torch.nn as nn


def poincare_distance(x, y, c=1.0, eps=1e-9):
    """Poincaré ball distance: d(x, y) = arccosh(1 + 2c·||x-y||² / ((1-c·||x||²)(1-c·||y||²))) / sqrt(c)."""
    x_sq = (x * x).sum(dim=-1)
    y_sq = (y * y).sum(dim=-1)
    diff_sq = ((x - y) ** 2).sum(dim=-1)
    denom = (1.0 - c * x_sq).clamp_min(eps) * (1.0 - c * y_sq).clamp_min(eps)
    arg = 1.0 + 2.0 * c * diff_sq / denom
    arg = arg.clamp(min=1.0 + eps)
    sqrt_c = math.sqrt(c)
    return torch.acosh(arg) / sqrt_c


def project_to_poincare_ball(x, radius=0.3, c=1.0, eps=1e-9):
    """Project x to Poincaré ball of radius_max_norm * sqrt(1/c).

    Inputs: x is arbitrary (B, L, D) hidden states.
    Output: same shape, each point within Poincaré ball.
    """
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (radius / math.sqrt(c)) * 0.95
    scale = (max_norm / norm).clamp_max(1.0)
    return x * scale


class InterLayerAngularMarginLoss(nn.Module):
    """Inter-layer angular margin loss in Poincaré ball.

    For each consecutive layer pair from encoder hidden_states,
    compute mean-pooled representations and force Poincaré distance
    to be > margin (geometric separation).

    这是 R36 新曲率正则项 (几何变换), 不引入 schedule, 不引入 valid 偏置.
    """

    def __init__(self, radius: float = 0.3, c: float = 1.0, margin: float = 0.1,
                 reg_weight: float = 0.01):
        super().__init__()
        self.radius = radius
        self.c = c
        self.margin = margin
        self.reg_weight = reg_weight

    def forward(self, enc_hs_list):
        """Compute inter-layer angular margin loss.

        Args:
            enc_hs_list: list of (B, L, D) encoder hidden states per layer
        Returns:
            loss: scalar tensor, inter-layer angular margin loss
        """
        if len(enc_hs_list) < 2:
            return torch.tensor(0.0, device=enc_hs_list[0].device if enc_hs_list else "cpu")
        # Mean-pool each layer to (B, D)
        layer_means = []
        for hs in enc_hs_list:
            pooled = hs.mean(dim=1)  # (B, D)
            # Project to Poincaré ball
            pooled_proj = project_to_poincare_ball(pooled, radius=self.radius, c=self.c)
            layer_means.append(pooled_proj)
        # Inter-layer margin
        margin_loss = 0.0
        n_pairs = 0
        for i in range(len(layer_means) - 1):
            d = poincare_distance(layer_means[i], layer_means[i + 1], c=self.c)  # (B,)
            # Want d > margin
            margin_loss = margin_loss + torch.relu(self.margin - d).mean()
            n_pairs += 1
        if n_pairs == 0:
            return torch.tensor(0.0, device=enc_hs_list[0].device)
        margin_loss = margin_loss / n_pairs
        return self.reg_weight * margin_loss
