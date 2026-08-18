"""HALC v4: Hyperbolic FFN 注入 (R52 主目录迭代备胎, v3 升级).

在 HALC v2/v3 (仅在 encoder hidden_states 加 reg_loss) 之上升级:
1. 替换 encoder FFN 为 Hyperbolic FFN: logmap → linear → expmap (tangent space aggregation)
2. 引入 learnable per-layer FFN curvature κ_ffn ∈ R^L
3. 同时保留 v2/v3 的 reg_loss (几何正则)

文献支撑:
- Hypformer (NeurIPS 2024) full-stack hyperbolic transformer — FFN/Attention 全部双曲化
- HRec (ICLR 2025) tangent-space aggregation (log map → linear → exp map)

数学公式 (Hyperbolic FFN):
- x_logmap = logmap0(x, c_ffn)  # Poincaré ball → tangent space
- y_tangent = LinearFFN(x_logmap)  # 欧氏 FFN on tangent space
- y_ball = expmap0(y_tangent, c_ffn)  # tangent → Poincaré ball

预期: 较 HALC v2 (test_R@10=0.1072) 进一步 +0.01~0.02 → test_R@10 0.115-0.13

注意: 由于 v4 需要改 HG_Rec T5 架构, 实施复杂度高. 此模块先准备, 待 v3 R37 决策后决定是否启 v4.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicFFN(nn.Module):
    """Hyperbolic FFN: logmap → Linear → expmap, 可替换 HG_Rec encoder FFN."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        learnable_curvature: bool = True,
        init_curvature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        # learnable per-layer κ_ffn
        if learnable_curvature:
            self.log_curvature = nn.Parameter(
                torch.tensor(math.log(math.expm1(init_curvature)))
            )
        else:
            self.register_buffer("log_curvature", torch.tensor(math.log(math.expm1(init_curvature))))
        # standard FFN linear (在 tangent space)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    @property
    def curvature(self) -> torch.Tensor:
        return F.softplus(self.log_curvature) + 1e-5

    def poincare_logmap0(self, x: torch.Tensor) -> torch.Tensor:
        c = self.curvature
        sqrt_c = torch.sqrt(c)
        norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        max_norm = (1.0 / sqrt_c - 1e-5)
        norm_x_clamp = norm_x.clamp_max(max_norm)
        factor = torch.arctanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)
        return factor * x

    def poincare_expmap0(self, v: torch.Tensor) -> torch.Tensor:
        c = self.curvature
        sqrt_c = torch.sqrt(c)
        norm_v = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        max_norm = (1.0 / sqrt_c - 1e-5)
        norm_v_clamp = norm_v.clamp_max(max_norm)
        factor = torch.tanh(sqrt_c * norm_v_clamp) / (sqrt_c * norm_v)
        return factor * v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) 在 Poincaré ball
        x_logmap = self.poincare_logmap0(x)  # → tangent space
        y_tangent = self.linear2(F.relu(self.linear1(x_logmap)))  # FFN
        y_tangent = self.dropout(y_tangent)
        y_ball = self.poincare_expmap0(y_tangent)  # → Poincaré ball
        return y_ball