"""HALC v5: Lorentz Model 备胎 (R36 曲率机制变更: Poincaré → Lorentz).

设计:
1. 用 Lorentz hyperboloid model L^d_c 替代 Poincaré ball B^d_c
2. learnable per-layer κ_l ∈ R^L
3. logmap0 在 Lorentz 原点 o=(1/√c, 0,..., 0) 计算
4. 保留 v2 curvature annealing + 强 reg

Lorentz Model 优势 (vs Poincaré):
- 数值稳定: 内积 -x_0^2 + Σx_i^2 = -1/c 严格满足, 无边界 clamp 问题
- Hypformer (NeurIPS 2024) + H2H-GPT (arXiv 2501.03221) 都用 Lorentz
- 训练更稳定, 适合深 Transformer (HG-Rec T5 6 层 + 4 解码层)

数学公式 (Lorentz hyperboloid L^d_c):
- 流形: H^d_c = {x ∈ R^(d+1) : ⟨x, x⟩_L = -1/c, x_0 > 0}
  其中 ⟨u, v⟩_L = -u_0*v_0 + Σu_i*v_i (Minkowski 内积)
- 原点: o = (1/√c, 0, ..., 0) ∈ R^(d+1)
- Poincaré → Lorentz: x_lorentz = (√(1 + c*||x||^2), x) / √c
- logmap_o: v_i = arcosh(c*x_0) * x_i / (√c * ||x_spatial||)
- expmap_o(v): y_0 = cosh(√c*||v||) / √c, y_i = sinh(√c*||v||)*v_i/(√c*||v||)
- 距离: d_L(x, y) = arcosh(-c * ⟨x, y⟩_L) / √c

实施: 在 _lib/halc.py 加 HALCLorentzRegularizer 类, 沿用 v2 API (set_epoch, annealed_curvature, reg_loss_for_layers)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HALCLorentzRegularizer(nn.Module):
    """HALC v5: per-layer learnable κ + Lorentz model + curvature annealing."""

    def __init__(
        self,
        num_layers: int = 7,
        init_curvature: float = 1.0,
        c_max: float = 1.0,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        reg_weight_max: float = 0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.c_max = c_max
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        # per-layer learnable log_curvature
        self.log_curvature = nn.Parameter(torch.zeros(num_layers).fill_(math.log(math.expm1(init_curvature))))
        # learnable reg_weight (bounded by tanh)
        self.raw_reg_weight = nn.Parameter(torch.tensor(0.01))
        self.reg_weight_max = reg_weight_max
        self.current_epoch = 0

    def annealed_curvature(self) -> torch.Tensor:
        """c_l(t) = c_l * sigmoid((t - warmup) / cooldown)."""
        t = torch.tensor(float(self.current_epoch), dtype=self.log_curvature.dtype, device=self.log_curvature.device)
        ratio = (t - self.warmup_epochs) / max(self.cooldown_epochs, 1.0)
        sigmoid_factor = torch.sigmoid(ratio)
        learnable_c = F.softplus(self.log_curvature) + 1e-5
        return learnable_c * sigmoid_factor

    @property
    def reg_weight(self) -> torch.Tensor:
        return self.reg_weight_max * torch.tanh(self.raw_reg_weight / self.reg_weight_max)

    def euclidean_to_lorentz(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Map x ∈ R^d (Euclidean) → L^d_c (Lorentz)."""
        # x_lorentz = (sqrt(1 + c*||x||^2), x) / sqrt(c)
        sqrt_c = torch.sqrt(c)
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        x0 = torch.sqrt(1.0 + c * x_norm_sq)
        return torch.cat([x0, x], dim=-1) / sqrt_c

    def logmap0_lorentz(self, x_lorentz: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """logmap at Lorentz origin o = (1/sqrt(c), 0, ..., 0). x_lorentz: (..., d+1)."""
        sqrt_c = torch.sqrt(c)
        x0 = x_lorentz[..., 0:1]  # (..., 1)
        x_spatial = x_lorentz[..., 1:]  # (..., d)
        x_spatial_norm = x_spatial.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        # arcosh(c * x_0) / sqrt(c) — distance from origin
        arg = (c * x0).clamp_min(1 + 1e-7)
        norm = torch.acosh(arg) / sqrt_c  # (..., 1)
        # v_i = norm * x_i / x_spatial_norm
        factor = norm / x_spatial_norm
        return factor * x_spatial  # (..., d)

    def reg_loss_for_layers(self, hidden_states_list) -> torch.Tensor:
        """Compute total reg loss across layers using Lorentz logmap.

        Args:
            hidden_states_list: list of (B, L, d_model) per-layer hidden states (Euclidean)
        Returns:
            total reg loss (scalar tensor)
        """
        n = min(len(hidden_states_list), self.num_layers)
        total = 0.0
        c_per_layer = self.annealed_curvature()  # (num_layers,)
        for i in range(n):
            h = hidden_states_list[i]  # (B, L_seq, d_model)
            c = c_per_layer[i]  # scalar
            # Step 1: Euclidean → Lorentz
            h_lorentz = self.euclidean_to_lorentz(h, c)  # (B, L_seq, d_model+1)
            # Step 2: logmap at Lorentz origin
            v = self.logmap0_lorentz(h_lorentz, c)  # (B, L_seq, d_model)
            # Step 3: reg loss = mean ||v||^2
            total = total + (v ** 2).sum(dim=-1).mean()
        return total / max(n, 1)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch