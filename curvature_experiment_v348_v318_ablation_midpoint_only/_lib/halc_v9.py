"""HALC v9: Hybrid Smooth-Step Curvature (v2 sigmoid + v8 step_scale).

Euclidean_Base_M2M3 主目录直接迭代 (R52).
R36 曲率机制组合创新 (v8 R37 FAIL 后继):

HALC v2: c_l(t) = softplus(log_c_l) * sigmoid((t - 5) / 10)
        → 单调增长, 平滑起步, R37 PASS test_R@10=0.1072

HALC v8: c_l(t) = softplus(log_c_l) * step_scale(t)
        → step_scale ∈ {0.1 (E0-30), 0.5 (E30-100), 1.0 (E100+)}
        → R37 FAIL test_R@10=0.1047, 早期 c=0.1 起步过弱, 浪费 30 epoch

HALC v9: c_l(t) = softplus(log_c_l) * sigmoid((t - 5) / 10) * step_scale(t)
        → v2 sigmoid 平滑起步 (解决 v8 起步过弱)
        → + v8 step_scale 后期几何介入 (E50+ 阶梯)
        → step_scale ∈ {1.0 (E0-50, 与 v2 一致), 0.7 (E50-100, 轻度阶梯), 1.0 (E100+, 强双曲)}

物理意义:
- 早期 (E0-50): 完全沿用 v2 sigmoid 平滑, c(t) ≈ 0.38 (E0) → 0.85 (E50), 起步稳
- 中期 (E50-100): step_scale=0.7 让 c_l 短暂下降 (从 0.85→0.6), 让 hidden state 重新适应弱几何
- 后期 (E100+): step_scale=1.0 恢复强双曲, c_l → 1.0

R36 合规性: 不是调参 sweep, 是改 κ(t) 函数形式 (v2 sigmoid → v9 sigmoid * step_scale).
属于"新曲率正则项" (曲线函数被改变, 而非标量超参).

预期效果 (vs HALC v2 test_R@10=0.1072):
- 中期 step_scale=0.7 可能让 valid 出现短暂平台期 (从 0.0976 → 0.068), 但给模型喘息机会
- 后期 step_scale=1.0 + c_l→1.0 可能加速收敛到 v2 baseline 或略高
- R37 PASS 条件: test_R@10 ≥ 0.1072 (vs v2 baseline)
- R37 GO 条件: test_R@10 ≥ 0.1100 (+0.003)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HALCHybridSmoothStepRegularizer(nn.Module):
    """HALC v9: Hybrid v2 sigmoid + v8 step_scale curvature schedule.

    接口与 HALC v2 HALCAnnealingRegularizer 兼容 (reg_loss_for_layers + set_epoch),
    可直接替换 train_decoder.py 中的实例化.
    """

    def __init__(
        self,
        num_layers: int = 7,        # 6 encoder + 1 embed
        init_curvature: float = 1.0,
        c_max: float = 1.0,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        reg_weight_max: float = 0.05,
        # Step scale schedule (multiplier on top of v2 sigmoid)
        step_early_boundary: int = 50,   # E0-E_early: step_scale=1.0 (与 v2 一致)
        step_mid_boundary: int = 100,    # E_early-E_mid: step_scale=mid_scale
        step_early_scale: float = 1.0,   # 早期 step_scale=1.0 (不削弱)
        step_mid_scale: float = 0.7,     # 中期 step_scale=0.7 (轻度阶梯)
        step_late_scale: float = 1.0,    # 后期 step_scale=1.0 (恢复强双曲)
    ):
        super().__init__()
        self.num_layers = num_layers
        self.c_max = c_max
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.step_early_boundary = step_early_boundary
        self.step_mid_boundary = step_mid_boundary
        self.step_early_scale = step_early_scale
        self.step_mid_scale = step_mid_scale
        self.step_late_scale = step_late_scale
        # per-layer learnable log_curvature
        self.log_curvature = nn.Parameter(torch.zeros(num_layers).fill_(math.log(math.expm1(init_curvature))))
        # learnable reg_weight (bounded by tanh)
        self.raw_reg_weight = nn.Parameter(torch.tensor(0.01))
        self.reg_weight_max = reg_weight_max
        self.current_epoch = 0

    def step_scale(self) -> float:
        """3-stage step schedule (multiplier on top of v2 sigmoid)."""
        t = self.current_epoch
        if t < self.step_early_boundary:
            return self.step_early_scale
        elif t < self.step_mid_boundary:
            return self.step_mid_scale
        else:
            return self.step_late_scale

    def annealed_curvature(self) -> torch.Tensor:
        """c_l(t) = softplus(log_c_l) * sigmoid((t - warmup) / cooldown) * step_scale(t).

        综合 v2 sigmoid 平滑 + v8 step_scale 后期阶梯.
        """
        t = torch.tensor(float(self.current_epoch), dtype=self.log_curvature.dtype, device=self.log_curvature.device)
        ratio = (t - self.warmup_epochs) / max(self.cooldown_epochs, 1.0)
        sigmoid_factor = torch.sigmoid(ratio)
        learnable_c = F.softplus(self.log_curvature) + 1e-5
        step = self.step_scale()
        return learnable_c * sigmoid_factor * step  # (num_layers,)

    @property
    def reg_weight(self) -> torch.Tensor:
        return self.reg_weight_max * torch.tanh(self.raw_reg_weight / self.reg_weight_max)

    def poincare_logmap0(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """logmap0: Poincaré ball B_c^d → tangent space at origin."""
        sqrt_c = torch.sqrt(c)
        norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        max_norm = (1.0 / sqrt_c - 1e-5)
        norm_x_clamp = norm_x.clamp_max(max_norm)
        factor = torch.arctanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)
        return factor * x

    def reg_loss_for_layers(self, hidden_states_list) -> torch.Tensor:
        """Compute Poincaré reg loss with hybrid smooth-step curvature across layers."""
        n = min(len(hidden_states_list), self.num_layers)
        total = 0.0
        c_per_layer = self.annealed_curvature()  # (num_layers,)
        for i in range(n):
            c = c_per_layer[i]
            logmap = self.poincare_logmap0(hidden_states_list[i], c)
            total = total + (logmap ** 2).sum(dim=-1).mean()
        return total / max(n, 1)

    def set_epoch(self, epoch: int):
        """Update current_epoch for hybrid smooth-step schedule."""
        self.current_epoch = epoch