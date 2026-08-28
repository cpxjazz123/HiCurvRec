"""HALC v8: Curriculum Curvature 3-stage step annealing.

Euclidean_Base_M2M3 主目录直接迭代 (R52).
R36 曲率机制变更 (HALC v2 sigmoid annealing → step annealing, 不是调参):

v2 (sigmoid): c_l(t) = softplus(log_c_l) * sigmoid((t - warmup) / cooldown)
              → 单调增长, 平滑过渡
v8 (step):   c_l(t) = piecewise schedule:
              - early  (E0-E_early):  c_l = softplus(log_c_l) * c_early  (近 Euclidean, 稳定起步)
              - mid    (E_early-E_mid): c_l = softplus(log_c_l) * c_mid (中等曲率, 几何介入)
              - late   (E_mid-E_max):   c_l = softplus(log_c_l) * c_late (强双曲, 充分双曲化)

物理意义:
- 早期 (E0-E_early): 模型学 Euclidean representation, 稳定起步
- 中期 (E_early-E_mid): 引入中等曲率, 让模型逐渐适应双曲空间
- 后期 (E_mid-E_max): 强双曲化, 充分利用双曲几何优势

R36 合规性: 不是调 LR/dropout/reg_weight sweep, 是改 κ(t) 函数形式 (sigmoid → piecewise step),
属于"新曲率正则项" (曲线函数本身被改变, 而非标量超参).

预期效果 (vs HALC v2 test_R@10=0.1072):
- 阶梯式曲率可能让 hidden state 在不同时期学到不同几何层次
- 早期 Euclidean 阶段避免 v6 valid-test 偏差扩 (v6 dual-manifold 起步即混合, 几何约束过强)
- R37 PASS 条件: test_R@10 ≥ 0.1072 (vs v2 baseline)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HALCCurriculumRegularizer(nn.Module):
    """HALC v8: Curriculum Curvature 3-stage step annealing.

    接口与 HALC v2 HALCAnnealingRegularizer 兼容 (reg_loss_for_layers + set_epoch),
    可直接替换 train_decoder.py 中的实例化.
    """

    def __init__(
        self,
        num_layers: int = 7,        # 6 encoder + 1 embed
        init_curvature: float = 1.0,
        warmup_epochs: int = 5,
        reg_weight_max: float = 0.05,
        # Curriculum stages
        early_boundary: int = 30,    # E0-E_early: 近 Euclidean
        mid_boundary: int = 100,     # E_early-E_mid: 中等曲率
        # 3 个阶段曲率缩放
        c_early_scale: float = 0.1,  # 早期 c_l *= 0.1 (近 Euclidean)
        c_mid_scale: float = 0.5,    # 中期 c_l *= 0.5
        c_late_scale: float = 1.0,   # 后期 c_l *= 1.0 (强双曲)
    ):
        super().__init__()
        self.num_layers = num_layers
        self.warmup_epochs = warmup_epochs
        self.early_boundary = early_boundary
        self.mid_boundary = mid_boundary
        self.c_early_scale = c_early_scale
        self.c_mid_scale = c_mid_scale
        self.c_late_scale = c_late_scale
        # per-layer learnable log_curvature (基础值, 实际 c_l = log_c_l * stage_scale)
        self.log_curvature = nn.Parameter(torch.zeros(num_layers).fill_(math.log(math.expm1(init_curvature))))
        # learnable reg_weight (bounded by tanh)
        self.raw_reg_weight = nn.Parameter(torch.tensor(0.01))
        self.reg_weight_max = reg_weight_max
        self.current_epoch = 0

    def curriculum_scale(self) -> float:
        """3-stage piecewise schedule: c_l(t) = log_c_l * scale(t)."""
        t = self.current_epoch
        if t < self.early_boundary:
            return self.c_early_scale
        elif t < self.mid_boundary:
            return self.c_mid_scale
        else:
            return self.c_late_scale

    def annealed_curvature(self) -> torch.Tensor:
        """c_l(t) = softplus(log_c_l) * scale(t) (3-stage step)."""
        learnable_c = F.softplus(self.log_curvature) + 1e-5
        scale = self.curriculum_scale()
        return learnable_c * scale  # (num_layers,)

    @property
    def reg_weight(self) -> torch.Tensor:
        return self.reg_weight_max * torch.tanh(self.raw_reg_weight / self.reg_weight_max)

    def poincare_logmap0(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """logmap0: Poincaré ball B_c^d → tangent space at origin.

        标准公式: logmap0(x) = (arctanh(sqrt(c)*||x||) / (sqrt(c)*||x||)) * x
        当 ||x|| → 0 时 factor → 1, 平滑过渡到 Euclidean.
        """
        sqrt_c = torch.sqrt(c)
        norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        max_norm = (1.0 / sqrt_c - 1e-5)
        norm_x_clamp = norm_x.clamp_max(max_norm)
        factor = torch.arctanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)
        return factor * x

    def reg_loss_for_layers(self, hidden_states_list) -> torch.Tensor:
        """Compute Poincaré reg loss with curriculum curvature across layers.

        Args:
            hidden_states_list: list of (B, L, d_model) per layer hidden states
                                (encoder_hidden_states from T5)
        Returns:
            total reg loss (scalar tensor)
        """
        n = min(len(hidden_states_list), self.num_layers)
        total = 0.0
        c_per_layer = self.annealed_curvature()  # (num_layers,)
        for i in range(n):
            c = c_per_layer[i]
            logmap = self.poincare_logmap0(hidden_states_list[i], c)
            total = total + (logmap ** 2).sum(dim=-1).mean()
        return total / max(n, 1)

    def set_epoch(self, epoch: int):
        """Update current_epoch for curriculum schedule."""
        self.current_epoch = epoch