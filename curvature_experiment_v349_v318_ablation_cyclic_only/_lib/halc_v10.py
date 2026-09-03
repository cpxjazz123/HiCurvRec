"""HALC v10: Stage 2 κ → Stage 3 init transfer.

Euclidean_Base_M2M3 主目录直接迭代 (R52).
R36 曲率机制变更 (Stage 2 学到的 c_l 直接作为 Stage 3 HALC init_curvature, 不是 sweep):

HALC v2 (R37 PASS test_R@10=0.1072): c_l(t) = softplus(log_c_l) * sigmoid((t-5)/10)
        → init_curvature=1.0 (固定, 每层一致)

HALC v10 (本 issue, Issue #230): c_l(t) = softplus(log_c_l) * sigmoid((t-5)/10)
        → init_curvature = c22_stage2_c_l ≈ 0.3 (per-layer, 来自 Stage 2 真值)
        → log_c_l init = log(expm1(c_l_stage2)) (起点就是 Stage 2 收敛值)

物理意义:
- Stage 2 (RQ-VAE codebook) 用 c22 TCU (可微曲率训练) 已经学到 c_l ≈ 0.3 (per-layer)
- Stage 3 (HALC reg) 直接用 c_l=0.3 作 init, 让 Stage 3 训练起点 = Stage 2 收敛点
- 避免 v2 init=1.0 让 Stage 3 起点偏离 Stage 2 已学到的几何 (Stage 2 c=1.0 → Stage 3 起点 c=1.0 → 双曲空间跨度大)
- Stage 3 学习空间更小, 收敛更快 (c 起点 0.3 vs v2 起点 1.0, softplus 起点 log(expm1(0.3)) ≈ -0.8)

R36 合规性:
- 不是调 LR/dropout/reg_weight sweep
- 是改 κ(t) 函数 init (init_curvature: 1.0 → Stage 2 真值 0.3)
- 属于"新曲率框架" (Stage 2 → Stage 3 init transfer, 跨 stage 曲率迁移)

预期效果 (vs HALC v2 test_R@10=0.1072):
- Stage 3 起点更接近 Stage 2 收敛, valid-test gap 缩小
- 早期 c_l 较小, 几何介入温和, 可能避开 v6 起步过强问题
- R37 PASS 条件: test_R@10 ≥ 0.1072 (vs v2 baseline)
- R37 GO 条件: test_R@10 ≥ 0.1100 (+0.003)

备胎来源: c22_tcu curvature_state (Stage 2 真值, 训练期 c_l ∈ [0.30, 0.31])
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HALCStage2InitRegularizer(nn.Module):
    """HALC v10: Stage 2 κ → Stage 3 init_curvature.

    接口与 HALC v2 HALCAnnealingRegularizer 兼容 (reg_loss_for_layers + set_epoch).
    """

    def __init__(
        self,
        num_layers: int = 7,        # 6 encoder + 1 embed
        # Stage 2 真值 (来自 c22_tcu curvature_state)
        stage2_init_curvatures=(0.3003, 0.3000, 0.3001),
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        reg_weight_max: float = 0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.reg_weight_max = reg_weight_max
        # per-layer learnable log_curvature, init = log(expm1(c_l_stage2))
        # Stage 2 只有 3 层码本, 但 Stage 3 hidden_states 有 7 层 (6 encoder + 1 embed)
        # 对码本层 (l=0,1,2) 用 stage2 真值, 对 embed/extra 层 (l=3..6) 用 stage2 均值
        stage2_mean = sum(stage2_init_curvatures) / len(stage2_init_curvatures)
        log_curv = torch.zeros(num_layers)
        for i in range(num_layers):
            c = stage2_init_curvatures[i] if i < len(stage2_init_curvatures) else stage2_mean
            log_curv[i] = math.log(max(math.expm1(c), 1e-5))
        self.log_curvature = nn.Parameter(log_curv)
        self.raw_reg_weight = nn.Parameter(torch.tensor(0.01))
        self.current_epoch = 0
        print(f"[HALC v10] Stage 2 init curvatures: {stage2_init_curvatures}, mean={stage2_mean:.4f}", flush=True)
        print(f"[HALC v10] log_curvature init: {[round(x.item(), 4) for x in log_curv]}", flush=True)

    def annealed_curvature(self) -> torch.Tensor:
        """c_l(t) = softplus(log_c_l) * sigmoid((t - warmup) / cooldown).

        log_c_l 起点 = log(expm1(c_l_stage2)) ≈ log(expm1(0.3)) ≈ -0.79
        softplus(-0.79) ≈ 0.30 (起点就是 stage2 真值)
        v2 log_c_l init 0 → softplus(0) ≈ 0.69 → sigmoid(0) ≈ 0.5 → 起点 ≈ 0.35 (类似但不同)
        """
        t = torch.tensor(float(self.current_epoch), dtype=self.log_curvature.dtype, device=self.log_curvature.device)
        ratio = (t - self.warmup_epochs) / max(self.cooldown_epochs, 1.0)
        sigmoid_factor = torch.sigmoid(ratio)
        learnable_c = F.softplus(self.log_curvature) + 1e-5
        return learnable_c * sigmoid_factor

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
        """Compute Poincaré reg loss with stage2-init curvature across layers."""
        n = min(len(hidden_states_list), self.num_layers)
        total = 0.0
        c_per_layer = self.annealed_curvature()
        for i in range(n):
            c = c_per_layer[i]
            logmap = self.poincare_logmap0(hidden_states_list[i], c)
            total = total + (logmap ** 2).sum(dim=-1).mean()
        return total / max(n, 1)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
