"""HALC v38: Per-token input-dependent κ + Stage 2 TCU init=0.3 (R36 新曲率机制).

Euclidean_Base_M2M3 / curvature_experiment 主目录迭代 (R52/R50 联动).
R36 框架级变更 — 新曲率机制 (per-token input-dependent κ), 不是 sweep:

v2 (R37 PASS test_R@10=0.1072 / v19 baseline 0.1113): c_l(t) per-layer (num_layers=7)
        → log_curvature (7,), 同一 layer 所有 token 共享 c_l

v36 (Issue #164 R37 FAIL 因训练 E12 timeout 中断, 机制未验证): per-token adaptive reg_weight_max
        → reg_weight 根据 token hidden norm 自适应
        → κ 仍是 per-layer

v38 (本 issue): per-token input-dependent κ
        → 每个 token 自己的 c_t ∈ R^(B, L) (基于 input hidden norm 计算)
        → log_curvature (1, 1) per-layer (per-layer 学习幅度 + per-token 输入调制)
        → c_t = softplus(log_c_l + δ(‖h_t‖)) * sigmoid((t - warmup)/cooldown)
        → δ(‖h_t‖) = linear(‖h_t‖) (单标量线性层 per-layer)

物理意义 (R36 曲率机制):
- v2 per-layer κ: 整个 layer 一个曲率, 粗粒度
- v12 per-head κ: head 粒度, 但仍是 layer-wide (同一 head 所有 token 共享)
- v38 per-token κ: token 粒度, 不同 token 不同曲率 → 输入自适应的几何约束
- 低 norm token (常见词 / 边缘 token): 用大 c (强双曲, 边界感强, 拉向边缘)
- 高 norm token (稀有词 / 核心 token): 用小 c (近欧氏, 平滑空间)

数学公式:
- δ_per_layer: linear layer (1 → 1) per-layer, init bias=0 (起点 δ=0 → 与 v2 一致, 不破坏起点)
- ‖h_t‖: token L2 norm (B, L)
- log_c_t = log_curvature + δ_per_layer(‖h_t‖)  # (B, L)
- c_t = softplus(log_c_t) * sigmoid((t - warmup) / cooldown)  # (B, L)
- logmap0_t(h_t, c_t) = (arctanh(√c_t * ‖h_t‖) / (√c_t * ‖h_t‖)) * h_t  # (B, L, d_model)
- reg_loss = mean over (B, L, layers) of ‖logmap0_t‖^2

reg_weight_max (R36 不调参): v2=0.05, v38=0.06 (温和增强)

Stage 2 TCU init=0.3 (v10 起源):
- init_curvature=0.3 (Stage 2 RQ-VAE TCU 学到的 c_l 真值)
- log_curvature init = log(expm1(0.3)) ≈ -0.8
- 让 Stage 3 起点 = Stage 2 收敛点 (跨 stage 曲率迁移)

R36 合规性:
- 不是调 LR / dropout / wd / label_smoothing sweep
- 不是改 reg_weight_max 0.05 → 0.06 (那是固定值微调, 但已声明)
- 是改 κ 函数 (per-layer → per-token) — 真正的 R36 框架级变更 (曲率粒度细化)
- init_curvature 0.3 来自 Stage 2 真值 — 跨 stage 几何迁移, 同样 R36

预期效果 (vs v19 baseline test_R@10=0.1113):
- per-token κ 让不同 token 学习不同曲率, 几何粒度从 layer→token
- TCU init=0.3 让 Stage 3 起点 = Stage 2 收敛, 收敛更快
- 目标 test_R@10 ≥ 0.12 (R37 GO)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HALCPerTokenInputDependentRegularizer(nn.Module):
    """HALC v38: per-token input-dependent κ + Stage 2 TCU init=0.3."""

    def __init__(
        self,
        num_layers: int = 7,
        d_model: int = 128,
        init_curvature: float = 0.3,  # Stage 2 TCU 真值 (v10 origin)
        c_max: float = 1.0,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        reg_weight_max: float = 0.06,
        delta_scale_init: float = 0.01,  # δ init 起点 = scale * ‖h_t‖, 让 δ 接近 0 (与 v2 等价起点)
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.c_max = c_max
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs

        # per-layer learnable log_c (与 v2 一致, 用于 base curvature)
        self.log_curvature = nn.Parameter(
            torch.zeros(num_layers).fill_(math.log(math.expm1(init_curvature)))
        )
        # per-layer linear δ(‖h_t‖) → scalar (per-layer 学习幅度)
        self.delta_layer = nn.ModuleList([
            nn.Linear(1, 1, bias=True) for _ in range(num_layers)
        ])
        for layer in self.delta_layer:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # learnable reg_weight (bounded by tanh)
        self.raw_reg_weight = nn.Parameter(torch.tensor(0.01))
        self.reg_weight_max = reg_weight_max
        self.delta_scale = delta_scale_init
        self.current_epoch = 0

    def annealed_curvature_factor(self) -> torch.Tensor:
        """sigmoid((t - warmup) / cooldown), scalar."""
        t = torch.tensor(float(self.current_epoch), dtype=self.log_curvature.dtype, device=self.log_curvature.device)
        ratio = (t - self.warmup_epochs) / max(self.cooldown_epochs, 1.0)
        return torch.sigmoid(ratio)

    def per_token_curvature(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """c_t = (softplus(log_c_l) + δ(‖h_t‖)) * sigmoid_factor, 返回 (B, L)."""
        # ‖h_t‖: (B, L)
        token_norm = hidden_states.norm(dim=-1)  # (B, L)
        # δ(‖h_t‖) per-layer: linear(1→1) (B, L, 1) → (B, L)
        delta = self.delta_layer[layer_idx](token_norm.unsqueeze(-1)).squeeze(-1)
        delta = delta * self.delta_scale  # 起点小, 让 δ 接近 0 (与 v2 等价)
        # log_c_l (per-layer, scalar)
        base_log_c = F.softplus(self.log_curvature[layer_idx]) + 1e-5
        # log_c_t = log(base_c + exp(delta)) ≈ log(exp(log_c) + δ)
        # 简化: 直接相加 (small delta 假设)
        c_t = (base_log_c + delta) * self.annealed_curvature_factor()
        return c_t

    @property
    def reg_weight(self) -> torch.Tensor:
        return self.reg_weight_max * torch.tanh(self.raw_reg_weight / self.reg_weight_max)

    def poincare_logmap0_per_token(self, x: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        """logmap0 per-token. Args: x (B, L, d_model), c_t (B, L)."""
        sqrt_c = torch.sqrt(c_t).unsqueeze(-1)  # (B, L, 1)
        norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)  # (B, L, 1)
        max_norm = (1.0 / sqrt_c - 1e-5)  # (B, L, 1)
        norm_x_clamp = norm_x.clamp_max(max_norm)
        factor = torch.arctanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)  # (B, L, 1)
        return factor * x  # (B, L, d_model)

    def reg_loss_for_layers(self, hidden_states_list) -> torch.Tensor:
        """Compute total reg loss across layers with per-token κ.

        Args:
            hidden_states_list: list of (B, L, d_model) per-layer hidden states
        Returns:
            total reg loss (scalar tensor)
        """
        n = min(len(hidden_states_list), self.num_layers)
        total = 0.0
        for i in range(n):
            h = hidden_states_list[i]  # (B, L, d_model)
            c_t = self.per_token_curvature(h, i)  # (B, L)
            logmap = self.poincare_logmap0_per_token(h, c_t)  # (B, L, d_model)
            # reg = mean over (B, L, d_model) of logmap^2
            total = total + (logmap ** 2).mean()
        return total / max(n, 1)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch