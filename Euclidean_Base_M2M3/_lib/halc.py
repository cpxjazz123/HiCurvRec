"""HALC v2: Hyperbolic Attention with Learnable Curvatures + annealing schedule.

Euclidean_Base_M2M3 主目录直接迭代 (R52).
不再为每个任务新建独立目录, 直接在 _lib/ 下新增模块.

核心创新 (HALC v2 vs HALC v1):
1. Per-layer learnable κ_l ∈ R^L (L=encoder layers + embed)
2. Curvature annealing: c_l(t) = c_l * sigmoid((t - warmup) / cooldown)
   - 早期 t < warmup: c_l ≈ 0 (近 Euclidean, 不干扰梯度)
   - 后期 t > warmup: c_l → 1 (强 hyperbolic, 鼓励几何结构)
3. tanh-clamped reg_weight ≤ 0.05 (避免破坏主 loss)

数学公式:
- c_l(t) = softplus(log_c_l) * sigmoid((t - warmup) / cooldown)
- poincare_logmap0(x, c) = (arctanh(sqrt(c)*||x||) / (sqrt(c)*||x||)) * x  if ||x||<1/sqrt(c)
- reg_loss = mean over layers of sum(||logmap(x)||^2, dim=-1)

预期效果 (per ICLR 2024 paper):
- 早期训练不被双曲几何干扰
- 后期 reg_loss 自然引导 hidden states 趋近 Poincaré 球面
- 比 HALC v1 (固定 c=1) 训练更稳定, valid NDCG 提升更快
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HALCAnnealingRegularizer(nn.Module):
    """HALC v2: per-layer learnable κ + curvature annealing schedule.
    v16 备胎 (R36 机制变更): differential per-layer-role schedule — encoder (浅层) 早 warmup,
    decoder (深层) 晚 warmup. 通过 encoder_warmup / decoder_warmup / encoder_cooldown /
    decoder_cooldown 区分 schedule. 默认 encoder_warmup=3 / decoder_warmup=7,
    encoder_cooldown=8 / decoder_cooldown=12 (vs HALC v2 统一 warmup=5, cooldown=10).
    """

    def __init__(
        self,
        num_layers: int = 7,        # 6 encoder + 1 embed
        init_curvature: float = 1.0,
        c_max: float = 1.0,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        # v16: differential schedule per layer role
        encoder_warmup: int = 3,    # v16: encoder 浅层 (L1-6) 早 warmup
        encoder_cooldown: int = 8,  # v16: encoder 浅层 cooldown 较短 (更快达到 c_max)
        decoder_warmup: int = 7,    # v16: decoder 深层 (L7) 晚 warmup
        decoder_cooldown: int = 12, # v16: decoder 深层 cooldown 较长 (更慢达到 c_max)
        reg_weight_max: float = 0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.c_max = c_max
        self.warmup_epochs = warmup_epochs  # 保留兼容 (但实际用 differential schedule)
        self.cooldown_epochs = cooldown_epochs
        # v16: per-layer warmup/cooldown 张量 (num_layers,)
        # 前 num_layers-1 层是 encoder, 最后一层是 decoder
        warmup_per_layer = torch.tensor(
            [encoder_warmup] * (num_layers - 1) + [decoder_warmup],
            dtype=torch.float32,
        )
        cooldown_per_layer = torch.tensor(
            [encoder_cooldown] * (num_layers - 1) + [decoder_cooldown],
            dtype=torch.float32,
        )
        # 注册为 buffer (不参与梯度但跟随 device)
        self.register_buffer("warmup_per_layer", warmup_per_layer)
        self.register_buffer("cooldown_per_layer", cooldown_per_layer)
        # per-layer learnable log_curvature
        self.log_curvature = nn.Parameter(torch.zeros(num_layers).fill_(math.log(math.expm1(init_curvature))))
        # learnable reg_weight (bounded by tanh)
        self.raw_reg_weight = nn.Parameter(torch.tensor(0.01))
        self.reg_weight_max = reg_weight_max
        self.current_epoch = 0

    def annealed_curvature(self) -> torch.Tensor:
        """v16: c_l(t) = c_l * sigmoid((t - warmup_l) / cooldown_l).
        每个 layer 有独立 warmup/cooldown (encoder 早, decoder 晚).
        """
        t = torch.tensor(float(self.current_epoch), dtype=self.log_curvature.dtype, device=self.log_curvature.device)
        # v16: per-layer ratio (num_layers,)
        ratio = (t - self.warmup_per_layer) / self.cooldown_per_layer.clamp_min(1.0)
        sigmoid_factor = torch.sigmoid(ratio)  # (num_layers,)
        learnable_c = F.softplus(self.log_curvature) + 1e-5
        return learnable_c * sigmoid_factor  # (num_layers,)

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
        """Compute total reg loss across layers.

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
        """Update current_epoch for annealing schedule."""
        self.current_epoch = epoch