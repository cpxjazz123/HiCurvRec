"""HALC v12: Per-Head κ (vs v2 per-layer κ).

Euclidean_Base_M2M3 主目录直接迭代 (R52).
R36 曲率机制变更 (per-layer → per-head, 几何粒度从 layer 变为 head):

v2 (R37 PASS test_R@10=0.1072): c_l(t) per-layer (num_layers=7)
        → 所有 attention head 共享同一个 c_l
        → 一个 layer 一个曲率, 几何粗粒度

v12 (本 issue, Issue #232): c_h(t) per-head (num_heads * num_layers)
        → 每个 attention head 有独立曲率 c_h
        → 几何细粒度 (允许 head 学不同曲率, 适应不同语义 head)

实现思路:
- hidden_states (B, L, d_model), reshape 为 (B, L, H, d_head) per-head
- 每个 head 用独立 c_h 计算 Poincaré logmap0 reg
- reg = mean over (B, L, H) per-head norm²

参数数变化:
- v2: num_layers=7 → log_curvature (7,)
- v12: num_heads=6 * num_layers=7 = 42 → log_curvature (42,)

物理意义:
- 不同 attention head 学不同语义: 一些 head 关注历史, 一些关注位置
- 让 head 自己选曲率: 位置 head 用 c 大 (双曲, 距离敏感), 历史 head 用 c 小 (近 Euclidean)
- 几何粒度细化, R36 新曲率机制

R36 合规性:
- 不是调参 sweep (per-head vs per-layer 是机制变更)
- 不是改 κ(t) 函数 (sigmoid 形式不变, 只改参数空间维度)

预期效果 (vs HALC v2 test_R@10=0.1072):
- per-head 几何细粒度, 允许 head 自适应
- R37 PASS 条件: test_R@10 ≥ 0.1072
- R37 GO 条件: test_R@10 ≥ 0.1100
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HALCPerHeadRegularizer(nn.Module):
    """HALC v12: Per-head κ attention on hidden_states.

    接口与 HALC v2 HALCAnnealingRegularizer 兼容 (reg_loss_for_layers + set_epoch).
    每个 hidden_state 层 reshape 为 (B, L, H, d_head), per-head 独立 c_h 计算 Poincaré reg.
    """

    def __init__(
        self,
        num_layers: int = 7,
        num_heads: int = 6,
        d_model: int = 128,
        init_curvature: float = 1.0,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        reg_weight_max: float = 0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, f"d_model {d_model} 必须被 num_heads {num_heads} 整除"
        self.d_head = d_model // num_heads
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.reg_weight_max = reg_weight_max
        # per-head per-layer learnable log_curvature (num_layers * num_heads)
        self.log_curvature = nn.Parameter(
            torch.zeros(num_layers * num_heads).fill_(math.log(math.expm1(init_curvature)))
        )
        self.raw_reg_weight = nn.Parameter(torch.tensor(0.01))
        self.current_epoch = 0
        print(f"[HALC v12] Per-Head κ: {num_layers} layers × {num_heads} heads = {num_layers * num_heads} curvatures, d_head={self.d_head}", flush=True)

    def annealed_curvature(self) -> torch.Tensor:
        """c_h_l(t) = softplus(log_c_h_l) * sigmoid((t - warmup) / cooldown).

        Returns (num_layers * num_heads,) tensor.
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
        """Per-head Poincaré reg across all layers.

        Args:
            hidden_states_list: list of (B, L, d_model) per-layer hidden states
        Returns:
            total reg loss (scalar tensor)
        """
        n = min(len(hidden_states_list), self.num_layers)
        total = 0.0
        c_all = self.annealed_curvature()  # (num_layers * num_heads,)
        for i in range(n):
            h_l = hidden_states_list[i]  # (B, L, d_model)
            B, L, d = h_l.shape
            # reshape to (B, L, H, d_head)
            h_per_head = h_l.view(B, L, self.num_heads, self.d_head)
            # c_h_l for this layer (num_heads,)
            c_h_l = c_all[i * self.num_heads : (i + 1) * self.num_heads]  # (num_heads,)
            for h in range(self.num_heads):
                c_h = c_h_l[h]
                logmap = self.poincare_logmap0(h_per_head[:, :, h, :], c_h)  # (B, L, d_head)
                total = total + (logmap ** 2).sum(dim=-1).mean()
        return total / max(n * self.num_heads, 1)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
