"""HALC v3: Per-channel-group learnable curvature + 强 reg (R52 主目录迭代备胎).

在 HALC v2 (per-layer κ_l) 之上升级:
1. Per-channel-group κ ∈ R^(num_layers × G)
   - d_model=128, num_groups=4 → group_size=32 (整除, 避免维度不整除问题)
   - 不同 channel groups 学习不同曲率, 增强细粒度几何控制
2. 强 reg: reg_weight_max 0.05 → 0.08 (温和增强, 避免破坏主 loss)
3. 保留 v2 curvature annealing (sigmoid((t-warmup)/cooldown))

文献支撑:
- Hypformer (NeurIPS 2024) full-stack hyperbolic transformer
- HRec (ICLR 2025) tangent-space aggregation
- Curved Representation Space (2024) per-channel trainable curvature

预期: 较 HALC v2 (test_R@10=0.1072) 进一步 +0.005-0.015 → test_R@10 0.115-0.12

数学公式:
- κ[l, g] = softplus(log_κ[l, g]) * sigmoid((t - warmup) / cooldown)
- logmap_g(x_g, c) = (arctanh(√c*||x_g||) / (√c*||x_g||)) * x_g
- reg_loss = mean over layers of mean over groups of ||logmap_g(x_g)||^2
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HALCPerChannelGroupRegularizer(nn.Module):
    """HALC v3: per-channel-group learnable κ + annealing + 强 reg."""

    def __init__(
        self,
        num_layers: int = 7,
        d_model: int = 128,
        num_groups: int = 4,
        init_curvature: float = 1.0,
        c_max: float = 1.0,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        reg_weight_max: float = 0.08,
    ):
        super().__init__()
        assert d_model % num_groups == 0, f"d_model={d_model} must be divisible by num_groups={num_groups}"
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_groups = num_groups
        self.group_size = d_model // num_groups
        self.c_max = c_max
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs

        # per-channel-group learnable κ ∈ R^(num_layers × num_groups)
        self.log_curvature = nn.Parameter(
            torch.zeros(num_layers, num_groups).fill_(math.log(math.expm1(init_curvature)))
        )
        # learnable reg_weight (bounded by tanh)
        self.raw_reg_weight = nn.Parameter(torch.tensor(0.01))
        self.reg_weight_max = reg_weight_max
        self.current_epoch = 0

    def annealed_curvature(self) -> torch.Tensor:
        """κ[l, g](t) = κ[l, g] * sigmoid((t - warmup) / cooldown)."""
        t = torch.tensor(float(self.current_epoch), dtype=self.log_curvature.dtype, device=self.log_curvature.device)
        ratio = (t - self.warmup_epochs) / max(self.cooldown_epochs, 1.0)
        sigmoid_factor = torch.sigmoid(ratio)
        learnable_c = F.softplus(self.log_curvature) + 1e-5  # (L, G)
        return learnable_c * sigmoid_factor

    @property
    def reg_weight(self) -> torch.Tensor:
        return self.reg_weight_max * torch.tanh(self.raw_reg_weight / self.reg_weight_max)

    def poincare_logmap0(self, x_g: torch.Tensor, c_g: torch.Tensor) -> torch.Tensor:
        """logmap0 per-group. Args: x_g (..., group_size), c_g (...)."""
        sqrt_c = torch.sqrt(c_g).unsqueeze(-1)  # broadcast to x_g
        norm_x = x_g.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        max_norm = (1.0 / sqrt_c - 1e-5)
        norm_x_clamp = norm_x.clamp_max(max_norm)
        factor = torch.arctanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)
        return factor * x_g

    def reg_loss_for_layers(self, hidden_states_list) -> torch.Tensor:
        """Compute total reg loss across layers and channel groups.

        Args:
            hidden_states_list: list of (B, L, d_model) per-layer hidden states
        Returns:
            total reg loss (scalar tensor)
        """
        n = min(len(hidden_states_list), self.num_layers)
        total = 0.0
        c_per_layer_group = self.annealed_curvature()  # (L, G)
        for i in range(n):
            h = hidden_states_list[i]  # (B, L_seq, d_model)
            # reshape to (B, L_seq, G, group_size)
            h_grouped = h.view(*h.shape[:-1], self.num_groups, self.group_size)
            layer_total = 0.0
            for g in range(self.num_groups):
                x_g = h_grouped[..., g, :]  # (B, L_seq, group_size)
                c_g = c_per_layer_group[i, g]  # scalar
                logmap = self.poincare_logmap0(x_g, c_g)
                layer_total = layer_total + (logmap ** 2).sum(dim=-1).mean()
            layer_total = layer_total / self.num_groups
            total = total + layer_total
        return total / max(n, 1)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch