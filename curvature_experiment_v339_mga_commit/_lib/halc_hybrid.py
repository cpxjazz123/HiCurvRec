"""HALC v6: Hybrid Poincaré + Lorentz dual-manifold regularization.

Euclidean_Base_M2M3 主目录直接迭代 (R52).
R36 曲率机制组合创新 (而非单纯调参): 在 HALC v2 单一 Poincaré reg 基础上,
同时引入 Lorentz (双曲面) 流形 reg, 由可学习混合权重 α 在两种几何间切换.

核心创新 (HALC v6 vs HALC v2):
1. 沿用 HALC v2 全部机制: per-layer learnable κ_l + curvature annealing + tanh-clamped reg_weight
2. + Lorentz 流形 logmap0 reg: 同一 hidden state 同时在 Poincaré 球 B_c^d 与 Lorentz 双曲面 L^d_c
   上做几何约束 (manuscript ensemble)
3. + 可学习混合权重 log_alpha (sigmoid → α ∈ (0,1)):
   total_reg = mean_l [ α * poincare_reg_l + (1-α) * lorentz_reg_l ]

数学公式:
- c_l(t) = softplus(log_c_l) * sigmoid((t - warmup) / cooldown)  # 共享 κ 给两流形
- poincare_logmap0(x, c) = (arctanh(sqrt(c)*||x||) / (sqrt(c)*||x||)) * x
- lorentz_logmap0: 假设 hidden 视为 Lorentz (d+1)-vec [x_0=||h||, h_rest], 则
  alpha_L = arccosh(sqrt(c*||x_rest||^2 + 1)) / sqrt(c)
  ||logmap0_L||^2 = alpha_L^2  (第一维 0, 余维是 unit * alpha_L)
- total_reg = mean over layers of [α * ||logmap0_P||^2 + (1-α) * ||logmap0_L||^2]
- final loss = task_loss + reg_weight * total_reg

物理意义 (R36 曲率机制):
- Poincaré: 边界性质强, 适合层级/树结构 (Le et al. 2019, Nickel & Kiela 2017)
- Lorentz: Minkowski 度规, 数值稳定, 长序列友好 (Law & Stam 2020, Chami et al. 2019)
- 两者组合 → manifold ensemble, 不同几何 family 互补, 比单一 geometry 更鲁棒
- α 自适应学习: 数据驱动决定哪个几何更优 (避免人为选错的 risk)

预期效果 (vs HALC v2 test_R@10=0.1072):
- 若双流形组合显著: test_R@10 → 0.11+
- 若组合无效 (α 趋近 0 或 1): 等同 HALC v2, test_R@10 ≈ 0.1072
- R37 FAIL 边界: test_R@10 < 0.1072 → R50 回滚
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HALCHybridRegularizer(nn.Module):
    """HALC v6: Hybrid Poincaré + Lorentz dual-manifold reg + learnable mixing α.

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
        init_log_alpha: float = 0.0,  # sigmoid(0) = 0.5, 平衡起步
    ):
        super().__init__()
        self.num_layers = num_layers
        self.c_max = c_max
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        # per-layer learnable log_curvature (共享给 Poincaré + Lorentz, R36 共享机制)
        self.log_curvature = nn.Parameter(torch.zeros(num_layers).fill_(math.log(math.expm1(init_curvature))))
        # learnable 混合权重 log_alpha (sigmoid → α ∈ (0,1))
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha))
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
        return learnable_c * sigmoid_factor  # (num_layers,)

    @property
    def reg_weight(self) -> torch.Tensor:
        return self.reg_weight_max * torch.tanh(self.raw_reg_weight / self.reg_weight_max)

    @property
    def hybrid_alpha(self) -> torch.Tensor:
        """sigmoid(log_alpha) - 混合权重 ∈ (0, 1)."""
        return torch.sigmoid(self.log_alpha)

    def poincare_logmap0(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """logmap0: Poincaré ball B_c^d → tangent space at origin.

        标准公式: logmap0(x) = (arctanh(sqrt(c)*||x||) / (sqrt(c)*||x||)) * x
        当 ||x|| → 0 时 factor → 1, 平滑过渡到 Euclidean。
        """
        sqrt_c = torch.sqrt(c)
        norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        max_norm = (1.0 / sqrt_c - 1e-5)
        norm_x_clamp = norm_x.clamp_max(max_norm)
        factor = torch.arctanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)
        return factor * x

    def lorentz_logmap0_norm_sq(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Lorentz logmap0 模长平方: ||logmap0_L(x)||^2 = alpha^2.

        Lorentz manifold L^d_c = {x ∈ R^{d+1} : -c*x_0^2 + ||x_rest||^2 = -1, x_0 > 0}.
        假设 hidden 作为 Lorentz (d+1)-vec 投影 (无显式约束, 作为 soft reg).

        logmap0_L(x) = (0, alpha_L * x_rest / ||x_rest||) ∈ T_o L^d_c
        alpha_L = arccosh(sqrt(c)*x_0) / sqrt(c)
        若 x 在 L^d_c 上, x_0 = sqrt((||x_rest||^2 + 1/c) / 1), 因此:
        z = sqrt(c) * x_0 = sqrt(c*||x_rest||^2 + 1)
        alpha_L = arccosh(z) / sqrt(c)

        ||logmap0_L||^2 = alpha_L^2 (第一维 0, 余维 unit * alpha_L, 模长 alpha_L).

        Args:
            x: (..., d+1) hidden state 视为 Lorentz 向量
            c: scalar curvature
        Returns:
            (..., ) ||logmap0_L(x)||^2 = alpha_L^2
        """
        c_safe = c.clamp_min(1e-5)
        sqrt_c = c_safe.sqrt()
        x_rest = x[..., 1:]
        x_rest_norm_sq = (x_rest ** 2).sum(dim=-1)  # (...,)
        # z = sqrt(c*||x_rest||^2 + 1), 需 z ≥ 1 for arccosh
        z = (c_safe * x_rest_norm_sq + 1.0).clamp_min(1.0 + 1e-7)
        alpha = torch.acosh(z) / sqrt_c  # (...,)
        return alpha ** 2

    def reg_loss_for_layers(self, hidden_states_list) -> torch.Tensor:
        """Compute hybrid Poincaré + Lorentz reg loss across layers.

        Args:
            hidden_states_list: list of (B, L, d_model) per layer hidden states
                                (encoder_hidden_states from T5)
        Returns:
            total hybrid reg loss (scalar tensor) = mean_l [α * P_reg_l + (1-α) * L_reg_l]
        """
        n = min(len(hidden_states_list), self.num_layers)
        total = 0.0
        c_per_layer = self.annealed_curvature()  # (num_layers,)
        alpha_p = self.hybrid_alpha  # scalar ∈ (0, 1)
        alpha_l = 1.0 - alpha_p
        for i in range(n):
            c = c_per_layer[i]
            h = hidden_states_list[i]  # (B, L, d)

            # Poincaré logmap0 reg
            logmap_p = self.poincare_logmap0(h, c)
            poincare_reg = (logmap_p ** 2).sum(dim=-1).mean()

            # Lorentz logmap0 reg: 把 h 当作 Lorentz (d+1)-vec, x_0 = ||h||
            # 不强制 x 在 L^d_c, 仅作为 soft reg 项
            x_0 = h.norm(dim=-1, keepdim=True)  # (B, L, 1)
            x_lorentz = torch.cat([x_0, h], dim=-1)  # (B, L, d+1)
            lorentz_reg = self.lorentz_logmap0_norm_sq(x_lorentz, c).mean()

            # 混合 (α 可学习)
            layer_reg = alpha_p * poincare_reg + alpha_l * lorentz_reg
            total = total + layer_reg
        return total / max(n, 1)

    def set_epoch(self, epoch: int):
        """Update current_epoch for annealing schedule."""
        self.current_epoch = epoch