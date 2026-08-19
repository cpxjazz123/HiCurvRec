"""v40 (Issue260) — Stage 3 T5 LayerNorm → Poincaré LayerNorm (R36 几何变换).

机制 (HNN/Gülçehre 2019 经典):
- 输入 x ∈ R^D
- logmap0(x, c) → 切空间 Euclidean → RMSNorm → expmap0 → Poincaré
- 整个流程保持向量在 Poincaré ball 上 (‖x‖ < 1/√c)

实现策略:
- 创建一个 PoincareT5LayerNorm class (接口与 T5LayerNorm 一致: __init__(hidden_size, eps) + forward(hidden_states))
- 在 HG_Rec.__init__ 创建 T5ForConditionalGeneration 后, 遍历 named_modules() 找到所有 T5LayerNorm 实例
- 替换为 PoincareT5LayerNorm 实例 (新可学习参数 + 共享 weight 形状)

C_fixed (默认 0.5) 与 v19 Stage 1 curriculum 末态一致 (R36 框架级变更, 不调参).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _expmap0_t(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Poincaré ball expmap at 0: y = tanh(√c ‖x‖) x / (√c ‖x‖).
    x: (..., D), c: scalar tensor.
    """
    sqrt_c = torch.sqrt(c.clamp_min(1e-10))
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    max_norm = (1.0 / sqrt_c - 1e-5)
    norm_x_clamp = norm_x.clamp_max(max_norm)
    factor = torch.tanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)
    return factor * x


def _logmap0_t(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Poincaré ball logmap at 0: y = arctanh(√c ‖x‖) x / (√c ‖x‖).
    x: (..., D), c: scalar tensor.
    """
    sqrt_c = torch.sqrt(c.clamp_min(1e-10))
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    max_norm = (1.0 / sqrt_c - 1e-5)
    norm_x_clamp = norm_x.clamp_max(max_norm)
    factor = torch.arctanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)
    return factor * x


class PoincareT5LayerNorm(nn.Module):
    """Poincaré ball RMSNorm (替换 transformers T5LayerNorm).

    forward(x):
        x_tan = logmap0(x, c)        # Euclidean 切空间
        x_norm = rms_scale(x_tan)    # 与 T5LayerNorm 相同的 RMSNorm
        x_p = expmap0(x_norm, c)     # 投影回 Poincaré ball
        # 保留 T5LayerNorm 的 weight (可学习 scale)
        return weight * x_p           # 注意 weight 是 R^D, 标量缩放在 Poincaré ball 不适用

    实际上 T5LayerNorm 的 weight 是 R^D 逐元素 scale. 在 Poincaré 几何里, 我们保持 logmap/expmap 框架
    但 weight 应用在 切空间 (logmap 之后) 更符合几何语义:
        x_tan = logmap0(x, c)
        x_norm = rms_scale(x_tan)
        x_norm = x_norm * weight   # 切空间逐元素 scale (与 Euclidean RMSNorm 等价的几何变换)
        x_p = expmap0(x_norm, c)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, c: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        # v40: 固定 c (R36 不调参, 默认与 v19 Stage 1 curriculum 末态 0.5 一致)
        self.c = float(c)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1. logmap0: 投影到切空间 (Euclidean)
        c = torch.tensor(self.c, device=hidden_states.device, dtype=hidden_states.dtype)
        x_tan = _logmap0_t(hidden_states, c)

        # 2. RMSNorm (与 T5LayerNorm 完全一致, only scale, no mean, no bias)
        variance = x_tan.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_norm = x_tan * torch.rsqrt(variance + self.variance_epsilon)
        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x_norm = x_norm.to(self.weight.dtype)
        x_norm = self.weight * x_norm

        # 3. expmap0: 投影回 Poincaré ball
        x_p = _expmap0_t(x_norm, c)
        return x_p


def replace_t5_layernorm(model: nn.Module, c: float = 0.5) -> int:
    """遍历 model.named_modules(), 找到所有 T5LayerNorm 实例并替换为 PoincareT5LayerNorm.

    Returns:
        n_replaced: 替换数量 (供 log 报告)
    """
    from transformers.models.t5.modeling_t5 import T5LayerNorm
    n_replaced = 0
    # 先收集 (parent_name, child_name, module)
    replacements = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, T5LayerNorm):
                replacements.append((parent, child_name, child))

    # 替换 (避免迭代时修改)
    for parent, child_name, old_ln in replacements:
        hidden_size = old_ln.weight.shape[0]
        eps = old_ln.variance_epsilon
        new_ln = PoincareT5LayerNorm(hidden_size=hidden_size, eps=eps, c=c)
        # 共享 weight (不重新初始化, 保持与原 T5LayerNorm 完全一致起点)
        with torch.no_grad():
            new_ln.weight.copy_(old_ln.weight)
        # 移到相同 device/dtype
        new_ln = new_ln.to(device=old_ln.weight.device, dtype=old_ln.weight.dtype)
        setattr(parent, child_name, new_ln)
        n_replaced += 1
    return n_replaced