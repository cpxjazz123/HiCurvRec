"""Issue #125 (2026-08-12): Stage3 per-head learnable curvature (κ_h).

设计:
- 给 HAB (HyperbolicAttentionBias) get_B_geo 输出从 (B, 1, L, L) 扩展到 (B, num_heads, L, L).
- 每个 attention head 用不同 κ_h (learnable, init=0.5) 缩放 B_geo 距离.
- κ_h learnable: nn.Parameter, num_heads=6 (T5 默认).
- lambda_h_raw learnable: nn.Parameter, num_heads=6, init=0 (λ=0 → 不影响 baseline, 训练中学到合适值).

公式:
  final_B_geo[b, h, i, j] = -λ_h_eff[h] * exp(κ_h[h]) * B_geo_orig[b, 0, i, j]
  λ_h_eff[h] = λ_max * tanh(λ_h_raw[h] / λ_max)

R36 曲率机制变更 (新增 per-head learnable curvature, 不算调参).
R35 单 ckpt + beam=20 仍生效 (per-head bias 不影响 inference beam 数量).

安装:
  from per_head_curvature import install_per_head_curvature
  hab_module = install_per_head_curvature(hab_module, num_heads=6, kappa_h_init=0.5, lambda_h_init=0.0)

注意:
- 必须在 install_hab 之后调用, 这样 get_B_geo 才会被 monkey-patch
- per-head 参数加到 hab_module 上, 跟 Stage3 训练一起梯度下降
- Stage3 ckpt 保存时会自动保存 kappa_h + lambda_h_raw (作为 hab_module 的参数)
"""
import types
import torch
import torch.nn as nn


# Issue125 hardcoded defaults (R30+R43 禁调参)
PER_HEAD_NUM_HEADS_DEFAULT = 6  # T5 默认 num_heads=6
PER_HEAD_KAPPA_H_INIT = 0.5  # κ_h init: c = exp(0.5) ≈ 1.65 (健康)
PER_HEAD_LAMBDA_H_INIT = 0.0  # λ_h init=0 → 训练初期不贡献 (与 baseline 严格等价)
PER_HEAD_LAMBDA_H_MAX = 0.20  # λ_h 上限 (与 HAB_LAMBDA_MAX 对齐)


def install_per_head_curvature(hab_module, num_heads=PER_HEAD_NUM_HEADS_DEFAULT,
                                kappa_h_init=PER_HEAD_KAPPA_H_INIT,
                                lambda_h_init=PER_HEAD_LAMBDA_H_INIT,
                                lambda_h_max=PER_HEAD_LAMBDA_H_MAX):
    """给 HAB get_B_geo 加 per-head learnable curvature (Issue125 spec).

    Args:
        hab_module: HyperbolicAttentionBias 实例 (已经被 install_hab 创建)
        num_heads: T5 attention head 数量 (默认 6)
        kappa_h_init: κ_h 初始值 (默认 0.5)
        lambda_h_init: λ_h_raw 初始值 (默认 0.0 → 训练初期不贡献)
        lambda_h_max: λ_h 上限 (默认 0.20)

    Returns:
        hab_module (modified in-place, 含 kappa_h + lambda_h_raw + num_heads 属性)

    Side effects:
        - hab_module.get_B_geo 被 monkey-patch, 输出 (B, num_heads, L, L) 而不是 (B, 1, L, L)
        - 新增 nn.Parameter kappa_h (num_heads,)
        - 新增 nn.Parameter lambda_h_raw (num_heads,)
    """
    # 1. 注册 per-head 参数 (作为 hab_module 子模块的 Parameter)
    # 关键: 创建后立即移到 hab_module 当前 device (install_hab 已 to(device), 但新 param 是 CPU)
    ref_param = next(hab_module.parameters(), None)
    target_device = ref_param.device if ref_param is not None else torch.device("cpu")
    if not hasattr(hab_module, "kappa_h"):
        kappa_h = nn.Parameter(torch.full((num_heads,), float(kappa_h_init), device=target_device))
        # 用 setattr 绕过 nn.Module __setattr__ 检查 (直接注册到 _parameters)
        hab_module._parameters["kappa_h"] = kappa_h
    if not hasattr(hab_module, "lambda_h_raw"):
        lambda_h_raw = nn.Parameter(torch.full((num_heads,), float(lambda_h_init), device=target_device))
        hab_module._parameters["lambda_h_raw"] = lambda_h_raw
    hab_module.num_heads = int(num_heads)
    hab_module.lambda_h_max = float(lambda_h_max)

    # 2. 缓存原始 get_B_geo (如果还没缓存)
    if not hasattr(hab_module, "_original_get_B_geo_per_head"):
        hab_module._original_get_B_geo_per_head = hab_module.get_B_geo

    # 3. 定义 per-head 版本的 get_B_geo (monkey-patch)
    def get_B_geo_per_head(self, input_ids, layer_id_lut_tensor, attention_mask_2d=None):
        # 调原始 get_B_geo 得到 (B, 1, L, L)
        B_geo_orig = self._original_get_B_geo_per_head(
            input_ids, layer_id_lut_tensor, attention_mask_2d)
        # B_geo_orig shape: (B, 1, L, L)
        # per-head 缩放: kappa_h[h] 调整距离 scale (类似 c=exp(κ) 参数化)
        kappa_h_scale = torch.exp(self.kappa_h)  # (num_heads,)
        # per-head lambda
        lambda_h_eff = self.lambda_h_max * torch.tanh(self.lambda_h_raw / self.lambda_h_max)  # (num_heads,)
        # final_B_geo[b, h, i, j] = -lambda_h_eff[h] * kappa_h_scale[h] * B_geo_orig[b, 0, i, j]
        # broadcast: (1, num_heads, 1, 1) * (B, 1, L, L) → (B, num_heads, L, L)
        per_head_scale = (-lambda_h_eff * kappa_h_scale).view(1, -1, 1, 1)  # (1, num_heads, 1, 1)
        B_geo_per_head = per_head_scale * B_geo_orig  # (B, num_heads, L, L)
        return B_geo_per_head

    hab_module.get_B_geo = types.MethodType(get_B_geo_per_head, hab_module)
    return hab_module


def get_per_head_kappa_stats(hab_module):
    """返回当前 per-head κ 状态 (用于 verdict 写盘).

    Returns:
        dict with kappa_h (list), lambda_h_raw (list), num_heads (int)
    """
    if not hasattr(hab_module, "kappa_h"):
        return {"installed": False}
    return {
        "installed": True,
        "num_heads": int(hab_module.num_heads),
        "kappa_h": [float(x) for x in hab_module.kappa_h.detach().cpu().tolist()],
        "kappa_h_grad": [float(x) for x in hab_module.kappa_h.grad.detach().cpu().tolist()] if hab_module.kappa_h.grad is not None else None,
        "lambda_h_raw": [float(x) for x in hab_module.lambda_h_raw.detach().cpu().tolist()],
        "lambda_h_eff": [float(x) for x in (hab_module.lambda_h_max * torch.tanh(hab_module.lambda_h_raw / hab_module.lambda_h_max)).detach().cpu().tolist()],
    }