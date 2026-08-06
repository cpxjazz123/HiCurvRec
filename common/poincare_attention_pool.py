"""Poincaré Attention Pooling — 双曲空间版的 DECOR PromptFormer 上下文摘要.

设计 (vs DECOR Euclidean attention pooling):
  fused_embeds (B, L, D) 欧氏向量 → 投影到 Poincaré ball
  bos_queries (N, D) learnable points on ball (norm<1)
  e_ctx = expmap0(mean(logmap0(fused_embeds_p)))  # 流形上的 batch 摘要
  attn = softmax(-poincare_distance(e_ctx, bos_queries))  # 双曲距离代替点积
  bos_vec = 加权组合 (mobius_matvec)
  fused_embeds = fused_embeds + gate * bos_vec.unsqueeze(1)  # 注入每个位置

数学验证 (Eckart-Young 类似):
  - logmap0 + mean + expmap0 是双曲 Frechet mean (唯一)
  - poincare_distance 在 norm<1 时单调, 距离 0 = 同一点, 距离 ∞ = 边界
  - 与 Euclidean attention 的关键差异: 层次结构敏感性

参考:
  - Ganea et al. Hyperbolic Neural Networks (NeurIPS 2018)
  - Chami et al. Hyperbolic Graph Convolutional Neural Networks (NeurIPS 2019)
  - Nickel & Kiela, Poincaré Embeddings (NeurIPS 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-5
_MAX_NORM = 0.9999


def _safe_norm(x, dim=-1, keepdim=True, max_norm=_MAX_NORM):
    """norm with clamping to keep inside Poincaré ball."""
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim).clamp(min=_EPS, max=max_norm)
    return n


def poincare_distance(x, y, c=1.0):
    """Poincaré distance between x and y on ball of curvature c.

    d(x,y) = (1/sqrt(c)) * arccosh(1 + 2c*||x-y||²/((1-c||x||²)(1-c||y||²)))

    Shapes: (..., 1, D) and (..., N, D) → (..., N)
    """
    sqrt_c = c ** 0.5
    x_sq = (x * x).sum(dim=-1, keepdim=True).clamp(min=0, max=_MAX_NORM ** 2)  # (..., 1, 1)
    y_sq = (y * y).sum(dim=-1, keepdim=True).clamp(min=0, max=_MAX_NORM ** 2)  # (..., N, 1)
    diff_sq = ((x - y) ** 2).sum(dim=-1, keepdim=True)                          # (..., N, 1)
    num = 2 * c * diff_sq
    denom = (1 - c * x_sq) * (1 - c * y_sq)
    arg = 1 + num / denom.clamp(min=_EPS)
    # arccosh 需 input >= 1
    arg = arg.clamp(min=1.0 + _EPS)
    return (1 / sqrt_c) * torch.arccosh(arg).squeeze(-1)                          # (..., N)


def expmap0(v, c=1.0):
    """Euclidean tangent vector → Poincaré ball point. v at origin: tanh(sqrt(c)*||v||) * v / (sqrt(c)*||v||)"""
    sqrt_c = c ** 0.5
    v_norm = _safe_norm(v, dim=-1, keepdim=True)
    return torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm) * v


def logmap0(y, c=1.0):
    """Poincaré ball point → Euclidean tangent vector at origin."""
    sqrt_c = c ** 0.5
    y_norm = _safe_norm(y, dim=-1, keepdim=True)
    return torch.atanh(sqrt_c * y_norm) / (sqrt_c * y_norm) * y


def mobius_matvec(M, x, c=1.0):
    """Apply matrix M ∈ R^{m×n} to point x ∈ D^n on ball, output D^m.

    Reference: Ganea 2018 Eq.(7): M⊗x = (1/sqrt(c)) tanh(||Mx||/||x|| atanh(sqrt(c)||x||)) Mx/||Mx||
    """
    sqrt_c = c ** 0.5
    x_norm = _safe_norm(x, dim=-1, keepdim=True)
    Mx = x @ M.T                                  # (..., m)
    Mx_norm = _safe_norm(Mx, dim=-1, keepdim=True)
    return (torch.tanh(Mx_norm * torch.atanh(sqrt_c * x_norm) / x_norm.clamp(min=_EPS))
            / (Mx_norm * sqrt_c)) * Mx


class PoincareAttentionPool(nn.Module):
    """Poincaré-ball norm-weighted attention pooling (路径 A v2).

    关键设计 (修复 v1 缺陷):
      attn = softmax(-raw_norm / tau) 直接用原始 fused_embeds norm 作 logit
      → norm 小的 token (抽象) attention 权重高
      → 加权求和时, norm 小的 token 主导方向 (向量本来就短, 但权重高)
      → bos_vec 接近 norm 最小的 token

    数学流程:
      1. 算 raw norm = ||fused_embeds|| (欧氏, 不投影)
      2. attention = softmax(-norm / tau) — norm 小 → logit 大 → 权重高
      3. bos_vec = sum_i attn_i * fused_embeds_i (欧氏加权)
      4. (可选) 投影到 Poincaré ball 仅用于注入, 不影响加权
      5. 注入到 fused_embeds: alpha * bos_vec 加到每个位置

    Args:
        d_model: T5 d_model (128 for HG-Rec).
        alpha_init: initial injection gate (sigmoid'd to (0, 1)).
        tau_init: initial temperature for softmax(-norm/tau) (default 1.0).
        use_learnable_tau: if True, tau becomes learnable.
    """

    def __init__(self, d_model, alpha_init=0.1, tau_init=1.0,
                 use_learnable_tau=True, proj_scale=0.99):
        super().__init__()
        self.d_model = d_model
        self.proj_scale = float(proj_scale)

        # temperature 可学习, 控制 norm-weight 锐度
        if use_learnable_tau:
            self.tau_raw = nn.Parameter(torch.tensor(tau_init), requires_grad=True)
        else:
            self.register_buffer("tau_raw", torch.tensor(tau_init))

        # output gate: 控制 bos_vec 注入强度
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)

    def forward(self, fused_embeds, attention_mask=None):
        """Compute bos_vec (B, D) via norm-based softmax pooling and inject.

        Args:
            fused_embeds: (B, L, D) — T5 standard lookup (欧氏向量)
            attention_mask: (B, L) — 1 = valid, 0 = pad

        Returns:
            fused_embeds_with_pool: (B, L, D) — 注入 bos_vec 后
            bos_vec: (B, D) — 池化出的 batch 摘要
        """
        B, L, D = fused_embeds.shape

        # 1. 原始 norm (欧氏, 不投影 — 这是修复 v1 的关键)
        x_norm = torch.linalg.vector_norm(fused_embeds, dim=-1).clamp(min=_EPS)  # (B, L)

        # 2. mask 屏蔽 PAD
        if attention_mask is not None:
            mask = attention_mask.float()                              # (B, L)
        else:
            mask = torch.ones(B, L, device=fused_embeds.device)

        # 3. norm-based softmax attention: attn_i ∝ exp(-norm_i / tau)
        #    norm 小 → logit 大 → 权重高
        tau = self.tau_raw.abs() + _EPS  # tau > 0
        logits = -x_norm / tau                                             # (B, L)
        # mask = 0 → logit 极小 (不用 -inf 避免 NaN, 用 -1e9 让 softmax ≈ 0)
        logits = logits.masked_fill(mask == 0, -1e9)
        attn = F.softmax(logits, dim=-1)                                   # (B, L) 归一化
        # 安全: 全 PAD 行 attn 应全 0 (mask=0 时 logits=-1e9, softmax 后 ≈ 0)
        # 但 NaN 安全: clamp
        attn = attn.nan_to_num(0.0)

        # 4. 加权求和: bos_vec = sum_i attn_i * fused_embeds_i (欧氏加权)
        bos_vec = (attn.unsqueeze(-1) * fused_embeds).sum(dim=1)          # (B, D)

        # 5. (可选) 投影 bos_vec 到 Poincaré ball, 仅用于注入 (不影响加权)
        #    保证 bos_vec 也有 norm<1 的几何约束
        bos_norm = torch.linalg.vector_norm(bos_vec, dim=-1, keepdim=True).clamp(min=_EPS)
        bos_vec_p = torch.tanh(self.proj_scale * bos_norm) / bos_norm * bos_vec

        # 6. 注入到 fused_embeds (欧氏空间加, gate 控制强度)
        alpha = torch.sigmoid(self.alpha_raw)
        fused_embeds_with_pool = fused_embeds + alpha * bos_vec_p.unsqueeze(1)

        return fused_embeds_with_pool, bos_vec_p