"""双曲几何工具 (M2 Intrinsic Möbius 减法 + M3 跨层曲率传输, Issue #166 treatment 移植).

来源: tasks/Issue166_stage2_sid_attribution/stage2_new/treatment/_lib/cross_layer_quantizer.py

v322 NOVEL: 增 Lorentz manifold 操作 (R36n c manifold 几何替换 — Poincaré → Lorentz)
  论文支撑: Law et al. NeurIPS 2019 "Lorentzian Distance Learning for Hyperbolic Representations"
            Nickel & Kiela 2017 Poincaré Embeddings
"""
import torch

C_MIN = 0.3  # C10: 0.5→0.3 — 曲率下界扩展 (实测最终 c 全饱和到 0.5 下界, 给 L0/L2 向下空间, 更接近欧氏)
C_MAX = 2.0


# === v322: Lorentz manifold 工具 (R36n c 全新子方向) ===
# Lorentz manifold L^n_c = {y ∈ R^{n+1} : -y_0^2 + y_1^2 + ... + y_n^2 = -1/c, y_0 > 0}
# 数值优势: acosh 而非 artanh (无除法 norm, 梯度更平滑, 不需要 project to ball)
# 内积: ⟨x,y⟩_L = -x_0 y_0 + Σ_{i≥1} x_i y_i
# 距离: d_L(x,y) = (1/√c) · acosh(-c · ⟨x,y⟩_L)

def _euclidean_to_lorentz_t(x, c, eps=1e-6):
    """Euclidean x ∈ R^d → Lorentz manifold L^d_c.

    显式构造: y_0 = √(‖x‖² + 1) / √c,  y_i = x_i / √c
    验证约束: -y_0^2 + Σ y_i^2 = -(‖x‖²+1)/c + ‖x‖²/c = -1/c ✓
    保证 y_0 > 0 + Lorentz constraint -y_0^2 + Σ y_i^2 = -1/c.
    x: (..., D); 输出 (..., D+1)
    """
    sqrt_c = c ** 0.5
    x_sq = (x * x).sum(dim=-1, keepdim=True)              # (..., 1)
    y0 = (x_sq + 1.0).clamp_min(eps).sqrt() / sqrt_c      # (..., 1)  y_0 > 0
    yi = x / sqrt_c                                        # (..., D)
    return torch.cat([y0, yi], dim=-1)                    # (..., D+1)


def _lorentz_inner_t(x, y):
    """Lorentzian (Minkowski) 内积 ⟨x,y⟩_L = -x_0 y_0 + Σ_{i≥1} x_i y_i.
    x, y: (..., D+1). 假设 D+1 ≥ 1.
    """
    x0 = x[..., 0:1]
    y0 = y[..., 0:1]
    spatial = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return -x0 * y0 + spatial


def _lorentz_distance_t(x, y, c, eps=1e-6):
    """Lorentzian 距离 d_L(x,y) = (1/√c) · acosh(-c · ⟨x,y⟩_L).

    对 Lorentz manifold 上的点 (满足 -y_0^2 + Σ y_i^2 = -1/c) 计算.
    x, y: (..., D+1); c: (1,)
    """
    sqrt_c = c ** 0.5
    inner = _lorentz_inner_t(x, y)                          # (..., 1)
    # -c · ⟨x,y⟩_L ≥ 1 (Cauchy-Schwarz on Lorentz), clamp 到 [1+eps, inf)
    z = (-c * inner).clamp(min=1.0 + eps)
    return torch.acosh(z) / sqrt_c


def _lorentz_to_poincare_t(x, c, eps=1e-6):
    """Lorentz → Poincaré ball 投影 (供 Stage 2 推理 + sinkhorn 复用).
    x: (..., D+1), 取空间分量 x[:, 1:] / x[:, 0:1] / √c ∈ ball (D,)
    """
    return x[..., 1:] / x[..., 0:1].clamp_min(eps) / (c ** 0.5)


def _poincare_to_lorentz_t(x, c, eps=1e-6):
    """Poincaré ball x ∈ R^d → Lorentz manifold (供 Stage 1 与 Lorentz codebook 兼容).
    公式: y_0 = (1 + c‖x‖²)/(1 - c‖x‖²) / √c,  y_i = 2x_i / √c / (1 - c‖x‖²)
    保证 y_0 > 0 + 双曲约束.
    """
    sqrt_c = c ** 0.5
    x_sq = c * (x * x).sum(dim=-1, keepdim=True)
    den = (1.0 - x_sq).clamp_min(eps)
    y0 = (1.0 + x_sq) / den / sqrt_c       # (..., 1)
    yi = 2.0 * x / den / sqrt_c             # (..., D)
    return torch.cat([y0, yi], dim=-1)


def _eps(x, default=1e-6):
    return torch.finfo(x.dtype).eps if x.dtype.is_floating_point else default

def _proj_to_ball_t(x, c, eps=1e-6):
    r = (1.0 / c) ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) * r
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def _expmap0_t(u, c):
    sqrt_c = c ** 0.5
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(_eps(u))
    factor = torch.tanh(sqrt_c * norm_u) / (sqrt_c * norm_u)
    return _proj_to_ball_t(factor * u, c)


def _logmap0_t(x, c):
    sqrt_c = c ** 0.5
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
    factor = torch.atanh((sqrt_c * norm_x).clamp(max=1 - 1e-5)) / (sqrt_c * norm_x)
    return factor * x


def _transport_between_t(x, c_from, c_next, eps=1e-6):
    """M3 fused: 跨层曲率传输 Exp_0^{c2}(Log_0^{c1}(x)) 径向闭式 (无逐段 kernel).

    u = logmap_c1(x) → ||u|| = atanh(sqrt(c1)*r)/sqrt(c1)
    expmap_c2(u) = tanh(sqrt(c2)*||u||)/(sqrt(c2)*||u||) * u
    合成: out = tanh(b*atanh(a)) / (b*r) * x,  a = sqrt(c1)*r, b = sqrt(c2/c1)
    c1==c2 时退化为恒等 (tanh(atanh(a))/a * x = x).
    """
    r = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    a = (c_from ** 0.5) * r
    b = (c_next / c_from) ** 0.5
    a_safe = a.clamp(max=1 - 1e-5)
    factor = torch.tanh(b * torch.atanh(a_safe)) / (b * a)
    return factor * x


def _sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    """Sinkhorn-Knopp 均衡分配 (HG-Rec utils.py 拷贝): 把距离矩阵转成 balanced assignment.

    行/列交替归一化强制每个 codebook 条目被均衡使用 → 直接防 codebook collapse.
    distances: (B, K); epsilon: 温度; iterations: 迭代数.
    """
    Q = torch.exp(-distances / epsilon)
    B = Q.shape[0]
    K = Q.shape[1]
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    for _ in range(sinkhorn_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K
    Q *= B
    return Q


def _poincare_distance_t(x, y, c, eps=1e-6):
    """Poincaré 双曲距离 (HG-Rec: d_c(x,y) = 2/√c · artanh(√c · ||(-x) ⊕_c y||)).

    用于 codebook 分配 argmin — 双曲空间距离结构使分配更均匀, 缓解 codebook collapse
    (HG-Rec loss_type='poincare' 机制复制).
    x, y: (..., D) 或 (B, K, D) 广播; c: (1,)
    """
    neg_x = -x
    u = _mobius_add_t(neg_x, y, c)
    norm_u = u.norm(dim=-1, keepdim=True).clamp(max=1 - 1e-5)
    return (2.0 / (c ** 0.5)) * torch.atanh(norm_u)


def _mobius_add_t(x, y, c):
    """tensor-safe Möbius addition (c broadcastable, 用于 per-item 内在 residual 减法)."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / den.clamp_min(_eps(den))


def _artanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def _sigmoid(x):
    return torch.sigmoid(x)


def _mlr_logits_t(x, a, r, c, eps=1e-6):
    """C21 (HyperVQ, ICML 2025): Unidirectional 双曲 MLR logits.

    logits_k(x) = (λ_qk·‖a_k‖/√c)·artanh(√c·⟨−q_k⊕_c x, a_k⟩/(λ_qk·‖a_k‖))
    q_k = exp_0^c(r_k·[a_k])  (超平面代表点, [a_k] = 归一化方向)
    λ_qk = 2/(1−c·‖q_k‖²)

    替代最近邻距离分配 — 用双曲超平面判别, 提升 codebook 利用率与解耦.
    x: (..., D) 投影后的 Poincaré 点; a: (K,D) 法向量; r: (K,) 标量; c: 曲率.
    """
    sqrt_c = c ** 0.5
    a_norm = a.norm(dim=-1, keepdim=True).clamp_min(eps)      # (K,1)
    a_dir = a / a_norm                                        # (K,D)
    q = _expmap0_t(a_dir * r.unsqueeze(-1), c)                # (K,D) 代表点
    q_norm_sq = (q * q).sum(-1, keepdim=True)                 # (K,1)
    lam_q = (2.0 / (1.0 - c * q_norm_sq)).clamp_min(eps)      # (K,1)
    mob = _mobius_add_t(-q.unsqueeze(0), x.unsqueeze(1), c)   # (...,K,D)
    inner = (mob * a.unsqueeze(0)).sum(-1)                    # (...,K)
    scale = (lam_q.squeeze(-1) * a_norm.squeeze(-1)) / sqrt_c  # (K,)
    denom = (lam_q.squeeze(-1) * a_norm.squeeze(-1)).clamp_min(eps)
    arg = (sqrt_c * inner / denom).clamp(max=1 - 1e-5)
    logits = scale * torch.atanh(arg)                         # (...,K)
    return logits

