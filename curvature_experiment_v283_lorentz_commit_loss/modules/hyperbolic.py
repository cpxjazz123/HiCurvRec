"""双曲几何工具 (M2 Intrinsic Möbius 减法 + M3 跨层曲率传输, Issue #166 treatment 移植).

来源: tasks/Issue166_stage2_sid_attribution/stage2_new/treatment/_lib/cross_layer_quantizer.py
"""
import torch

C_MIN = 0.3  # C10: 0.5→0.3 — 曲率下界扩展 (实测最终 c 全饱和到 0.5 下界, 给 L0/L2 向下空间, 更接近欧氏)
C_MAX = 2.0


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


# === v283 NOVEL: Lorentz manifold helpers (R36n c+e 几何替换 + 几何变换) ===
# 论文支撑: Chami et al. ICML 2019 "Hyperbolic Graph Convolutional Neural Networks"
# Lorentz model 比 Poincaré 在 manifold boundary 数值更稳定 (无 arctanh 饱和).
# 状态: R^d 切空间 (cascade 内部表示) — exp_0_L 升到 d+1 维 Lorentz, commit 走 Lorentz, log_0_L 降回 R^d.
# 数学:
#   origin o = (1/√c, 0, ..., 0), 切空间 T_o H^n = {v ∈ R^(n+1) : v_0 = 0} ≅ R^n
#   ⟨x, y⟩_L = -x_0 y_0 + Σ x_i y_i (Lorentz/Minkowski 内积)
#   ⟨x, x⟩_L = -1/c (hyperboloid 约束)
#   d_L(x, y) = arccosh(-c ⟨x,y⟩_L) / √c


def _lorentz_inner_t(x, y):
    """Lorentz/Minkowski 内积: ⟨x,y⟩_L = -x_0 y_0 + Σ x_i y_i. x,y: (..., D+1)."""
    return -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)


def _lorentz_exp0_t(v, c):
    """v: (..., D) 切空间向量 → (..., D+1) Lorentz manifold point.
    exp_0^c(v) = (cosh(‖v‖/√c)/√c, sinh(‖v‖/√c) · v / (√c · ‖v‖)).
    """
    sqrt_c = c ** 0.5
    norm_v = v.norm(dim=-1, keepdim=True).clamp_min(_eps(v))
    alpha = norm_v * sqrt_c  # 沿 v 测地距离 ‖v‖_L (Euclidean norm in tangent space)
    x_0 = torch.cosh(alpha) / sqrt_c
    x_s = torch.sinh(alpha) * v / (sqrt_c * norm_v)
    return torch.cat([x_0, x_s], dim=-1)


def _lorentz_log0_t(x, c):
    """x: (..., D+1) Lorentz manifold point → (..., D) 切空间向量.
    log_0^c(x) = arccosh(√c x_0) · x_s / (√c · sinh(arccosh(√c x_0)/√c) · ‖x_s‖)
              化简: 用 √(u²-1) = sinh(arccosh(u)), β = arccosh(√c x_0), log = β x_s / sinh(β) (‖x_s‖ 在 x on manifold 上 = sinh(β)/√c).
    """
    sqrt_c = c ** 0.5
    x_0 = x[..., 0:1]
    x_s = x[..., 1:]
    arg = (sqrt_c * x_0).clamp(min=1.0 + 1e-7)
    beta = torch.acosh(arg)  # arccosh(√c x_0)
    sinh_beta = torch.sqrt(arg * arg - 1.0).clamp_min(_eps(arg))  # sinh(arccosh(u)) = √(u²-1)
    # log_0 = (β / sinh(β)) · x_s (Chami et al. 2019 Eq. 8 simplified)
    factor = beta / sinh_beta
    return factor * x_s


def _lorentz_mobius_add_t(x, y, c):
    """x, y: (..., D+1) Lorentz points. Closed-form gyrogroup add (Ungar 2008 / Geoopt).
    Geoopt 实现 (k = -c 代入, ⟨y,y⟩_L = -1/c):
      num = (1 + 2c⟨x,y⟩_L) x + (3 - 2c⟨x,y⟩_L) y
      den = 3 + 6c⟨x,y⟩_L + 4(c⟨x,y⟩_L)²
    Reference: geoopt.manifolds.lorentz.math.l_add (Chami et al. ICML 2019 Algorithm 1).
    """
    xy = _lorentz_inner_t(x, y)  # (..., 1) Minkowski 内积
    cxy = c * xy
    # Geoopt form (k = -c, y_norm_sq = -1/c 代入):
    num = (1.0 + 2.0 * cxy) * x + (3.0 - 2.0 * cxy) * y
    den = 3.0 + 6.0 * cxy + 4.0 * cxy * cxy
    return num / den.clamp_min(_eps(den))


def _lorentz_distance_t(x, y, c):
    """x, y: (..., D+1) Lorentz points. d_L(x,y) = arccosh(-c ⟨x,y⟩_L) / √c.
    用于 codebook argmin 距离 (Chami et al. 2019): Lorentz 距离比 Poincaré 在 ball boundary 更稳定.
    """
    sqrt_c = c ** 0.5
    xy = _lorentz_inner_t(x, y)
    arg = (-c * xy).clamp(min=1.0 + 1e-7)
    return torch.acosh(arg) / sqrt_c


def _lorentz_from_poincare_t(p, c):
    """Poincaré ball 点 p (‖p‖ < 1/√c) → Lorentz hyperboloid 点 (d+1 维).
    映射 (Chami et al. 2019 / Nickel & Kiela 2017):
      x_0 = (1 + c‖p‖²) / (√c (1 - c‖p‖²))
      x_s = 2√c · p / (1 - c‖p‖²)
    """
    sqrt_c = c ** 0.5
    p_norm_sq = (p * p).sum(dim=-1, keepdim=True)  # (..., 1)
    denom = (1.0 - c * p_norm_sq).clamp_min(_eps(p))
    x_0 = (1.0 + c * p_norm_sq) / (sqrt_c * denom)
    x_s = 2.0 * sqrt_c * p / denom
    return torch.cat([x_0, x_s], dim=-1)


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

