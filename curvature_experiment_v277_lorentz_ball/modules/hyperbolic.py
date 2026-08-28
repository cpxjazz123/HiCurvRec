"""双曲几何工具 (M2 Intrinsic Möbius 减法 + M3 跨层曲率传输, Issue #166 treatment 移植).

来源: tasks/Issue166_stage2_sid_attribution/stage2_new/treatment/_lib/cross_layer_quantizer.py

v277 NOVEL (R36n c): Lorentz ball manifold 替换 (Nickel & Kiela 2018 + Ganea 2018).
USE_LORENTZ_BALL=True 时, _expmap0_t / _logmap0_t / _mobius_add_t 全部走 Lorentz 公式.
Lorentz model: H^n_c = {x ∈ R^{n+1} : -x_0² + x_1² + ... + x_n² = -1/c, x_0 > 0}
origin o = (1/√c, 0, ..., 0). tangent space @ o = R^n (Euclidean, x_0=0 implicit).
"""
import torch

C_MIN = 0.3  # C10: 0.5→0.3 — 曲率下界扩展 (实测最终 c 全饱和到 0.5 下界, 给 L0/L2 向下空间, 更接近欧氏)
C_MAX = 2.0

# v277: Lorentz ball manifold 替换 flag (R36n c, Nickel & Kiela 2018)
USE_LORENTZ_BALL = False  # 默认 False (Poincaré). v277 启用 True


def _eps(x, default=1e-6):
    return torch.finfo(x.dtype).eps if x.dtype.is_floating_point else default

def _proj_to_ball_t(x, c, eps=1e-6):
    r = (1.0 / c) ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) * r
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


# === v277 Lorentz ball 实现 (Nickel & Kiela 2018) ===
def _lorentz_project_t(x, c, eps=1e-6):
    """Lorentz ball 投影: ensure -x_0² + sum(x_1..n²) = -1/c, x_0 > 0.
    x: (..., D+1) — last dim 是 x_0 (time-like). Reshape 为 (..., D+1) with x[..., 0] = x_0.
    """
    x0 = x[..., 0:1]
    spatial = x[..., 1:]
    # 约束: spatial_norm² - x0² = -1/c  → x0² = spatial_norm² + 1/c
    spatial_norm_sq = (spatial * spatial).sum(dim=-1, keepdim=True)
    target_x0 = (spatial_norm_sq + 1.0 / c).clamp_min(eps).sqrt()
    return torch.cat([target_x0, spatial], dim=-1)


def _lorentz_inner_t(x, y):
    """Lorentzian inner product: -x_0 y_0 + sum(x_i y_i). Returns (..., 1) for broadcast."""
    x0 = x[..., 0:1]  # keepdim
    y0 = y[..., 0:1]  # keepdim
    spatial_dot = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return -x0 * y0 + spatial_dot


def _lorentz_distance_t(x, y, c, eps=1e-6):
    """Lorentz distance: d_L(x, y) = (1/√c) arccosh(-c ⟨x, y⟩_L).
    x, y: (..., D+1) Lorentz points; c: scalar."""
    inner = _lorentz_inner_t(x, y)
    # -c * inner must be >= 1 for arccosh; clamp for numerical stability
    arg = (-c * inner).clamp(min=1.0 + eps)
    return (1.0 / c.sqrt()) * torch.acosh(arg)


def _expmap0_lorentz_t(v, c, eps=1e-6):
    """Exponential map at Lorentz origin: exp_o(v) for v in R^n (tangent @ origin).
    v: (..., n) — n 维切空间 (no time-like dim).
    exp_o(v) = (cosh(√c‖v‖)·1/√c, sinh(√c‖v‖)·v/(√c‖v‖))
    """
    sqrt_c = c.sqrt()
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    sqrt_c_v = sqrt_c * v_norm
    x0 = torch.cosh(sqrt_c_v) / sqrt_c
    spatial = torch.sinh(sqrt_c_v) * v / (sqrt_c * v_norm)
    return torch.cat([x0, spatial], dim=-1)


def _logmap0_lorentz_t(x, c, eps=1e-6):
    """Log map at Lorentz origin: log_o(x) for x in H^n_c.
    x: (..., n+1) Lorentz point. Returns v in R^n (no time-like dim).
    推导: log_o(x) = (1/√c) arccosh(√c x_0) / ||x_spatial|| · x_spatial
    """
    sqrt_c = c.sqrt()
    x0 = x[..., 0:1]
    spatial = x[..., 1:]
    spatial_norm = spatial.norm(dim=-1, keepdim=True).clamp_min(eps)
    sqrt_c_x0 = (sqrt_c * x0).clamp(min=1.0 + eps)
    arccosh_x0 = torch.acosh(sqrt_c_x0)
    factor = arccosh_x0 / (sqrt_c * spatial_norm)
    return factor * spatial


def _mobius_add_lorentz_t(x, y, c, eps=1e-6):
    """Lorentz 版本 Möbius add: 用 exp/log 闭式 — for tangent-space add only.
    注意: Lorentz 没有 closed-form Möbius add 直接定义 (Poincaré 才有).
    v277 用 exp_o(log_o(x) + log_o(y)) 作 tangent-space 近似.
    """
    u_x = _logmap0_lorentz_t(x, c, eps)
    u_y = _logmap0_lorentz_t(y, c, eps)
    return _expmap0_lorentz_t(u_x + u_y, c, eps)


def _transport_between_lorentz_t(x, c_from, c_next, eps=1e-6):
    """M3 Lorentz 版: 跨层曲率传输 Exp_o^{c_next}(Log_o^{c_from}(x)).
    """
    u = _logmap0_lorentz_t(x, c_from, eps)
    return _expmap0_lorentz_t(u, c_next, eps)


# === wrappers: 根据 USE_LORENTZ_BALL flag 路由 ===
def _euclid_to_lorentz_t(z, c, eps=1e-6):
    """Lift D-dim Euclidean point z to D+1-dim Lorentz point: x = (√(1/c+||z||²), z).
    Internal helper: hide D+1 dim behind D-dim interface for RqVae."""
    z_norm_sq = (z * z).sum(dim=-1, keepdim=True)
    x0 = (z_norm_sq + 1.0 / c).clamp_min(eps).sqrt()
    return torch.cat([x0, z], dim=-1)


def _lorentz_to_euclid_t(x, eps=1e-6):
    """Project D+1-dim Lorentz point back to D-dim Euclidean: drop time-like dim x_0."""
    return x[..., 1:]


def _expmap0_t(u, c, eps=1e-6):
    """expmap0 wrapper: Poincare 或 Lorentz 取决于 USE_LORENTZ_BALL.
    输出 dim 一致 (D-dim). Lorentz 内部 lift 到 D+1 计算后 drop x_0."""
    if USE_LORENTZ_BALL:
        u_lor = _euclid_to_lorentz_t(u, c, eps)
        x_lor = _expmap0_lorentz_t(u_lor[..., 1:], c, eps)  # u_lor[1:] 是 R^n 切空间向量
        return _lorentz_to_euclid_t(x_lor)
    sqrt_c = c ** 0.5
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(_eps(u))
    factor = torch.tanh(sqrt_c * norm_u) / (sqrt_c * norm_u)
    return _proj_to_ball_t(factor * u, c)


def _logmap0_t(x, c, eps=1e-6):
    """logmap0 wrapper: Poincare 或 Lorentz 取决于 USE_LORENTZ_BALL.
    输入 D-dim, 输出 D-dim. Lorentz 内部 lift x 到 D+1 计算后取 spatial 部分."""
    if USE_LORENTZ_BALL:
        x_lor = _euclid_to_lorentz_t(x, c, eps)
        v_lor = _logmap0_lorentz_t(x_lor, c, eps)
        return v_lor  # 已经是 R^n 切空间向量, dim = D
    sqrt_c = c ** 0.5
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
    factor = torch.atanh((sqrt_c * norm_x).clamp(max=1 - 1e-5)) / (sqrt_c * norm_x)
    return factor * x


def _transport_between_t(x, c_from, c_next, eps=1e-6):
    """M3 fused: 跨层曲率传输 Exp_0^{c2}(Log_0^{c1}(x))."""
    if USE_LORENTZ_BALL:
        x_lor = _euclid_to_lorentz_t(x, c_from, eps)
        out_lor = _transport_between_lorentz_t(x_lor, c_from, c_next, eps)
        return _lorentz_to_euclid_t(out_lor)
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
    USE_LORENTZ_BALL=True 时返回 Lorentz 距离 (arccosh): d_L(x, y) = (1/√c) arccosh(-c ⟨x_lor, y_lor⟩_L)
    """
    if USE_LORENTZ_BALL:
        x_lor = _euclid_to_lorentz_t(x, c, eps)
        y_lor = _euclid_to_lorentz_t(y, c, eps)
        return _lorentz_distance_t(x_lor, y_lor, c, eps)
    neg_x = -x
    u = _mobius_add_t(neg_x, y, c)
    norm_u = u.norm(dim=-1, keepdim=True).clamp(max=1 - 1e-5)
    return (2.0 / (c ** 0.5)) * torch.atanh(norm_u)


def _mobius_add_t(x, y, c):
    """tensor-safe Möbius addition (c broadcastable, 用于 per-item 内在 residual 减法).
    USE_LORENTZ_BALL=True 时返回 Lorentz tangent-space 加法: exp_o(log_o(x) + log_o(y))."""
    if USE_LORENTZ_BALL:
        u_x = _logmap0_t(x, c)
        u_y = _logmap0_t(y, c)
        return _expmap0_t(u_x + u_y, c)
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

