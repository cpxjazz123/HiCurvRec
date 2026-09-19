"""当前 RQ-VAE 使用的 Poincaré 双曲几何运算。"""
import torch


def _eps(x, default=1e-6):
    return torch.finfo(x.dtype).eps if x.dtype.is_floating_point else default


def _proj_to_ball_t(x, c, eps=1e-6):
    radius = (1.0 / c) ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) * radius
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
    """M3: 通过原点在两种曲率之间传输切向量。"""
    radius = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    a = (c_from ** 0.5) * radius
    b = (c_next / c_from) ** 0.5
    a_safe = a.clamp(max=1 - 1e-5)
    factor = torch.tanh(b * torch.atanh(a_safe)) / (b * a)
    return factor * x


def _sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    """将距离矩阵转成 balanced assignment，防止 codebook collapse。"""
    Q = torch.exp(-distances / epsilon)
    batch_size = Q.shape[0]
    codebook_size = Q.shape[1]
    Q /= Q.sum().clamp_min(torch.finfo(Q.dtype).tiny)
    for _ in range(sinkhorn_iterations):
        Q /= Q.sum(dim=1, keepdim=True).clamp_min(torch.finfo(Q.dtype).tiny)
        Q /= batch_size
        Q /= Q.sum(dim=0, keepdim=True).clamp_min(torch.finfo(Q.dtype).tiny)
        Q /= codebook_size
    Q *= batch_size
    return Q


def _poincare_distance_t(x, y, c, eps=1e-6):
    """Poincaré 距离 d_c(x,y) = 2/√c · artanh(√c·||(-x)⊕_c y||)。"""
    difference = _mobius_add_t(-x, y, c)
    sqrt_c = c ** 0.5
    scaled_norm = (sqrt_c * difference.norm(dim=-1, keepdim=True)).clamp(
        max=1 - 1e-5
    )
    return (2.0 / sqrt_c) * torch.atanh(scaled_norm)


def _mobius_add_t(x, y, c):
    """支持曲率广播的 Möbius 加法。"""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return numerator / denominator.clamp_min(_eps(denominator))


# === iter5 v375: spherical manifold support for mixed-curvature product ===
def _sphere_distance_t(x, y, eps=1e-6):
    """球面距离: d(x,y) = arccos(<x,y>). 假设 x,y 已 L2-normalize 到 S^{d-1}."""
    inner = (x * y).sum(dim=-1).clamp(-1 + eps, 1 - eps)
    return torch.arccos(inner)


def _sphere_expmap0_t(v, eps=1e-6):
    """球面 exp_0(v) = v / ||v|| (规范化到 S^{d-1}). 距离 = ||v||, 方向 = v/||v||."""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return v / norm


def _sphere_logmap0_t(x, eps=1e-6):
    """球面 log_0(x) = arccos(<x,0_c>) * x / ||x||. 0_c 是球面任意点, 取 (1,0,...,0) 等价."""
    # 简化: 0_c 取 e_1 = (1, 0, ..., 0). 在球面上, log 0 映射到以 0_c 为原点的切空间.
    # 工程简化: 直接返回 x 本身, 因为下游只用 norm/方向, 球面量化的"残差"用 v 本身即可.
    return x
