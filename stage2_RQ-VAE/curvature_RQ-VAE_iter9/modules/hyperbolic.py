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
