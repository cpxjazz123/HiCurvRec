"""Three fixed-curvature distances in one common, scaled tangent space.

Poincare exp at the origin uses tanh(sqrt(-c)*||u||/2), not
`tanh(sqrt(-c)*||u||)`: tangent lengths then agree with the Euclidean and
spherical experts. Hyperbolic distances evaluate the Poincare asinh formula
in radial/angular log coordinates, so saturated tanh never loses radial
separation. No expert owns a fitted scale or additional codebook coordinates.
"""
import math

import torch


CURVATURES = (-1, 0, 1)


def _pairs(x, y, scale):
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Geometry inputs must have the same tangent dimension.")
    if torch.is_tensor(scale) and scale.is_cuda:
        torch._assert_async(torch.isfinite(scale) & (scale > 0),
                            "The common tangent scale must be finite and positive.")
    elif not math.isfinite(scale) or not bool(scale > 0):
        raise ValueError("The common tangent scale must be finite and positive.")
    return x / scale, y / scale


def _pair_norm(x, y, paired=False):
    if x.ndim == 2 and y.ndim == 2 and not paired:
        # The direct kernel accumulates coordinate differences without an
        # [items,codes,dimension] temporary or Gram-matrix cancellation.
        dtype = torch.promote_types(x.dtype, y.dtype)
        kernel_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        return torch.cdist(x.to(kernel_dtype), y.to(kernel_dtype),
                           compute_mode="donot_use_mm_for_euclid_dist").to(dtype)
    return torch.linalg.vector_norm(x - y, dim=-1)


def _log_sinh(nonnegative):
    # Finite arguments even for exact zero; callers mask zero terms before
    # logaddexp, avoiding both log(0) backward and tanh-boundary clipping.
    safe = nonnegative.clamp_min(torch.finfo(nonnegative.dtype).tiny)
    return safe + torch.log(-torch.expm1(-2 * safe)) - math.log(2)


def _hyperbolic_squared(rx, ry, angular, curvature):
    k = math.sqrt(-curvature)
    a, b = k * rx, k * ry
    eps = torch.finfo(rx.dtype).tiny
    radial = (a - b).abs() / 2
    absent = -torch.finfo(rx.dtype).max
    radial_log = torch.where(radial > 0, 2 * _log_sinh(radial), absent)
    angular_log = _log_sinh(a) + _log_sinh(b) + torch.log(angular.clamp_min(eps))
    angular_log = torch.where((angular > 0) & (rx > 0) & (ry > 0), angular_log, absent)
    # Poincare identity:
    # sinh(k*d/2)^2 = sinh((a-b)/2)^2
    #                    + sinh(a)*sinh(b)*||direction_x-direction_y||^2/4.
    # Finite logaddexp identity avoids the compiler's saved-predicate lowering.
    maximum = torch.maximum(radial_log, angular_log)
    z = (maximum + torch.log1p(torch.exp(-(radial_log - angular_log).abs()))) / 2
    positive = z.clamp_min(0)
    asinh_exp = torch.where(z > 0,
                            positive + torch.log1p(torch.sqrt(1 + torch.exp(-2 * positive))),
                            torch.asinh(torch.exp(z.clamp_max(0))))
    return (2 * asinh_exp / k).square()


def _sphere_squared(x, y, rx, ry, curvature, paired=False):
    k = math.sqrt(curvature)
    ax = (k * rx).unsqueeze(-1)
    ay = (k * ry).unsqueeze(-1)
    # torch.sinc is sin(pi*t)/(pi*t), including its analytic value at zero.
    sx = torch.cat((torch.cos(ax), k * torch.sinc(ax / math.pi) * x), dim=-1)
    sy = torch.cat((torch.cos(ay), k * torch.sinc(ay / math.pi) * y), dim=-1)
    # Chord-angle atan2 remains stable at coincidence and near antipodes,
    # unlike acos(dot), and uses ambient coordinates only for the metric.
    difference = _pair_norm(sx, sy, paired)
    total = _pair_norm(sx, -sy, paired)
    return (2 * torch.atan2(difference, total) / k).square()


def _expert_scaled(x, y, curvature_on, paired=False):
    """Shared radial/directional arithmetic for exactly the three experts."""
    matrix = x.ndim == 2 and y.ndim == 2 and not paired
    euclidean = _pair_norm(x, y).square() if matrix else (x - y).square().sum(-1)
    if not curvature_on:
        finite = torch.isfinite(x).all() & torch.isfinite(y).all()
        if x.is_cuda:
            torch._assert_async(finite, "Geometry tangents must have finite norms.")
        elif not bool(finite):
            raise ValueError("Geometry tangents must have finite norms.")
        return euclidean.unsqueeze(-1).expand(*euclidean.shape, 3)
    rx = torch.linalg.vector_norm(x, dim=-1)
    ry = torch.linalg.vector_norm(y, dim=-1)
    finite = torch.isfinite(rx).all() & torch.isfinite(ry).all()
    if x.is_cuda:
        torch._assert_async(finite, "Geometry tangents must have finite norms.")
    elif not bool(finite):
        raise ValueError("Geometry tangents must have finite norms.")
    dx = x / rx.clamp_min(torch.finfo(x.dtype).tiny).unsqueeze(-1)
    dy = y / ry.clamp_min(torch.finfo(y.dtype).tiny).unsqueeze(-1)
    angular = _pair_norm(dx, dy, paired).square() / 4
    hx, hy = (rx[:, None], ry[None, :]) if matrix else (rx, ry)
    return torch.stack((_hyperbolic_squared(hx, hy, angular, -1), euclidean,
                        _sphere_squared(x, y, rx, ry, 1, paired)), dim=-1)


_CUDA_MATH = None


def _cuda_functions():
    global _CUDA_MATH
    if _CUDA_MATH is None:
        from tools.issue253.cuda_math import compile_geometry
        _CUDA_MATH = compile_geometry(_expert_scaled, _mixed_scaled, _fixed_scaled)
    return _CUDA_MATH


def _flat_pair(x, y):
    """Unit axes encode rowwise pairs, not a pairwise matrix reduction."""
    paired = x.ndim == y.ndim == 3 and x.shape[-2] == y.shape[-2] == 1
    return (x.squeeze(-2), y.squeeze(-2), True) if paired else (x, y, False)


def expert_squared(x, y, scale=1.0, curvature_on=True):
    """Pairwise [...,3] squared distances in the same dtype/common tangent."""
    x, y = _pairs(x, y, scale)
    if x.is_cuda and x.dtype == y.dtype == torch.float32:
        x, y, paired = _flat_pair(x, y)
        value = _cuda_functions()[0](x, y, curvature_on, paired)
        return value.unsqueeze(-2) if paired else value
    return _expert_scaled(x, y, curvature_on)


def _mixed_scaled(x, y, weights, curvature_on):
    distances = _expert_scaled(x, y, curvature_on)
    if weights.ndim == distances.ndim - 1:
        weights = weights.unsqueeze(-2)
    return (distances * weights).sum(-1)


def mixed_cost(x, y, weights, scale=1.0, curvature_on=True):
    """Same live item-weighted expert cost; no independent metric-only gate."""
    x, y = _pairs(x, y, scale)
    if x.is_cuda and x.dtype == y.dtype == torch.float32:
        x, y, paired = _flat_pair(x, y)
        if paired:
            distances = _cuda_functions()[0](x, y, curvature_on, True).unsqueeze(-2)
            if weights.ndim == distances.ndim - 1:
                weights = weights.unsqueeze(-2)
            return (distances * weights).sum(-1)
        return _cuda_functions()[1](x, y, weights, curvature_on)
    return _mixed_scaled(x, y, weights, curvature_on)


def _fixed_scaled(x, y, paired=False):
    return _expert_scaled(x, y, True, paired).mean(-1)


def fixed_squared(x, y, scale=1.0):
    """Gate-independent symmetric three-expert mean for loss and evaluation."""
    x, y = _pairs(x, y, scale)
    if x.is_cuda and x.dtype == y.dtype == torch.float32:
        x, y, paired = _flat_pair(x, y)
        value = _cuda_functions()[2](x, y, paired)
        return value.unsqueeze(-1) if paired else value
    return _fixed_scaled(x, y)
