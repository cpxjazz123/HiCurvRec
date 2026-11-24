"""Five fixed-curvature distances in one common, scaled tangent space.

Poincare exp at the origin uses tanh(sqrt(-c)*||u||/2), not
`tanh(sqrt(-c)*||u||)`: tangent lengths then agree with the Euclidean and
spherical experts. Hyperbolic distances evaluate the Poincare asinh formula
in radial/angular log coordinates, so saturated tanh never loses radial
separation. No expert owns a fitted scale or additional codebook coordinates.
"""
import math

import torch


CURVATURES = (-3, -1, 0, 1, 3)


def _pairs(x, y, scale):
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Geometry inputs must have the same tangent dimension.")
    if not bool(scale > 0):
        raise ValueError("The common tangent scale must be positive.")
    x, y = x / scale, y / scale
    if x.ndim == 2 and y.ndim == 2:
        return x[:, None, :], y[None, :, :]
    return x, y


def _log_sinh(nonnegative):
    # Finite arguments even for exact zero; callers mask zero terms before
    # logaddexp, avoiding both log(0) backward and tanh-boundary clipping.
    safe = nonnegative.clamp_min(torch.finfo(nonnegative.dtype).tiny)
    return safe + torch.log(-torch.expm1(-2 * safe)) - math.log(2)


def _hyperbolic_squared(x, y, curvature):
    k = math.sqrt(-curvature)
    rx = torch.linalg.vector_norm(x, dim=-1)
    ry = torch.linalg.vector_norm(y, dim=-1)
    a, b = k * rx, k * ry
    eps = torch.finfo(x.dtype).tiny
    dx = x / rx.clamp_min(eps).unsqueeze(-1)
    dy = y / ry.clamp_min(eps).unsqueeze(-1)
    angular = (dx - dy).square().sum(-1) / 4
    radial = (a - b).abs() / 2
    absent = -torch.finfo(x.dtype).max
    radial_log = torch.where(radial > 0, 2 * _log_sinh(radial), absent)
    angular_log = _log_sinh(a) + _log_sinh(b) + torch.log(angular.clamp_min(eps))
    angular_log = torch.where((angular > 0) & (rx > 0) & (ry > 0), angular_log, absent)
    # Poincare identity:
    # sinh(k*d/2)^2 = sinh((a-b)/2)^2
    #                    + sinh(a)*sinh(b)*||direction_x-direction_y||^2/4.
    z = torch.logaddexp(radial_log, angular_log) / 2
    positive = z.clamp_min(0)
    asinh_exp = torch.where(z > 0,
                            positive + torch.log1p(torch.sqrt(1 + torch.exp(-2 * positive))),
                            torch.asinh(torch.exp(z.clamp_max(0))))
    return (2 * asinh_exp / k).square()


def _sphere_squared(x, y, curvature):
    k = math.sqrt(curvature)
    ax = k * torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    ay = k * torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    # torch.sinc is sin(pi*t)/(pi*t), including its analytic value at zero.
    sx = torch.cat((torch.cos(ax), k * torch.sinc(ax / math.pi) * x), dim=-1)
    sy = torch.cat((torch.cos(ay), k * torch.sinc(ay / math.pi) * y), dim=-1)
    # Chord-angle atan2 remains stable at coincidence and near antipodes,
    # unlike acos(dot), and uses ambient coordinates only for the metric.
    difference = torch.linalg.vector_norm(sx - sy, dim=-1)
    total = torch.linalg.vector_norm(sx + sy, dim=-1)
    return (2 * torch.atan2(difference, total) / k).square()


def expert_squared(x, y, scale=1.0, curvature_on=True):
    """Return pairwise squared distances [...,5], preserving input dtype.

    Matrix inputs [B,D],[J,D] produce [B,J,5]; pre-broadcast inputs are also
    accepted. Production uses D=32; arbitrary D enables tiny reference tests.
    With curvature disabled every expert is the same Euclidean computation.
    """
    x, y = _pairs(x, y, scale)
    euclidean = (x - y).square().sum(-1)
    if not curvature_on:
        return euclidean.unsqueeze(-1).expand(*euclidean.shape, 5)
    return torch.stack((_hyperbolic_squared(x, y, -3),
                        _hyperbolic_squared(x, y, -1), euclidean,
                        _sphere_squared(x, y, 1),
                        _sphere_squared(x, y, 3)), dim=-1)


def mixed_cost(x, y, weights, scale=1.0, curvature_on=True):
    """Item-weighted five-expert assignment costs, normally [B,J].

    Behavior-off is represented by the caller's fixed uniform weights; there
    is deliberately no second learned or metric-only gate in this module.
    """
    distances = expert_squared(x, y, scale=scale, curvature_on=curvature_on)
    if weights.ndim == distances.ndim - 1:
        weights = weights.unsqueeze(-2)
    return (distances * weights).sum(-1)


def fixed_squared(x, y, scale=1.0):
    """Gate-independent symmetric five-expert mean for loss and evaluation."""
    return expert_squared(x, y, scale=scale).mean(-1)
