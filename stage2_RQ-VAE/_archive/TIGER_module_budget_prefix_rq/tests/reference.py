"""Independent Lorentz/spherical and probability-domain assignment references.

No production imports. Geometry uses ambient inner products rather than the
production radial/angular identity; assignment uses the frozen probability
iteration rather than the production log-domain iteration.
"""
import math
import torch


def lorentz_squared(x, y, curvature):
    """Independent hyperboloid exponential and Lorentz-product reference."""
    k = math.sqrt(-curvature)
    rx = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    ry = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    tx, ty = torch.cosh(k * rx), torch.cosh(k * ry)
    sx = torch.sinh(k * rx) * x / rx.clamp_min(1e-30)
    sy = torch.sinh(k * ry) * y / ry.clamp_min(1e-30)
    inner = tx @ ty.T - sx @ sy.T
    return torch.acosh(inner.clamp_min(1)).square() / (-curvature)


def sphere_squared(x, y, curvature):
    """Independent ambient exponential, dot product and acos reference."""
    k = math.sqrt(curvature)
    rx = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    ry = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    ax = torch.cat((torch.cos(k * rx), torch.sin(k * rx) * x / rx.clamp_min(1e-30)), -1)
    ay = torch.cat((torch.cos(k * ry), torch.sin(k * ry) * y / ry.clamp_min(1e-30)), -1)
    return torch.acos((ax @ ay.T).clamp(-1, 1)).square() / curvature


def frozen_center(cost):
    middle = (cost.max() + cost.min()) / 2
    return (cost - middle) / (cost.max() - middle + 1e-5)


def frozen_sinkhorn(cost, epsilon, iterations):
    """Unmodified probability-domain frozen TIGER arithmetic, float64."""
    q = torch.exp(-frozen_center(cost).double() / epsilon)
    b, j = q.shape
    q = q / q.sum()
    for _ in range(iterations):
        q = q / q.sum(0, keepdim=True) / j
        q = q / q.sum(1, keepdim=True) / b
    return q * b
