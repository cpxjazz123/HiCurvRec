import math

import torch

EPS = 1e-12

# FROZEN: the c = 1 ball radius, never recomputed (not needed by the distance
# itself, which carries its own 2/sqrt(c) factor).
R_C = 1.0


def _norm(v):
    """Euclidean coordinate norm over the last axis, as the specification defines it."""
    return torch.sqrt(torch.clamp((v * v).sum(-1), min=0.0))


def _as_curvature(curvature):
    """The declared curvature argument as a plain float (it may arrive as a tensor scalar)."""
    if isinstance(curvature, torch.Tensor):
        return float(curvature.detach().reshape(-1)[0].item())
    return float(curvature)


def lift(v, curvature=1.0):
    """tanh(sqrt(c)||v||) v / (sqrt(c)||v|| + eps); the identity at c = 0, 0 at v = 0."""
    c = _as_curvature(curvature)
    if c == 0.0:
        return v
    s = math.sqrt(c)
    n = _norm(v)
    return v * (torch.tanh(s * n) / (s * n + EPS)).unsqueeze(-1)


def mobius(x, y, curvature=1.0):
    """The canonical Mobius sum of the specification, in the listing's +2c<.,.> form.

    The distance passes X = -x, i.e. mobius(-x, y, c): that is the negated form the
    specification requires (1 - 2c<x,y> + c||y||^2 on X = -x). At c = 0 it degenerates
    to x + y.
    """
    c = _as_curvature(curvature)
    xy = (x * y).sum(-1, keepdim=True)
    xn = (x * x).sum(-1, keepdim=True)
    yn = (y * y).sum(-1, keepdim=True)
    if c == 0.0:
        return x + y
    num = (1.0 + 2.0 * c * xy + c * yn) * x + (1.0 - c * xn) * y
    den = 1.0 + 2.0 * c * xy + c * c * xn * yn + EPS
    return num / den


def dist(x, y, curvature=1.0):
    """d_c(x, y) = (2/sqrt(c)) artanh(min(sqrt(c) ||(-x) (+) y||, 1 - 1e-6)); 2||x - y|| at c = 0."""
    c = _as_curvature(curvature)
    if c == 0.0:
        return 2.0 * _norm(x - y)
    s = math.sqrt(c)
    m = mobius(-x, y, c)
    return (2.0 / s) * torch.atanh(torch.clamp(s * _norm(m), max=1.0 - 1e-6))


def distance(x, y, curvature=1.0):
    """The specification's continuous surface: d = d_curvature(x, y).

    x, y: (..., D) tensors (the surface is exercised at radius 0.4, width 2);
    curvature: the declared scalar, c = 1.0 for the curved arm, 0.0 for GEOMETRY_OFF.
    Returns d of shape (...).
    """
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, dtype=torch.float32)
    if x.shape != y.shape:
        y = y.expand_as(x)
    return dist(x, y, curvature)

