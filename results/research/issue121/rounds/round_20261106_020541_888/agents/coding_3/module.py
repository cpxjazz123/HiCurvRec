import math

import torch

EPS = 1e-12
R_C = 1.0  # FROZEN: the c = 1 ball radius, never recomputed


def _norm(v):
    """Euclidean coordinate norm along the last axis: sqrt(clamp(sum v^2, min=0))."""
    return torch.sqrt(torch.clamp((v * v).sum(-1), min=0.0))


def lift(v, c):
    """exp_0^c: v -> tanh(sqrt(c)||v||) v / (sqrt(c)||v|| + eps); identity at c == 0; 0 -> 0."""
    c = float(c)
    if c == 0.0:
        return v
    n = _norm(v)
    scale = torch.tanh(math.sqrt(c) * n) / (math.sqrt(c) * n + EPS)
    return v * scale.unsqueeze(-1)


def mobius(x, y, c):
    """Canonical Mobius sum with the +2c<.,.> convention; distance calls it as mobius(-x, y, c)."""
    c = float(c)
    xy = (x * y).sum(-1, keepdim=True)
    xn = (x * x).sum(-1, keepdim=True)
    yn = (y * y).sum(-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * yn) * x + (1.0 - c * xn) * y
    den = 1.0 + 2.0 * c * xy + c * c * xn * yn + EPS
    return num / den


def distance(x, y, curvature):
    """d_c(x, y) = (2/sqrt(c)) artanh(min(sqrt(c)||(-x) (+) y||, 1 - 1e-6)); 2||x - y|| at c = 0.

    x, y: (..., D) Euclidean coordinates in the ball (c||x||^2 < 1, c||y||^2 < 1).
    curvature: the scalar c (the specification's fixed c = 1.0); it carries no gradient.
    returns: (...,) the distance of every pair, in ball units (R_c = 1/sqrt(c)).
    """
    c = float(curvature)

    dtype = x.dtype if torch.is_tensor(x) else torch.float32
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=dtype)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y, dtype=dtype)
    if y.dtype != x.dtype:
        y = y.to(x.dtype)

    if c == 0.0:
        # LIFT is the identity and the c -> 0 limit is 2||x - y||.
        return 2.0 * _norm(x - y)

    root_c = math.sqrt(c)
    mx = mobius(-x, y, c)
    return (2.0 / root_c) * torch.atanh(
        torch.clamp(root_c * _norm(mx), max=1.0 - 1e-6)
    )


# the specification's declared continuous-surface name
compute = distance

