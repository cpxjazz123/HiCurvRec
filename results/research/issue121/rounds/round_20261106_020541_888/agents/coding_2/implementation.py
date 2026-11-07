import math

import torch

EPS = 1e-12          # the specification's epsilon inside the Mobius denominator
R_C = 1.0            # FROZEN: the c = 1 ball radius, 1/sqrt(1), never recomputed
_ATANH_MAX = 1.0 - 1e-6   # the specification's min(..., 1 - 1e-6) clamp


def _norm(v):
    return torch.sqrt(torch.clamp((v * v).sum(-1), min=0.0))


def _as_tensor(v):
    if not torch.is_tensor(v):
        return torch.as_tensor(v, dtype=torch.float64)
    if not v.is_floating_point():
        return v.to(torch.get_default_dtype())
    return v


def lift(v, c):
    # exp_0^c(v) = tanh(sqrt(c)||v||) v / (sqrt(c)||v|| + eps), LIFT(0) = 0, c = 0 -> identity
    c = float(c)
    if c == 0.0:
        return v
    sc = math.sqrt(c)
    n = _norm(v)
    return v * (torch.tanh(sc * n) / (sc * n + EPS)).unsqueeze(-1)


def mobius(x, y, c):
    # x (+) y = ((1 + 2c<x,y> + c||y||^2) x + (1 - c||x||^2) y)
    #           / (1 + 2c<x,y> + c^2||x||^2||y||^2 + eps)
    # distance() calls this with x -> -x, which negates the 2c<.,.> terms.
    c = float(c)
    xy = (x * y).sum(-1, keepdim=True)
    xx = (x * x).sum(-1, keepdim=True)
    yy = (y * y).sum(-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * yy) * x + (1.0 - c * xx) * y
    den = 1.0 + 2.0 * c * xy + c * c * xx * yy + EPS
    return num / den


def distance(x, y, curvature=1.0):
    # d_c(x, y) = (2/sqrt(c)) artanh(min(sqrt(c) ||(-x)(+)y||, 1 - 1e-6)); d_0 = 2||x - y||
    x = _as_tensor(x)
    y = _as_tensor(y)
    c = float(curvature)
    if c == 0.0:
        return 2.0 * _norm(x - y)
    sc = math.sqrt(c)
    arg = torch.clamp(sc * _norm(mobius(-x, y, c)), max=_ATANH_MAX)
    return (2.0 / sc) * torch.atanh(arg)