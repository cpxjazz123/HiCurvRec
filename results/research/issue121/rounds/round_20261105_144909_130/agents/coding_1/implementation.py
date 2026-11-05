import math

import torch

EPS = 1e-12
_ATANH_MAX = 1.0 - 1e-6


def _norm(v):
    return torch.sqrt(torch.clamp((v * v).sum(-1), min=0.0))


def _as_float_tensor(v):
    if not torch.is_tensor(v):
        v = torch.as_tensor(v, dtype=torch.get_default_dtype())
    elif not torch.is_floating_point(v):
        v = v.to(torch.get_default_dtype())
    return v


def lift(v, c):
    v = _as_float_tensor(v)
    c = float(c)
    if c == 0.0:
        return v
    sc = math.sqrt(c)
    n = _norm(v)
    scale = torch.tanh(sc * n) / (sc * n + EPS)
    return v * scale.unsqueeze(-1)


def mobius(x, y, c):
    x = _as_float_tensor(x)
    y = _as_float_tensor(y)
    c = float(c)
    xy = (x * y).sum(-1, keepdim=True)
    xn = (x * x).sum(-1, keepdim=True)
    yn = (y * y).sum(-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * yn) * x + (1.0 - c * xn) * y
    den = 1.0 + 2.0 * c * xy + c * c * xn * yn + EPS
    return num / den


def distance(x, y, curvature):
    x = _as_float_tensor(x)
    y = _as_float_tensor(y)
    c = float(curvature)
    if c == 0.0:
        return 2.0 * _norm(x - y)
    sc = math.sqrt(c)
    arg = torch.clamp(sc * _norm(mobius(-x, y, c)), max=_ATANH_MAX)
    return (2.0 / sc) * torch.atanh(arg)