import math

import torch

EPS = 1e-12


def _norm(v):
    return torch.sqrt(torch.clamp((v * v).sum(-1), min=0.0))


def lift(v, c):
    c = float(c)
    if c == 0.0:
        return v
    s = math.sqrt(c)
    n = _norm(v)
    return v * (torch.tanh(s * n) / (s * n + EPS)).unsqueeze(-1)


def mobius(x, y, c):
    c = float(c)
    xy = (x * y).sum(-1, keepdim=True)
    xx = (x * x).sum(-1, keepdim=True)
    yy = (y * y).sum(-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * yy) * x + (1.0 - c * xx) * y
    den = 1.0 + 2.0 * c * xy + c * c * xx * yy + EPS
    return num / den


def dist(x, y, c):
    c = float(c)
    if c == 0.0:
        return 2.0 * _norm(x - y)
    s = math.sqrt(c)
    arg = s * _norm(mobius(-x, y, c))
    return (2.0 / s) * torch.atanh(torch.clamp(arg, max=1.0 - 1e-6))


def distance(x, y, curvature):
    return dist(x, y, float(curvature))

