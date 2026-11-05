import math

import torch

EPS = 1e-12
ATANH_MAX = 1.0 - 1e-6


def _norm(v):
    return torch.sqrt(torch.clamp((v * v).sum(-1), min=0.0))


def _mobius(x, y, c):
    xy = (x * y).sum(-1, keepdim=True)
    xn = (x * x).sum(-1, keepdim=True)
    yn = (y * y).sum(-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * yn) * x + (1.0 - c * xn) * y
    den = 1.0 + 2.0 * c * xy + (c * c) * xn * yn + EPS
    return num / den


def distance(x, y, curvature):
    x_t = torch.as_tensor(x)
    y_t = torch.as_tensor(y)
    out_dtype = torch.promote_types(x_t.dtype, y_t.dtype)
    if not out_dtype.is_floating_point:
        out_dtype = torch.get_default_dtype()
    xd = x_t.to(torch.float64)
    yd = y_t.to(torch.float64)
    if (xd.dim() == yd.dim() and xd.dim() >= 2
            and xd.shape[-1] == yd.shape[-1]
            and tuple(xd.shape[:-1]) != tuple(yd.shape[:-1])
            and all(s != 1 for s in xd.shape[:-1])
            and all(s != 1 for s in yd.shape[:-1])):
        xd = xd.unsqueeze(-2)
        yd = yd.unsqueeze(-3)
    if torch.is_tensor(curvature):
        c = float(curvature.detach().reshape(-1)[0].item()) if curvature.numel() else 0.0
    else:
        c = float(curvature)
    if c == 0.0:
        d = 2.0 * _norm(xd - yd)
    else:
        sc = math.sqrt(c)
        d = (2.0 / sc) * torch.atanh(torch.clamp(sc * _norm(_mobius(-xd, yd, c)), max=ATANH_MAX))
    return d.to(out_dtype)