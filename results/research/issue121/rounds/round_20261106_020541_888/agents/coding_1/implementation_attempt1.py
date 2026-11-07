import math

import torch

EPS = 1e-12
C_FIXED = 1.0
R_C_FIXED = 1.0
ATANH_CLAMP = 1.0 - 1e-6


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
    num = (1.0 - 2.0 * c * xy + c * yy) * x + (1.0 - c * xx) * y
    den = 1.0 - 2.0 * c * xy + c * c * xx * yy + EPS
    return num / den


def distance(x, y, curvature):
    c = float(curvature)
    if c == 0.0:
        return _norm(x - y)
    s = math.sqrt(c)
    m = mobius(-x, y, c)
    mn = (m * m).sum(-1, keepdim=True)
    d = (2.0 / s) * torch.atanh(
        torch.clamp(s * torch.sqrt(torch.clamp(mn, min=0.0)),
                    min=0.0, max=ATANH_CLAMP))
    return d.squeeze(-1)


dist = distance


def stratum(alpha_k):
    return torch.sigmoid(alpha_k)


def init_alpha(K1, device=None, dtype=torch.float32):
    k = torch.arange(K1, device=device, dtype=dtype)
    return torch.logit((k + 0.5) / float(K1))


def cost_matrix(r, a, rho_i, s_k, beta, curvature=C_FIXED):
    c = float(curvature)
    R_c = R_C_FIXED
    u = lift(r, c)
    q_lift = lift(a, c)
    D_k = distance(u[:, None, :], q_lift[None, :, :], c)
    O_ik = beta * R_c * (1.0 - torch.abs(rho_i[:, None] - s_k[None, :]))
    return D_k, O_ik, D_k - O_ik


def beta_from_median(D_k, k1):
    d_chosen = D_k.gather(-1, k1[:, None]).squeeze(-1)
    return torch.median(d_chosen) / R_C_FIXED


def straight_through(a, P_ik, K1):
    k1 = torch.argmax(P_ik, dim=-1)
    p_i = torch.nn.functional.one_hot(k1, K1).to(P_ik.dtype)
    p_i = p_i + P_ik - P_ik.detach()
    return k1, p_i