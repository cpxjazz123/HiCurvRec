"""The prefix-conditioned curvature metric of issue #177.

``d_kappa(u, v) = (1/sqrt(k)) * arccosh(1 + 2k||u-v||^2 / ((1 - k||u||^2)(1 - k||v||^2)))`` on the ball of curvature -k. The
alternative closed form at the origin, ``d_kappa(0, v) = (2/sqrt(k)) artanh(sqrt(k)||v||)``, is what the tests check against,
and ``GEOMETRY_OFF`` replaces the whole thing with the Euclidean norm so the frozen baseline can be reproduced bit for bit.
"""
from __future__ import annotations

import torch

#: Switch of the issue: with it on the Euclidean path is used and the frozen baseline's arithmetic is reproduced exactly.
GEOMETRY_OFF = False
#: Switch of the issue: with it on the per-prefix kappa table is randomly permuted, keeping the distribution.
BEHAVIOUR_OFF = False

#: Numerical floor and ball guard, both fixed by the issue rather than tuned.
KAPPA_FLOOR = 1e-3
BALL_EPS = 1e-7


def distance(u: torch.Tensor, v: torch.Tensor, kappa, curvature: float = 1.0) -> torch.Tensor:
    """``d_kappa`` for one or many rows; ``kappa`` is a scalar or a ``[B, 1]`` column."""
    if GEOMETRY_OFF:
        # c = 0 limit in the frozen convention: 2 * ||u - v|| (see poincare_contract.distance)
        return 2.0 * (u - v).norm(dim=-1)
    k = torch.as_tensor(kappa, dtype=u.dtype, device=u.device)
    if k.dim() == 0:
        k = k.expand(1)
    k = k.reshape(-1, 1).clamp_min(KAPPA_FLOOR)
    u2 = (u.pow(2).sum(-1, keepdim=True) * k).clamp(max=1.0 - BALL_EPS)
    v2 = (v.pow(2).sum(-1, keepdim=True) * k).clamp(max=1.0 - BALL_EPS)
    diff2 = (u - v).pow(2).sum(-1, keepdim=True)
    argument = 1.0 + 2.0 * k * diff2 / ((1.0 - u2) * (1.0 - v2))
    return (torch.acosh(argument.clamp_min(1.0 + 1e-12)) / torch.sqrt(k)).squeeze(-1)


def at_origin(v: torch.Tensor, kappa, ) -> torch.Tensor:
    """The closed form ``(2/sqrt(k)) artanh(sqrt(k)||v||)`` used to validate :func:`distance`."""
    k = torch.as_tensor(kappa, dtype=v.dtype, device=v.device)
    if k.dim() == 0:
        k = k.expand(1)
    k = k.reshape(-1, 1).clamp_min(KAPPA_FLOOR)
    root = torch.sqrt(k)
    norm = v.norm(dim=-1, keepdim=True)
    return ((2.0 / root) * torch.atanh((root * norm).clamp(max=1.0 - BALL_EPS))).squeeze(-1)


def prefix_kappa(table: dict, level: int, prefix) -> float:
    """The kappa a prefix carries at ``level``; level 1 has no prefix and takes the frozen 1.0."""
    if level == 1:
        return float(table["level_1"]["kappa"])
    block = table[f"level_{level}"]["rows"]
    key = str(tuple(prefix))
    return float(block.get(key, table["mean_frozen"]))
