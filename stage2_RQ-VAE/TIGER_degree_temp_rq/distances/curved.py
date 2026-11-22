"""Squared geodesic distance on the constant-curvature ball, as the assignment cost.

Issue #236 measured the item graph's sectional curvature at -1.02, so this candidate asks
the only question issue #234 never asked: what if the *assignment cost* is the geodesic
distance of that measured curvature, instead of the Euclidean one?  The mechanism enters
the Sinkhorn cost matrix -- the one place CLAUDE.md 19h allows a mechanism to live -- and
touches neither the input embedding nor the loss.

Definition (kappa-stereographic ball of curvature KAPPA, radius 1/sqrt|KAPPA|):

    u = tanh(r / tau) * x / r          r = |x|,  tau = mean codeword norm
    d(u, v)^2 = ( arcosh(1 + 2|u-v|^2 / ((1-|u|^2)(1-|v|^2))) / sqrt|KAPPA| )^2

`tau` is not a tuned hyper-parameter: it is the mean codeword norm at the current step, so
the map is scale-equivariant and the ball is filled by construction.  KAPPA is the measured
value, not a search variable.

Why this can help: on a negatively curved ball, distances grow faster for points near the
boundary, so the assignment penalises splitting a tight cluster across codewords more than
the Euclidean metric does, while Sinkhorn keeps the assignment balanced (the balancing rule
is the baseline's, per 19h).
"""
from __future__ import annotations

import torch

NAME = "curved"
DESCRIPTION = ("squared geodesic distance on the curvature-(-1) stereographic ball; "
               "the assignment cost of issue #236's measured curvature")

#: Measured on the frozen behaviour graph (#236): geodesic mean -1.0188. Frozen, not tuned.
KAPPA = -1.0
EPS = 1e-6
#: Hard cap on the mapped radius.  Without it, points near the boundary make
#: `(1-|u|^2)` small enough that the float error of the expanded `|u-v|^2` is amplified
#: past the geometric signal (measured: a self-distance of 14 instead of 0).
RADIUS_CAP = 0.9


def _to_ball(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Radial map `r -> tanh(r/tau)` that keeps direction and bounds the radius by 1."""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return RADIUS_CAP * torch.tanh(norm / tau) * x / norm


def sq_geodesic(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """`(B, K)` squared geodesic distance with curvature `KAPPA`."""
    root = abs(KAPPA) ** 0.5
    u2 = (u * u).sum(-1, keepdim=True).clamp(max=RADIUS_CAP ** 2)
    v2 = (v * v).sum(-1).clamp(max=RADIUS_CAP ** 2)
    diff2 = (
        u.pow(2).sum(-1, keepdim=True)
        - 2 * u @ v.t()
        + v.pow(2).sum(-1, keepdim=True).t()
    ).clamp_min(0.0)
    inner = 1.0 + 2.0 * diff2 / ((1.0 - u2) * (1.0 - v2).unsqueeze(0))
    geodesic = torch.acosh(inner.clamp_min(1.0 + EPS)) / root
    return geodesic.pow(2)


def distances(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    tau = codebook.norm(dim=-1).mean().clamp_min(EPS)
    return sq_geodesic(_to_ball(latent, tau), _to_ball(codebook, tau))
