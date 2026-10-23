"""The verifier's own parallel transport (issue #112, P1/P2).

Independent of any candidate: this is the reference implementation of parallel transport in the
Poincare ball of curvature -c, and the fixture that decides whether a candidate's transport is
the real thing.

Two things this module exists to pin down, because both were got wrong in a candidate's probe:

1. **A tangent vector must be transported FROM the point it lives at.** Parallel transport
   ``PT_{x->y}`` takes a vector in ``T_x``. A residual computed as ``log0(x) - log0(a)`` lives in
   ``T_0`` (the origin's tangent space), so transporting it "from x" is a category error -- and it
   fails the isometry check, which is exactly how the mistake was caught.

2. **The scaling convention.** With ``g_x = lambda(x)^2 I`` and ``lambda(x) = 2/(1 - c||x||^2)``,
   the Riemannian norm of a tangent vector is ``lambda(x) * ||v||``, and the transport scales by
   ``lambda(x)/lambda(y)``.

The properties that are checked: the isometry, the round trip, the identity at the origin, and the
curvature limit (at a fixed NORMALISED radius the transport's deviation from the identity is
independent of c -- the curvature is a rescaling, finding F1).
"""
from __future__ import annotations

import math

import torch

FLOOR = 1e-30


def conformal(x: torch.Tensor, c: float) -> torch.Tensor:
    """``lambda(x) = 2 / (1 - c||x||^2)``: the scale of the metric ``g_x = lambda^2 I``."""
    return 2.0 / (1.0 - c * (x * x).sum(-1)).clamp_min(FLOOR)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    return numerator / denominator.clamp_min(FLOOR)


def gyration(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """``gyr[a, b] v``: the rotation of the tangent space induced by the Mobius isometry."""
    return mobius_add(-mobius_add(a, b, c), mobius_add(a, mobius_add(b, v, c), c), c)


def parallel_transport(v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """``PT_{x->y}(v)`` for a tangent vector ``v`` at ``x``, along the unique geodesic.

    ``v`` must live in ``T_x``. Passing a vector that lives in ``T_0`` while ``x`` is not the
    origin is a misuse, and the isometry property is what detects it.
    """
    lam_x = conformal(x, c)
    lam_y = conformal(y, c)
    moved = gyration(y, -x, v, c)
    return (lam_x / lam_y).unsqueeze(-1) * moved


def transport_from_origin(v: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Express an ORIGIN-frame tangent vector in the frame at ``y``."""
    origin = torch.zeros_like(y)
    return parallel_transport(v, origin, y, c)


def riemannian_norm(v: torch.Tensor, point: torch.Tensor, c: float) -> torch.Tensor:
    """The norm of a tangent vector in the metric at ``point``: ``lambda(point) * ||v||``."""
    return conformal(point, c) * torch.linalg.vector_norm(v, dim=-1)


def expmap0(x: torch.Tensor, c: float) -> torch.Tensor:
    """The exponential map at the origin: a tangent vector -> the ball of curvature -c.

    Its image always satisfies ``c||y||^2 < 1``, so the ball constraint is structural rather
    than something a clamp has to maintain.
    """
    root = math.sqrt(c)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(root * norm) / (root * norm) * x


def logmap0(x: torch.Tensor, c: float) -> torch.Tensor:
    """The exponential map at the origin, inverted."""
    root = math.sqrt(c)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.atanh((root * norm).clamp(max=1.0 - 1e-12)) / (root * norm) * x


__all__ = [
    "conformal",
    "expmap0",
    "gyration",
    "logmap0",
    "mobius_add",
    "parallel_transport",
    "riemannian_norm",
    "transport_from_origin",
]
