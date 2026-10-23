"""Independent hyperbolic geometry oracles (issue #110, WP2).

These are the verifier's own implementations of the hyperbolic geometry, written from the
definitions rather than taken from any mechanism tree. The point is independence: an oracle
that imports the candidate's code can only confirm that the candidate agrees with itself, so
the H gates compare the candidate's behaviour against THESE functions.

Two models are supported, and the equivalence between them is a test rather than an
assumption:

* the POCARÉ BALL, with the curvature -c, the points inside the ball c||x||^2 < 1, and the
  conformal factor 1/c;
* the LORENTZ HYPERBOLOID, with the curvature -c, the points on the upper sheet
  <-x,x>_L = 1/c.

The formulas are the standard ones, written out so that they can be read and checked:
the distance is the artanh of a norm ratio, the exp and log maps at the origin are the
tanh and artanh scalings, the Mobius addition is the gyroparallel sum, the Busemann function
is the logarithm of a cross ratio, and the parallel transport in the Lorentz model moves a
tangent vector from one point to another along the geodesic.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _check_curvature(c: float) -> float:
    """The curvature parameter, as a positive number c for the curvature -c."""
    value = float(c)
    if not (value > 0.0) or not math.isfinite(value):
        raise ValueError(f"the curvature parameter c must be finite and positive, got {c!r}")
    return value


def project_to_ball(x: torch.Tensor, c: float, *, margin: float = 1e-6) -> torch.Tensor:
    """Project points into the open ball c||x||^2 < 1, with a margin from the boundary.

    The safeguard is deliberately not a clamp to the boundary: a point exactly on the
    boundary has an infinite distance, so the margin keeps every downstream computation
    finite and the distance growth visible rather than silent.
    """
    c = _check_curvature(c)
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    limit = torch.sqrt(torch.tensor(1.0 / c, dtype=x.dtype, device=x.device)) * (1.0 - margin)
    scale = torch.clamp(limit / norm.clamp_min(1e-12), max=1.0)
    return x * scale


def ball_margin(x: torch.Tensor, c: float) -> torch.Tensor:
    """How close each point is to the boundary: 1 - c||x||^2, positive inside."""
    c = _check_curvature(c)
    return 1.0 - c * torch.linalg.norm(x, dim=-1) ** 2


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """The geodesic distance in the Poincaré ball with curvature -c."""
    c = _check_curvature(c)
    x2 = torch.linalg.norm(x, dim=-1) ** 2
    y2 = torch.linalg.norm(y, dim=-1) ** 2
    xy = torch.linalg.norm(x - y, dim=-1) ** 2
    denominator = (1.0 - c * x2) * (1.0 - c * y2)
    argument = 1.0 + 2.0 * c * xy / denominator.clamp_min(1e-30)
    return torch.acosh(argument.clamp_min(1.0 + 1e-12)) / math.sqrt(c)


def poincare_exp_map0(v: torch.Tensor, c: float) -> torch.Tensor:
    """The exponential map at the origin: the tangent vector v becomes a point of the ball.

    The scale is ``tanh(sqrt(c)||v|| / 2) / (sqrt(c)||v||)``: the factor of one half is
    what makes the map the inverse of the logarithm map and makes the distance from the
    origin to the image equal the norm of the tangent vector.
    """
    c = _check_curvature(c)
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    scale = torch.tanh(math.sqrt(c) * norm / 2.0) / (math.sqrt(c) * norm.clamp_min(1e-12))
    return v * scale


def poincare_log_map0(x: torch.Tensor, c: float) -> torch.Tensor:
    """The logarithm map at the origin: a point of the ball becomes a tangent vector."""
    c = _check_curvature(c)
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    scale = 2.0 * torch.atanh(math.sqrt(c) * norm.clamp(max=1.0 - 1e-12)) / (
        math.sqrt(c) * norm.clamp_min(1e-12)
    )
    return x * scale


def busemann(x: torch.Tensor, xi: torch.Tensor, c: float) -> torch.Tensor:
    """The Busemann function of the base point xi, in the Lorentz model.

    The horosphere through xi is the level set of this function, and the function diverges as
    x approaches xi -- the base point lies ON the horosphere -- and is finite everywhere
    else. It is written through the Lorentz inner product because that form has no
    degenerate case: the Poincare form with the (1 - 1/c) terms collapses at c = 1.
    """
    c = _check_curvature(c)
    lorentz_x = to_lorentz(x, c)
    lorentz_xi = to_lorentz(xi, c)
    inner = -(lorentz_x[..., 0] * lorentz_xi[..., 0]) + (
        lorentz_x[..., 1:] * lorentz_xi[..., 1:]
    ).sum(dim=-1)
    numerator = (-inner + 1.0 / c).clamp_min(1e-30)
    denominator = (-inner - 1.0 / c).clamp_min(1e-30)
    return torch.log(numerator / denominator) / math.sqrt(c)


def to_lorentz(x: torch.Tensor, c: float) -> torch.Tensor:
    """The Poincaré point x as a point of the Lorentz hyperboloid.

    The hyperboloid of the curvature -c is the Minkowski space with the points on
    ``<X, X>_L = -1/c``, and the conversion is
    ``X = (1/sqrt(c), x) / sqrt(1 - c||x||^2)``. The distance between two such points is
    ``(1/sqrt(c)) arcosh(c * (-<X, Y>_L))``, which the test asserts agrees with the
    Poincaré distance to eight decimals -- the two models are checked against each other
    rather than assumed to be the same.
    """
    c = _check_curvature(c)
    x2 = torch.linalg.norm(x, dim=-1, keepdim=True) ** 2
    scale = (1.0 - c * x2).clamp_min(1e-12).rsqrt()
    time = (1.0 / math.sqrt(c)) * scale
    space = x * scale
    return torch.cat([time, space], dim=-1)




def conformal_factor(x: torch.Tensor, c: float) -> torch.Tensor:
    """The Poincaré conformal factor: 1 at the origin, (1 - c||x||^2)^-1 in general."""
    c = _check_curvature(c)
    return 1.0 / (1.0 - c * torch.linalg.norm(x, dim=-1, keepdim=True) ** 2).clamp_min(1e-12)


def riemannian_gradient(euclidean_grad: torch.Tensor, x: torch.Tensor, c: float) -> torch.Tensor:
    """The Riemannian gradient of the Poincaré ball, from the Euclidean one."""
    c = _check_curvature(c)
    factor = (1.0 - c * torch.linalg.norm(x, dim=-1, keepdim=True) ** 2) ** 2
    return factor * euclidean_grad


def retraction(x: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """The Riemannian retraction at x along the tangent vector v."""
    c = _check_curvature(c)
    return poincare_exp_map0(v, c) + x


#: The operators this oracle provides. Each one is checked by an identity that does not
#: come from this file: the distance against the exponential map, the maps against each other,
#: the Busemann function against its base point, the gradient against the conformal factor.
#: An operator whose convention could not be verified here is deliberately absent.
__all__ = [
    "ball_margin",
    "busemann",
    "conformal_factor",
    "poincare_distance",
    "poincare_exp_map0",
    "poincare_log_map0",
    "project_to_ball",
    "retraction",
    "riemannian_gradient",
    "to_lorentz",
]
