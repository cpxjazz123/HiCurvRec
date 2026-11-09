"""Curvature -1 Poincare-ball operators for the c117 candidate.

Frozen candidate: geometry-asymmetric-next-item-hyperbolic-semantic-id
(run_f8049f2ce2774d7c, candidate sha256 c8701416d4b694141d60366f61d3b70b4f
bbc958f8188447d822f8d36e6a283d).

The operator is the one the proposal freezes:

    B^32 = {x in R^32 : ||x|| < 1},  curvature -1
    exp_0(v) = tanh(||v||) * v / ||v||,   exp_0(0) = 0
    log_0(x)  = artanh(||x||) * x / ||x||, log_0(0) = 0
    a (+) b   = ((1 + 2<a,b> + ||b||^2) a + (1 - ||a||^2) b)
                / (1 + 2<a,b> + ||a||^2 ||b||^2)
    D(x, y)   = 2 * artanh(|| (-x) (+) y ||)

Every routine is differentiable and safe at the two removable singularities:
the zero-norm limit of exp_0/log_0 and the boundary of the ball. The radius is
clamped below 1 - EPSILON so artanh never sees 1.

In the small-radius limit these operators reduce to the Euclidean ones
(D -> ||x - y|| and log_0 -> x), which is what makes the baseline tree the
capacity-matched Euclidean control of this candidate.
"""
from __future__ import annotations

import torch

#: The radius is kept strictly below 1 so artanh stays finite.
EPSILON = 1e-6


def _norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)


def _direction(tensor: torch.Tensor, *, epsilon: float = EPSILON) -> torch.Tensor:
    """The unit direction of a vector; exactly zero for the zero vector."""
    norm = _norm(tensor)
    scale = torch.where(norm > epsilon, 1.0 / norm.clamp(min=epsilon), torch.zeros_like(norm))
    return tensor * scale


def exp_0(v: torch.Tensor, *, epsilon: float = EPSILON) -> torch.Tensor:
    """Exponential map at the origin of the curvature -1 ball.

    The radius is clamped below 1 - epsilon, as the frozen candidate requires, so
    artanh never sees the boundary and the geometry stays differentiable.
    """
    norm = _norm(v)
    radius = torch.tanh(norm).clamp(max=1.0 - epsilon)
    return radius * _direction(v, epsilon=epsilon)


def log_0(x: torch.Tensor, *, epsilon: float = EPSILON) -> torch.Tensor:
    """Logarithm map at the origin of the curvature -1 ball."""
    radius = _norm(x).clamp(max=1.0 - epsilon)
    artanh = 0.5 * torch.log((1.0 + radius) / (1.0 - radius))
    return artanh * _direction(x, epsilon=epsilon)


def project_to_ball(x: torch.Tensor, *, epsilon: float = EPSILON) -> torch.Tensor:
    """Scale a vector into the open ball without changing its direction."""
    radius = _norm(x)
    limit = 1.0 - epsilon
    scale = torch.where(radius > limit, limit / radius.clamp(min=epsilon), torch.ones_like(radius))
    return x * scale


def mobius_add(a: torch.Tensor, b: torch.Tensor, *, epsilon: float = EPSILON) -> torch.Tensor:
    """Möbius addition in the curvature -1 ball."""
    inner = (a * b).sum(dim=-1, keepdim=True)
    a_sq = (a * a).sum(dim=-1, keepdim=True)
    b_sq = (b * b).sum(dim=-1, keepdim=True)
    numerator = (1.0 + 2.0 * inner + b_sq) * a + (1.0 - a_sq) * b
    denominator = 1.0 + 2.0 * inner + a_sq * b_sq
    # The denominator is positive for points inside the ball; the clamp only
    # protects the float error at the boundary.
    return numerator / denominator.clamp(min=epsilon)


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, *, epsilon: float = EPSILON) -> torch.Tensor:
    """D(x, y) = 2 * artanh(|| (-x) (+) y ||), the frozen candidate operator.

    The operator is exact: the only clamp is on the argument of artanh, and it
    binds only for a pair whose Mobius offset reaches the boundary, where the
    true distance diverges anyway. A smooth reparameterization of the argument
    was tried and rejected: it changed the operator itself (the collinear
    closed form 2*atanh(1/3) came out as 2/3) and the independent MATH test
    caught it.
    """
    offset = mobius_add(-x, y, epsilon=epsilon)
    squared = (offset * offset).sum(dim=-1, keepdim=True).clamp(min=0.0)
    norm = torch.sqrt(squared)
    return 2.0 * torch.atanh(norm.clamp(max=1.0 - epsilon))


__all__ = [
    "EPSILON",
    "exp_0",
    "log_0",
    "project_to_ball",
    "mobius_add",
    "hyperbolic_distance",
]
