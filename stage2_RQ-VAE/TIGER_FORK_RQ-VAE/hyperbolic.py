"""Poincare-ball operations for the behaviour-geometry experiments.

Shared by Exp2 (hyperbolic contrastive), Exp3 (curvature sweep), Exp4/5
(directional transition) and Exp7 (hyperbolic residual quantization).

Conventions
-----------
Curvature is supplied as the positive ``c`` of a ball of curvature ``-c``.
``expmap0`` sends a tangent vector at the origin to the ball; its output always
satisfies ``||z|| < 1``, so downstream distances stay finite by construction.

Numerical guards, all of them required for stable training:

  * ``expmap0`` clamps the tangent norm it divides by, so a zero latent maps to
    the origin instead of producing NaN.
  * ``_inside`` rescales any point whose norm reaches ``1 - BOUNDARY_EPS``, and
    floors the norm away from zero, so ``1 - ||z||**2`` never collapses.
  * ``poincare_distance`` clamps its ``acosh`` argument to ``1 + ACOSH_EPS``.
    Without it the gradient of ``acosh`` at coincident points is infinite, and
    an all-positive batch at initialisation does contain coincident pairs.

All distances are returned in the *hyperbolic* scale: two points whose images
sit at radius ``r`` with angle ``theta`` obey
``cosh(sqrt(c) * d) = 1 + 2 c ||x - y||^2 / ((1 - c||x||^2)(1 - c||y||^2))``.
"""

from __future__ import annotations

import torch

BOUNDARY_EPS = 1e-5
ACOSH_EPS = 1e-7
NORM_FLOOR = 1e-12


def _norms(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=-1)


def _inside(x: torch.Tensor, c: float) -> torch.Tensor:
    """Rescale points whose ``sqrt(c) * ||x||`` reaches the ball boundary."""
    max_norm = (1.0 - BOUNDARY_EPS)
    scaled = (c ** 0.5) * _norms(x)
    overflow = scaled > max_norm
    if bool(overflow.any()):
        factor = torch.where(
            overflow, max_norm / scaled.clamp_min(NORM_FLOOR), torch.ones_like(scaled)
        )
        x = x * factor.unsqueeze(-1)
    return x


def expmap0(x: torch.Tensor, c: float) -> torch.Tensor:
    """Exponential map at the origin: tangent vector -> Poincare ball."""
    sqrt_c = c ** 0.5
    norm = _norms(x).clamp_min(NORM_FLOOR).unsqueeze(-1)
    return torch.tanh(sqrt_c * norm) / (sqrt_c * norm) * x


def ball_radius(c: float) -> float:
    """Radius of the ball of curvature ``-c``."""
    return 1.0 / (c ** 0.5)


def encode_ball(latent: torch.Tensor, c: float, normalizer) -> torch.Tensor:
    """Map an encoder latent into the ball with a FIXED, batch-independent scale.

    ``normalizer`` is a scalar (or 1-element tensor) computed once from the
    training set as ``atanh(target_radius_fraction) / median(||latent||)``. The
    map is then a pure function of the latent: the same item always receives the
    same ball coordinate, no matter which items share its batch and no matter
    whether it is being used as an anchor or as a candidate.

    An earlier version divided by the *batch* RMS. That was wrong in two
    separately measurable ways:

      * one item's ``||z||`` moved between 0.5046 and 0.5561 purely from batch
        membership (0.700 when encoded alone), so the "distance" between two
        items was not a function of the items;
      * anchors and candidates were mapped by two separate calls, each dividing
        by its own RMS (0.064164 vs 0.062453), which placed them in two
        different balls and made the reported distance meaningless as a metric.

    A fixed normalizer removes both. It is a constant within a run, exactly like
    the temperature, and only needs recomputing if the encoder's output scale is
    deliberately changed.
    """
    if not torch.is_tensor(normalizer):
        normalizer = torch.as_tensor(normalizer, dtype=latent.dtype, device=latent.device)
    return expmap0(latent * normalizer, c)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Geodesic distance in the ball, shape broadcast over leading dims."""
    x = _inside(x, c)
    y = _inside(y, c)
    sqrt_c = c ** 0.5
    x_norm_sq = (c ** 0.5) ** 2 * _norms(x).square()
    y_norm_sq = (c ** 0.5) ** 2 * _norms(y).square()
    denominator = ((1.0 - x_norm_sq) * (1.0 - y_norm_sq)).clamp_min(NORM_FLOOR)
    argument = 1.0 + 2.0 * c * (x - y).square().sum(-1) / denominator
    return torch.acosh(argument.clamp_min(1.0 + ACOSH_EPS)) / sqrt_c


def pair_poincare_distance(
    anchors: torch.Tensor, others: torch.Tensor, c: float
) -> torch.Tensor:
    """All-pairs distance matrix between two batches already in the ball."""
    return poincare_distance(anchors[:, None, :], others[None, :, :], c)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Mobius addition, the ball's translation-like operation."""
    x = _inside(x, c)
    y = _inside(y, c)
    x_sq = _norms(x).square().unsqueeze(-1)
    y_sq = _norms(y).square().unsqueeze(-1)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denominator = (1 + 2 * c * xy + c * c * x_sq * y_sq).clamp_min(NORM_FLOOR)
    return _inside(numerator / denominator, c)


def mobius_matvec(matrix: torch.Tensor, x: torch.Tensor, c: float) -> torch.Tensor:
    """Mobius matrix-vector product: the linear layer of a hyperbolic MLP."""
    x = _inside(x, c)
    sqrt_c = c ** 0.5
    x_norm = _norms(x).clamp_min(NORM_FLOOR)
    mx = x @ matrix.t()
    mx_norm = _norms(mx).clamp_min(NORM_FLOOR)
    artanh = torch.atanh((sqrt_c * x_norm).clamp(max=1.0 - BOUNDARY_EPS))
    scale = torch.tanh((mx_norm / x_norm) * artanh) / (sqrt_c * mx_norm)
    return _inside(scale.unsqueeze(-1) * mx, c)


def spatial_sq_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The matched Euclidean control: squared Euclidean distance in the ball."""
    return (x - y).square().sum(-1)
