"""Poincare-ball geometry and the branch-anchored tangent quantizer.

This module holds the curved part of the candidate mechanism. Everything here
follows one convention, because the specification asks for exactly that: the
same projection, the same base-point maps and the same distance are used for the
residuals, the anchors and the code vectors.

The convention
--------------
The ball has curvature ``-c``. The metric is ``ds^2 = lambda_x^2 dx^2`` with
``lambda_x = 2 / (1 - c ||x||^2)``, so the Riemannian norm of a tangent vector
at ``x`` is ``lambda_x`` times its Euclidean norm. Two consequences fix the
base-point maps, and they are the reason the conformal factor appears at all:

    log_x(y) = (1 / lambda_x) * log_0((-x) (+) y)
    exp_x(v) = x (+) exp_0(lambda_x * v)

They are the only pair that is consistent with both the metric and the round
trip ``exp_x(log_x(y)) == y``. The factor multiplies the exponential map and
divides the logarithm; the opposite direction is the same mistake with the
sign flipped, and it silently breaks the round trip. ``tests`` in
``smoke_test.py`` check the round trip in both directions, so the convention is
verified rather than asserted.

At the origin ``lambda_0 = 2``, and the maps reduce to the usual
``exp_0(v) = tanh(sqrt(c) ||v|| / 2) v / (sqrt(c) ||v||)`` and
``log_0(y) = (2 / sqrt(c)) artanh(sqrt(c) ||y||) y / ||y||``.
"""

from __future__ import annotations

import torch

#: Numerical floor for the denominators of the maps.
_EPS = 1e-9

#: The safeguards of the specification ask for clamped log-map norms. The
#: curvature is 1/R^2 with R the measured output radius of the encoder, so the
#: ball is small and a residual near its boundary would otherwise produce a
#: tangent vector of unbounded length.
LOG_MAP_MAX_NORM = 10.0


def ball_add(u: torch.Tensor, v: torch.Tensor, curvature: float) -> torch.Tensor:
    """The Mobius addition of the Poincare ball.

    The denominator is ``1 + 2 c <u, v> + c^2 ||u||^2 ||v||^2``. The form with
    ``c (||u||^2 + ||v||^2)`` belongs to the tangent-space addition and is not
    the same operation.
    """
    c = float(curvature)
    u_sq = u.square().sum(-1, keepdim=True)
    v_sq = v.square().sum(-1, keepdim=True)
    inner = (u * v).sum(-1, keepdim=True)
    numerator = (1.0 + 2.0 * c * inner + c * v_sq) * u + (1.0 - c * u_sq) * v
    denominator = 1.0 + 2.0 * c * inner + (c * c) * u_sq * v_sq
    return numerator / denominator.clamp_min(_EPS)


#: The projection margin of the specification, ``(1 - eps) / sqrt(c)``. The
#: proposal fixes it at the machine epsilon, and the curvature of this
#: candidate is c = 1, where the points of the latent stay well inside the
#: ball; the stiff margin was needed for a curvature of 142 and not here.
PROJECTION_EPS = 1e-5


def ball_project(x: torch.Tensor, curvature: float, eps: float = PROJECTION_EPS) -> torch.Tensor:
    """Project points strictly inside the ball of radius ``(1 - eps) / sqrt(c)``."""
    c = float(curvature)
    radius = (1.0 - eps) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    scale = torch.clamp(radius / norm, max=1.0)
    return x * scale


def conformal_factor(base: torch.Tensor, curvature: float) -> torch.Tensor:
    """``lambda_x = 2 / (1 - c ||x||^2)``, the metric factor at ``base``."""
    c = float(curvature)
    squared = (c * base.square().sum(-1, keepdim=True)).clamp(max=1.0 - 1e-6)
    return 2.0 / (1.0 - squared)


def ball_log_map(base: torch.Tensor, point: torch.Tensor, curvature: float) -> torch.Tensor:
    """The tangent vector at ``base`` whose exponential map reaches ``point``.

    The displacement is ``(-base) (+) point``: the point measured from the
    origin and brought back to the base. Using ``base (+) (-point)`` gives the
    opposite direction.
    """
    c = float(curvature)
    difference = ball_add(-base, point, c)
    norm = difference.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    log_origin = (2.0 / (c ** 0.5)) * torch.atanh(
        (c ** 0.5) * norm.clamp(max=1.0 - 1e-5)
    ) / norm * difference
    tangent = log_origin / conformal_factor(base, c)
    # The bound the safeguards ask for on the log-map norm. A hard clamp would
    # zero the gradient wherever it binds, so the bound is soft: below the
    # threshold the tangent is unchanged, above it the length is compressed
    # towards the threshold and the gradient survives.
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    compressed = LOG_MAP_MAX_NORM * torch.tanh(tangent_norm / LOG_MAP_MAX_NORM) / tangent_norm.clamp_min(_EPS)
    # Below the threshold the tangent is untouched, so the maps still invert
    # each other exactly; above it the length is compressed smoothly, so the
    # gradient of the residual path survives.
    scale = torch.where(tangent_norm <= LOG_MAP_MAX_NORM,
                         torch.ones_like(tangent_norm), compressed)
    return tangent * scale


def ball_exp_map(base: torch.Tensor, tangent: torch.Tensor, curvature: float) -> torch.Tensor:
    """The point of the ball that the tangent vector at ``base`` reaches.

    The tangent at the base becomes the Euclidean tangent at the origin after
    the conformal factor is applied, and the norm that enters the hyperbolic
    tangent is the norm of that scaled vector, not the norm of the input.
    """
    c = float(curvature)
    scaled = tangent * conformal_factor(base, c)
    norm = scaled.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    out = (torch.tanh((c ** 0.5) * norm / 2.0) / ((c ** 0.5) * norm)) * scaled
    return ball_add(base, out, c)


def ball_distance(x: torch.Tensor, y: torch.Tensor, curvature: float) -> torch.Tensor:
    """The geodesic distance of the ball, in the numerically stable form."""
    c = float(curvature)
    x_sq = (c * x.square().sum(-1)).clamp(max=1.0 - 1e-6)
    y_sq = (c * y.square().sum(-1)).clamp(max=1.0 - 1e-6)
    numerator = (x - y).square().sum(-1) * c
    denominator = (1.0 - x_sq) * (1.0 - y_sq)
    # The argument of acosh is clamped to 1 + 1e-7, not to 1: the derivative of
    # acosh is unbounded at 1, and two distinct float32 points can round the
    # argument to exactly 1, which would make the gradient of the ranking loss
    # infinite instead of zero.
    argument = (1.0 + 2.0 * numerator / denominator.clamp_min(_EPS)).clamp_min(1.0 + 1e-7)
    return torch.acosh(argument) / (c ** 0.5)


def _sinkhorn_indices(distances: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    """The baseline's balanced assignment, applied to a distance matrix.

    The distances are centred first, exactly as the baseline's VQ layer does:
    without that the exponential underflows and the assignment degenerates.
    """
    maximum = distances.max()
    minimum = distances.min()
    middle = (maximum + minimum) / 2
    amplitude = maximum - middle + 1e-5
    if not bool(amplitude > 0):
        raise ValueError("Cannot center a constant distance matrix.")
    centred = ((distances - middle) / amplitude).double()
    q = torch.exp(-centred / float(epsilon))
    batch_size, codebook_size = q.shape
    q /= q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    for _ in range(int(iterations)):
        q /= q.sum(0, keepdim=True)
        q /= codebook_size
        q /= q.sum(1, keepdim=True)
        q /= batch_size
    q *= batch_size
    if torch.isnan(q).any() or torch.isinf(q).any():
        raise RuntimeError("Sinkhorn assignment returned NaN or infinity.")
    return torch.argmax(q, dim=-1)


