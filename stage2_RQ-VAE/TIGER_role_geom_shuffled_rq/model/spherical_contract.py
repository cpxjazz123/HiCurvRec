"""The spherical branch of the role geometry, with its own numerical contract.

The baseline's curved half lives in the Poincare ball (``poincare_contract``), whose curvature
convention is positive inside the ball.  Issue #215 adds a third geometry for the same tangent
block: the unit sphere.

A tangent vector at the origin of a sphere cannot be mapped into the sphere's own ``d`` dimensions:
the exponential map of the sphere sends the origin's tangent space into the sphere embedded in
``d + 1`` dimensions,

    exp_0(v) = ( sin(|v|) * v/|v| , cos(|v|) )   in R^{d+1},

and the geodesic distance between two unit sphere points is ``arccos(p . q)``.  That is the exact
map, and the distances are computed on the embedded points, so no approximation of a geodesic is
used anywhere.

Two numerical facts are part of the contract rather than an afterthought:

* the map is only injective for ``|v| < pi``: past that the sine turns negative and the point
  wraps around the sphere, so two different tangent vectors can name the same point.  The
  antipodal pair has no unique geodesic, and ``arccos`` returns ``pi`` for it.  Both are counted
  by :func:`sphere_report` so a run can report how often it met them.
* ``log_0`` divides by ``sin(theta)``, which vanishes at the poles: the inverse is undefined there
  and the value is clamped instead, which is what :func:`sphere_logmap0` does.
"""
from __future__ import annotations

import torch

#: Below this the norm is treated as zero and the map is the identity.
SPHERE_EPS = 1e-7
#: The tangent radius at which the exponential map stops being injective.
SPHERE_POLE = 3.141592653589793
#: How far the inner product is kept from the poles of the cosine.
SPHERE_DOT_EPS = 1e-7


def sphere_expmap0(v: torch.Tensor) -> torch.Tensor:
    """``exp_0`` of the unit sphere: ``(sin(|v|) v/|v|, cos(|v|))`` in one more dimension."""
    norm = v.norm(dim=-1, keepdim=True)
    direction = torch.where(norm > SPHERE_EPS, v / norm.clamp_min(SPHERE_EPS), torch.zeros_like(v))
    return torch.cat([torch.sin(norm) * direction, torch.cos(norm)], dim=-1)


def sphere_logmap0(p: torch.Tensor) -> torch.Tensor:
    """``log_0`` of the unit sphere, clamped at the poles where it does not exist."""
    theta = torch.acos(p[..., -1:].clamp(-1.0, 1.0))
    sine = torch.sin(theta).abs().clamp_min(SPHERE_EPS)
    direction = p[..., :-1] / sine
    return theta * direction


def sphere_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The geodesic distance between two sphere points, in ``[0, pi]``."""
    dot = (p * q).sum(-1).clamp(-1.0, 1.0)
    return torch.acos(dot)


def sphere_squared_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The squared geodesic distance: ``acos(p . q) ** 2``, with the cosine kept off the poles.

    The inner product is clamped a little inside ``[-1, 1]``.  At ``p . q = 1`` the derivative of
    ``acos`` is infinite, and a NaN gradient in the cost is a NaN model after one optimizer step;
    the clamped value costs at most ``2e-7`` of squared distance and keeps the gradient finite.
    """
    dot = (p * q).sum(-1).clamp(-1.0 + SPHERE_DOT_EPS, 1.0 - SPHERE_DOT_EPS)
    return torch.acos(dot).pow(2)


def sphere_report(p: torch.Tensor, q: torch.Tensor) -> dict:
    """How many pairs are antipodal.

    The counter exists because the exponential map of the sphere is not injective: a run that
    meets an antipodal pair has to say so rather than let the ambiguity hide inside a distance.
    """
    with torch.no_grad():
        dot = (p * q).sum(-1).clamp(-1.0, 1.0)
        antipodal = int((dot <= -1.0 + 1e-6).sum().item())
    return {"antipodal_pairs": antipodal}


def sphere_pole_count(tangent: torch.Tensor) -> int:
    """How many tangent vectors have reached or passed the pole, where the map stops being injective.

    The count is taken on the tangent side on purpose: once the vector has passed the pole the
    sphere point alone cannot say so, because the map has already wrapped it.
    """
    with torch.no_grad():
        norms = tangent.norm(dim=-1)
        return int((norms >= SPHERE_POLE - 1e-6).sum().item())
