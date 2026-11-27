"""Frozen kappa-stereographic distance for one local triple.

Convention (Bachmann et al., ICML 2020, kappa-stereographic; same family as
``tools/graph_infra/manifold.py``): curvature ``k``, ball radius ``1/sqrt(|k|)``
for ``k<0``, and the continuous flat limit ``d_0(x,y) = 2*||x-y||``.  The frozen
coordinate map ``w = Z_ref/(2*S)`` makes ``k=0`` equal to
``||Z_i - Z_j||/S``.

Every distance of one triple uses that triple's own frozen ``kappa_t``; the
value is a table lookup and carries no gradient.  The distance keeps the full
gradient with respect to the quantized representation.
"""
from __future__ import annotations

import math

import torch

#: asin is evaluated at 1-eps on the spherical branch; at exactly one its
#: derivative is infinite, and no coordinate of the frozen bounded box reaches
#: the antipode.
_SIN_LIMIT = 1.0 - 1e-4


class _SafeSqrt(torch.autograd.Function):
    """Exact square root with a finite derivative at zero.

    The composition ``asin(sqrt(t))**2`` is smooth at ``t=0`` but the intermediate
    derivative of ``sqrt`` is infinite; flooring only the *backward* denominator
    keeps coincident points exactly at zero distance.
    """

    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return value.clamp_min(0.0).sqrt()

    @staticmethod
    def backward(ctx, grad):
        value, = ctx.saved_tensors
        return grad / (2.0 * value.clamp_min(0.0).clamp_min(1e-12).sqrt())


def squared(x: torch.Tensor, y: torch.Tensor, kappa, *, paired: bool = False
            ) -> torch.Tensor:
    """Squared distance in the model of curvature ``kappa``.

    ``x``/``y`` broadcast to ``[..., D]``; ``kappa`` is a scalar or broadcastable
    to the leading shape.  The arithmetic is elementwise, so ``paired`` only
    documents intent.
    """
    del paired
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("kappa-stereographic inputs must share the last dimension")
    k = kappa if torch.is_tensor(kappa) else torch.as_tensor(
        kappa, dtype=x.dtype, device=x.device)
    k = k.to(x.dtype)
    xy = (x - y).square().sum(-1)
    x2 = x.square().sum(-1)
    y2 = y.square().sum(-1)
    magnitude = k.abs().clamp_min(torch.finfo(x.dtype).tiny)
    # 1 - |k|x^2 == 1 + k*x^2, one expression for both signs.
    denominator = (1.0 + k * x2) * (1.0 + k * y2)
    if bool((denominator <= 0).any()):
        raise ValueError("coordinate outside the frozen curvature domain")
    t2 = (magnitude * xy / denominator).clamp_min(0.0)
    # sqrt'(0) is infinite; the composition asin(sqrt(t))^2 is smooth, so the
    # custom root keeps the backward finite at coincident points while the
    # forward stays exactly zero there.
    root = _SafeSqrt.apply(t2.clamp_max(_SIN_LIMIT))
    sphere = torch.asin(root).square() / magnitude
    hyperbolic = torch.asinh(root).square() / magnitude
    values = torch.where(k > 0, sphere, torch.where(k < 0, hyperbolic, xy))
    return 4.0 * values


def distance(x: torch.Tensor, y: torch.Tensor, kappa) -> torch.Tensor:
    return squared(x, y, kappa).clamp_min(0.0).sqrt()


def project(x: torch.Tensor, kappa, eps: float = 1e-2) -> torch.Tensor:
    """Project coordinates into the open ball of their curvature.

    The frozen box bounds ``Z_ref``, not the quantized ``q``: training moves the
    code sum and can leave the ball.  The standard kappa-stereographic projection
    clamps only the radius, so the distance stays finite and differentiable for
    every point that was inside, while a point outside loses its radial push
    instead of producing a zero denominator.
    """
    k = kappa if torch.is_tensor(kappa) else torch.as_tensor(
        kappa, dtype=x.dtype, device=x.device)
    k = k.to(x.dtype)
    negative = k < 0
    # A finite placeholder for k >= 0: the unselected branch of ``where`` must
    # not contain inf, otherwise ``0 * inf`` in its backward yields NaN.
    radius = (1.0 - eps) / torch.where(negative, -k, torch.ones_like(k)).sqrt()
    norm = x.norm(dim=-1, keepdim=True)
    while radius.ndim < norm.ndim:
        radius = radius.unsqueeze(-1)
        negative = negative.unsqueeze(-1)
    scale = torch.where(norm > radius, radius / norm.clamp_min(torch.finfo(x.dtype).tiny),
                        torch.ones_like(norm))
    scale = torch.where(negative, scale, torch.ones_like(scale))
    return x * scale


def coordinate_scale(z) -> float:
    """Largest reference norm; the frozen scale is ``S`` in ``w = z/(2*S)``."""
    if torch.is_tensor(z):
        norms = z.detach().double().square().sum(-1).sqrt()
    else:
        import numpy as np
        array = np.asarray(z, dtype=np.float64)
        norms = np.sqrt((array * array).sum(-1))
    value = float(norms.max())
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid reference representation scale")
    return value
