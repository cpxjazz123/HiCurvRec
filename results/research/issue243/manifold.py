"""κ-stereographic operators (torch), the production side of CONTRACT.md §1.

Every function here has a numpy float64 twin in ``reference.py``; ``tests/test_reference.py``
compares them on random inputs, so a factor or sign error cannot survive.

The convention is the one GraphMoRE and geoopt use: ``k`` is the sectional curvature of
``M_k = {x : -k |x|^2 < 1}``, i.e. the ball radius is ``1/sqrt(|k|)``.  ``k = 0`` is the
Euclidean limit.
"""
from __future__ import annotations

import math

import torch

EPS = 1e-12


def tan_k(v: torch.Tensor, k: float) -> torch.Tensor:
    if k > 0:
        return torch.tan(math.sqrt(k) * v) / math.sqrt(k)
    if k < 0:
        return torch.tanh(math.sqrt(-k) * v) / math.sqrt(-k)
    return v


def tan_k_inv(u: torch.Tensor, k: float) -> torch.Tensor:
    if k > 0:
        return torch.atan(math.sqrt(k) * u) / math.sqrt(k)
    if k < 0:
        scale = math.sqrt(-k)
        return torch.atanh((scale * u).clamp(-1.0 + 1e-7, 1.0 - 1e-7)) / scale
    return u


def proju(x: torch.Tensor, k: float, eps: float = 1e-3) -> torch.Tensor:
    """Clamp into the coordinate domain.

    k < 0: open Poincare ball of radius (1-eps)/sqrt|k|.
    k > 0: the stereographic plane is unbounded (the pole is at infinity); clamping at the
           hyperbolic radius -- what #243 did -- is wrong and is part of why the spherical
           distance came out as pi/2.  Only a large numerical guard is applied.
    k = 0: identity.
    """
    if k == 0:
        return x
    norm = x.norm(dim=-1, keepdim=True)
    if k > 0:
        guard = 1e6 / math.sqrt(k)
        scale = torch.where(norm > guard, guard / norm.clamp(min=EPS), torch.ones_like(norm))
        return x * scale
    maxnorm = (1.0 - eps) / math.sqrt(abs(k))
    scale = torch.where(norm > maxnorm, maxnorm / norm.clamp(min=EPS), torch.ones_like(norm))
    return x * scale


def expmap0(x: torch.Tensor, k: float) -> torch.Tensor:
    """k > 0 uses r = R tan(theta / 2R) so that dist(0, expmap0(v)) equals |v|."""
    norm = x.norm(dim=-1, keepdim=True)
    if k > 0:
        radius = 1.0 / math.sqrt(k)
        mapped = radius * torch.tan(norm / (2.0 * radius))
    else:
        mapped = tan_k(norm, k)
    scale = torch.where(norm > EPS, mapped / norm.clamp(min=EPS), torch.ones_like(norm))
    return proju(scale * x, k)


def logmap0(y: torch.Tensor, k: float) -> torch.Tensor:
    y = proju(y, k)
    norm = y.norm(dim=-1, keepdim=True)
    if k > 0:
        radius = 1.0 / math.sqrt(k)
        mapped = 2.0 * radius * torch.atan(norm / radius)  # theta = 2R*arctan(r/R)
    else:
        mapped = tan_k_inv(norm, k)
    scale = torch.where(norm > EPS, mapped / norm.clamp(min=EPS), torch.ones_like(norm))
    return scale * y


def mobius_add(x: torch.Tensor, y: torch.Tensor, k: float) -> torch.Tensor:
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    denominator = 1.0 - 2.0 * k * xy + k * k * x2 * y2
    if bool((denominator.abs() < 1e-9).any()):
        raise FloatingPointError("mobius_add denominator vanished")
    numerator = (1.0 - 2.0 * k * xy - k * y2) * x + (1.0 + k * x2) * y
    return proju(numerator / denominator, k)


def mobius_matvec(matrix: torch.Tensor, x: torch.Tensor, k: float) -> torch.Tensor:
    return expmap0(x @ matrix.t(), k)


def dist(x: torch.Tensor, y: torch.Tensor, k: float) -> torch.Tensor:
    """Distance in the model of curvature ``k``.

    k > 0: closed form of the stereographic sphere,
           R * arccos((4R^2<x,y> + (|x|^2-R^2)(|y|^2-R^2)) / ((R^2+|x|^2)(R^2+|y|^2))).
           ``reference.dist`` computes the same number the long way, through the inverse
           stereographic projection and a three-dimensional angle.
    k < 0: Poincare ball, 2 * tan_k^{-1}(|(-x) +_k y|).
    k = 0: Euclidean.
    """
    if k == 0:
        return (x - y).norm(dim=-1)
    if k > 0:
        radius = 1.0 / math.sqrt(k)
        x2 = (x * x).sum(-1)
        y2 = (y * y).sum(-1)
        xy = (x * y).sum(-1)
        r2 = radius * radius
        numerator = 4.0 * r2 * xy + (x2 - r2) * (y2 - r2)
        denominator = (r2 + x2) * (r2 + y2)
        return radius * torch.acos((numerator / denominator).clamp(-1.0, 1.0))
    moved = mobius_add(-x, y, k)
    return 2.0 * tan_k_inv(moved.norm(dim=-1), k)


def project_to_manifold(x: torch.Tensor, k: float, eps: float = 1e-3) -> torch.Tensor:
    """Map a raw vector into the manifold; the zero vector maps to the origin."""
    return proju(x, k, eps)
