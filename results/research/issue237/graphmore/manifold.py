"""κ-stereographic manifold operations, standing in for ``geoopt.Stereographic``.

``geoopt`` is not installed in this environment and the official GraphMoRE code takes all
of its manifold arithmetic from it, so the small surface that the κ-GCN needs is
implemented here from the definitions (Bachmann et al., *Constant Curvature Graph
Convolutional Networks*, 2020; the same model ``geoopt`` exposes as
``Stereographic``):

| operation | definition |
|---|---|
| ``expmap0(v)`` | ``tanh(√c‖v‖)/(√c‖v‖)·v`` for ``κ<0``; ``tan(√κ‖v‖)/(√κ‖v‖)·v`` for ``κ>0`` |
| ``logmap0(y)`` | ``artanh(√c‖y‖)/(√c‖y‖)·y`` / ``arctan(√κ‖y‖)/(√κ‖y‖)·y`` |
| ``mobius_add(x,y)`` | ``((1-2c⟨x,y⟩-c‖y‖²)x + (1+c‖x‖²)y) / (1-2c⟨x,y⟩+c²‖x‖²‖y‖²)``, ``c=-κ`` |
| ``dist(x,y)`` | ``2/√c·artanh(√c‖(-x)⊕y‖)`` / ``2/√κ·arctan(√κ‖(-x)⊕y‖)`` |
| ``mobius_matvec(W,x)`` | ``expmap0(W·logmap0(x))`` — tangent-space linear, mapped back |

The last row is the one place this differs from ``geoopt``'s closed-form ``mobius_matvec``:
the official code's version is an algebraically equivalent rescaling of the same operation,
and with ``geoopt`` unavailable the tangent-space form is the honest implementation.  The
difference is recorded in the report; the *propagation* inside ``kappaGCNConv`` (logmap0 →
normalised sum → expmap0) follows the official code exactly.

A learnable curvature is parameterised so that it can never cross zero: negative
curvatures are ``-softplus(raw)`` and positive ones ``+softplus(raw)``.  Curvature 0 is
frozen, which is what makes that expert a plain Euclidean GCN.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stereographic(nn.Module):
    """Stereographic model of constant curvature ``k`` (0 = Euclidean)."""

    def __init__(self, k: float, learnable: bool = False, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        self.learnable = bool(learnable) and float(k) != 0.0
        if self.learnable:
            # raw -> ±softplus(raw) keeps the sign, and the initial value matches k
            magnitude = float(abs(k))
            target = torch.log(torch.expm1(torch.tensor(magnitude)))
            self.raw_curvature = nn.Parameter(target.clone().detach())
            self._sign = -1.0 if float(k) < 0 else 1.0
        else:
            self.register_buffer("raw_curvature", torch.tensor(float(k)))
            self._sign = -1.0 if float(k) < 0 else 1.0

    @property
    def k(self) -> torch.Tensor:
        if not self.learnable:
            return self.raw_curvature
        magnitude = F.softplus(self.raw_curvature)
        return self._sign * magnitude

    # ---- helpers -----------------------------------------------------------
    def _radius(self) -> torch.Tensor:
        k = self.k
        if float(torch.abs(k)) == 0.0:
            return torch.tensor(float("inf"))
        return 1.0 / torch.sqrt(torch.abs(k))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x.norm(dim=-1, keepdim=True)

    def _scale_factor(self, norm: torch.Tensor) -> torch.Tensor:
        """``tanh(√c·r)/(√c·r)`` (κ<0) or ``tan(√κ·r)/(√κ·r)`` (κ>0); 1 at r = 0."""
        k = self.k
        if float(torch.abs(k)) == 0.0:
            return torch.ones_like(norm)
        root = torch.sqrt(torch.abs(k))
        scaled = root * norm
        safe = scaled.clamp(min=self.eps)
        ratio = torch.where(
            scaled > self.eps,
            (torch.tanh(safe) if float(k) < 0 else torch.tan(safe)) / safe,
            torch.ones_like(scaled),
        )
        return ratio

    def _inverse_scale_factor(self, norm: torch.Tensor) -> torch.Tensor:
        """``artanh(√c·r)/(√c·r)`` (κ<0) or ``arctan(√κ·r)/(√κ·r)`` (κ>0)."""
        k = self.k
        if float(torch.abs(k)) == 0.0:
            return torch.ones_like(norm)
        root = torch.sqrt(torch.abs(k))
        scaled = root * norm
        safe = scaled.clamp(min=self.eps)
        if float(k) < 0:
            inner = scaled.clamp(max=1.0 - self.eps)
            ratio = torch.where(scaled > self.eps,
                                torch.atanh(inner) / safe,
                                torch.ones_like(scaled))
        else:
            ratio = torch.where(scaled > self.eps,
                                torch.atan(safe) / safe,
                                torch.ones_like(scaled))
        return ratio

    # ---- operations --------------------------------------------------------
    def proju(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp a point strictly inside the ball of curvature ``k``."""
        k = self.k
        if float(torch.abs(k)) == 0.0:
            return x
        radius = self._radius()
        limit = radius * (1.0 - 1e-5)
        norm = self._norm(x)
        scale = torch.where(norm > limit, limit / norm.clamp(min=self.eps),
                            torch.ones_like(norm))
        return x * scale

    def origin(self, shape) -> torch.Tensor:
        return torch.zeros(*shape, device=self.raw_curvature.device)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        out = v * self._scale_factor(self._norm(v))
        return self.proju(out)

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        y = self.proju(y)
        return y * self._inverse_scale_factor(self._norm(y))

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gyrovector addition; ``rho = -k`` carries the sign of the curvature.

        The sign convention is fixed by requiring ``dist(x, y) = 2/sqrt(|k|) *
        atanh(sqrt(|k|) * ||(-x) + y||)`` to equal the closed-form geodesic distance,
        which it does for ``rho = -k`` with the ``+2*rho`` numerator below (verified
        numerically in ``tests/test_manifold.py``).
        """
        k = self.k
        if float(torch.abs(k)) == 0.0:
            return x + y
        rho = -k
        xy = (x * y).sum(-1, keepdim=True)
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        numerator = (1.0 + 2.0 * rho * xy + rho * y2) * x + (1.0 - rho * x2) * y
        denominator = 1.0 + 2.0 * rho * xy + rho * rho * x2 * y2
        return self.proju(numerator / denominator.clamp_min(self.eps))

    def mobius_matvec(self, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Tangent-space linear map followed by ``expmap0`` (see the module docstring)."""
        return self.expmap0(self.logmap0(x) @ weight.t())

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance; ``2‖x - y‖`` in the ``κ → 0`` limit.

        For negative curvature the ``arcosh`` closed form is used because it stays
        accurate far from the origin, where ``atanh`` of a near-1 argument loses digits.
        """
        k = self.k
        if float(torch.abs(k)) == 0.0:
            return 2.0 * (x - y).norm(dim=-1)
        x = self.proju(x)
        y = self.proju(y)
        c = -k
        if float(k) < 0:
            x2 = (x * x).sum(-1)
            y2 = (y * y).sum(-1)
            diff2 = ((x - y) ** 2).sum(-1)
            inner = 1.0 + 2.0 * c * diff2 / ((1.0 - c * x2) * (1.0 - c * y2))
            return torch.acosh(inner.clamp(min=1.0)) / torch.sqrt(c)
        root = torch.sqrt(abs(k))
        scaled = root * self.mobius_add(-x, y).norm(dim=-1)
        return (2.0 / root) * torch.atan(scaled)
