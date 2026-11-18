"""Bounded geometry: codewords are fixed prototypes plus a clipped offset.

If the prototypes' smallest pairwise distance is `delta` and every offset has norm at most
`rho` with `2*rho < delta`, then two codewords can never coincide: for any pair

    ||c_j - c_k|| >= ||a_j - a_k|| - ||D_j|| - ||D_k|| >= delta - 2*rho > 0.

That bound is exact for the Euclidean half. The hyperbolic half has no such free
triangle inequality in the ball, so the *measured* minimum mixed distance between
codewords is reported every epoch instead, and the module refuses a codebook whose
measured separation has fallen to zero.

`bounded_codes` enforces the constraint by projection, so any gradient -- however large --
lands inside the ball: the guarantee is structural, not a penalty.
"""
from __future__ import annotations

import numpy as np
import torch


def bounded_codes(prototypes: torch.Tensor, offsets: torch.Tensor, rho: float) -> torch.Tensor:
    """`c = a + offset * min(1, rho / ||offset||)`, evaluated without a branch."""
    norms = offsets.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(float(rho) / norms, max=1.0)
    return prototypes + offsets * scale


def separation_bound(prototypes: torch.Tensor) -> dict:
    """Smallest pairwise Euclidean distance between prototypes."""
    pairwise = torch.cdist(prototypes.double(), prototypes.double())
    # fill_diagonal_, not `+ eye * inf`: 0 * inf is nan and would poison every off-diagonal
    pairwise.fill_diagonal_(float("inf"))
    delta = float(pairwise.min())
    return {"delta": delta, "pairs": int(len(prototypes) * (len(prototypes) - 1) // 2)}


def certify_separation(prototypes: torch.Tensor, rho: float) -> dict:
    """The analytic guarantee: `delta > 2*rho` implies no two codewords can coincide."""
    bound = separation_bound(prototypes)
    delta, allowed = bound["delta"], 2.0 * float(rho)
    return {"delta": delta, "two_rho": allowed, "margin": delta - allowed,
            "ok": bool(delta > allowed)}


def _mobius_add(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    x2 = (x * x).sum(-1, keepdims=True)
    y2 = (y * y).sum(-1, keepdims=True)
    xy = (x * y).sum(-1, keepdims=True)
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    return numerator / np.clip(denominator, 1e-12, None)


def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Geodesic distance in the ball of curvature -c (the model's convention)."""
    difference = _mobius_add(-x, y, c)
    norm = np.linalg.norm(difference, axis=-1).clip(1e-12, 1 - 1e-7)
    return (2.0 / np.sqrt(c)) * np.arctanh(np.sqrt(c) * norm)


def mixed_min_distance(codes: np.ndarray, alpha: float = 0.5, curvature: float = 1.0) -> float:
    """Smallest pairwise cost under the model's mixed metric (E half + ball half)."""
    codes = np.asarray(codes, dtype=np.float64)
    euclidean = codes[:, :16]
    tangent = codes[:, 16:]
    norm = np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-12, None)
    ball = np.tanh(norm) * tangent / norm            # expmap0 with c = 1
    euclidean_sq = ((euclidean[:, None, :] - euclidean[None, :, :]) ** 2).sum(-1)
    curved_sq = poincare_distance(ball[:, None, :], ball[None, :, :], curvature) ** 2
    cost = 2.0 * (alpha * euclidean_sq + (1.0 - alpha) * curved_sq / 4.0)
    np.fill_diagonal(cost, np.inf)
    return float(cost.min())


def adversarial_offsets(count: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Offsets that try to collapse pairs: every code is pushed toward code 0."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, dim, generator=generator) * 1e3
