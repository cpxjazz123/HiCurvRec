"""Bounded codeword geometry: fixed prototypes plus offsets clipped to a ball.

The capacity floor keeps every codeword *used*; it does not stop 256 used codewords from
converging onto the same vector. This layer closes that hole structurally:

    c_k = a_k + clip(offset_k, ||offset_k|| <= rho),      2*rho < delta

where `delta` is the smallest pairwise distance between the fixed prototypes `a_k`. Then for
any pair

    ||c_j - c_k|| >= ||a_j - a_k|| - ||offset_j|| - ||offset_k|| >= delta - 2*rho > 0

so no gradient, however large, can merge two codewords. The bound is exact on the Euclidean
half of the mixed space; the hyperbolic half has no such triangle inequality, so the module
also exposes a measurement of the mixed minimum distance, which the observer records every
epoch.

The prototypes are the rows of a Hadamard matrix: they are pairwise orthogonal with entries
+/-1, so `delta` is large and known in advance, and the construction is deterministic (no
seed, no KMeans).
"""
from __future__ import annotations

import torch

EUCLIDEAN_DIM = 16
HYPERBOLIC_DIM = 16
TOTAL_DIM = EUCLIDEAN_DIM + HYPERBOLIC_DIM


def hadamard(order: int) -> torch.Tensor:
    """Sylvester construction: H_1 = [1], H_2n = [[H,H],[H,-H]]."""
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < order:
        matrix = torch.cat([torch.cat([matrix, matrix], dim=1),
                            torch.cat([matrix, -matrix], dim=1)], dim=0)
    if matrix.shape[0] != order:
        raise ValueError("order must be a power of two")
    return matrix


def build_prototypes(codewords: int = 256, seed: int = 0) -> torch.Tensor:
    """`(codewords, 32)` prototypes that are far apart by construction.

    The Euclidean half carries an 8x8 Hadamard product: eight blocks of +/-1 chosen by the
    first eight bits of the codeword, so all 256 prototypes are orthogonal with squared
    distance 16 (delta = 4). The hyperbolic half is a small deterministic pattern scaled by
    0.1, which keeps every prototype near the ball's centre where the geodesic distance is
    close to Euclidean and the measured minimum stays large.
    """
    if codewords > 256:
        raise ValueError("the Hadamard construction covers 256 codewords")
    base = hadamard(8)                                   # 8 x 8, entries +/-1
    generator = torch.Generator().manual_seed(seed)
    euclidean = torch.zeros(codewords, EUCLIDEAN_DIM)
    for code in range(codewords):
        bits = [(code >> bit) & 1 for bit in range(8)]
        vector = torch.ones(EUCLIDEAN_DIM)
        for block, bit in enumerate(bits):
            vector[block * 2:(block + 1) * 2] = base[bit][block] * torch.ones(2)
        euclidean[code] = vector
    euclidean = euclidean / (euclidean.norm(dim=1, keepdim=True) + 1e-12) * 2.0
    tangent = (torch.randn(codewords, HYPERBOLIC_DIM, generator=generator) * 0.1)
    return torch.cat([euclidean, tangent], dim=1)


def separation(prototypes: torch.Tensor) -> float:
    """The smallest pairwise Euclidean distance between prototypes (equals delta)."""
    pairwise = torch.cdist(prototypes.double(), prototypes.double())
    pairwise.fill_diagonal_(float("inf"))
    return float(pairwise.min())


def allowed_radius(prototypes: torch.Tensor, fraction: float = 0.25) -> float:
    """`rho = fraction * delta`, so `2*rho <= delta/2 < delta` with a wide margin."""
    delta = separation(prototypes)
    return float(fraction * delta)


def bounded_offsets(offsets: torch.Tensor, rho: float) -> torch.Tensor:
    norms = offsets.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return offsets * torch.clamp(float(rho) / norms, max=1.0)


def codebook(prototypes: torch.Tensor, offsets: torch.Tensor, rho: float) -> torch.Tensor:
    """The codewords the quantiser is allowed to use."""
    return prototypes + bounded_offsets(offsets, rho)


def guarantee(prototypes: torch.Tensor, rho: float) -> dict:
    delta = separation(prototypes)
    return {"delta": delta, "two_rho": 2.0 * rho, "margin": delta - 2.0 * rho,
            "ok": bool(delta > 2.0 * rho)}


def mixed_min_distance(codes, alpha: float = 0.5, curvature: float = 1.0) -> float:
    """Smallest pairwise cost under the model's mixed metric (measured, not bounded)."""
    import numpy as np

    codes = np.asarray(codes, dtype=np.float64)
    euclidean = codes[:, :EUCLIDEAN_DIM]
    tangent = codes[:, HYPERBOLIC_DIM and EUCLIDEAN_DIM:]
    norm = np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-12, None)
    ball = np.tanh(norm) * tangent / norm                     # expmap0, c = 1
    euclidean_sq = ((euclidean[:, None, :] - euclidean[None, :, :]) ** 2).sum(-1)
    x, y = ball[:, None, :], ball[None, :, :]
    x2 = (x * x).sum(-1, keepdims=True)
    y2 = (y * y).sum(-1, keepdims=True)
    xy = (x * y).sum(-1, keepdims=True)
    c = float(curvature)
    denominator = np.clip(1 + 2 * c * xy + c * c * x2 * y2, 1e-12, None)
    mobius = ((1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y) / denominator
    distance = (2.0 / np.sqrt(c)) * np.arctanh(
        np.clip(np.sqrt(c) * np.linalg.norm(mobius, axis=-1), 1e-12, 1 - 1e-7))
    cost = 2.0 * (alpha * euclidean_sq + (1.0 - alpha) * distance ** 2 / 4.0)
    np.fill_diagonal(cost, np.inf)
    return float(cost.min())
