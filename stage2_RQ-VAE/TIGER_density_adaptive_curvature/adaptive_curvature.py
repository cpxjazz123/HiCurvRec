"""M3: Density-Adaptive Curvature.

Mechanism (the single variable relative to TIGER_CROWDING_CTRL)
--------------------------------------------------------------
The curvature of the ball used for codeword assignment is not a constant: it is
read off the *current* crowding of the L1 usage distribution. A concentrated
usage histogram (few codewords carry everything) raises the curvature, which
shrinks the ball and makes the geodesic assignment more discriminative; a spread
histogram lowers it back toward the Euclidean-like regime. The value is refreshed
at every evaluation step and is the only thing this tree changes about the
assignment.

Math
----
With the L1 codeword usage histogram ``counts`` over ``K = 256`` codewords,

    p_i = counts_i / sum(counts)
    H   = -sum_i p_i ln p_i / ln K          (normalised usage entropy, [0, 1])
    c   = c_min + (c_max - c_min) * (1 - H)  clamped to [c_min, c_max]

so a perfectly uniform usage (H = 1) gives the minimum curvature and a collapsed
usage (H -> 0) gives the maximum. ``c`` is then used as the curvature of
``encode_points`` + ``pair_poincare_distance`` for every level's assignment.

Composition with the shared layer
---------------------------------
``scripts/capacity_mask.py`` is installed *after* this module, so the +inf
sentinel for the 128 disabled L1 codewords always dominates the final argmin and
the capacity constraint cannot be relaxed by the curvature change.

Distributed detail: the histogram is computed on rank 0 (the only rank that walks
the full item set), then the scalar is broadcast to all four ranks right after the
evaluation barrier, so every rank assigns with the same curvature.

What is completely unchanged relative to the control
---------------------------------------------------
L1 capacity mask and diagnostics, all losses (reconstruction, commitment,
behaviour contrastive), pair sampling, codebook sizes and parameters, optimiser,
schedule, seed, epochs, evaluation interval.

Fixed parameters (no sweep): ``c_min = 0.25``, ``c_max = 4.0``, ``K = 256``;
initial value 1.0 (the fixed curvature the geodesic-assignment patch uses).
"""
from __future__ import annotations

import math
from types import MethodType

import torch

from hyperbolic import pair_poincare_distance
from scripts.hyp_losses import encode_points

C_MIN = 0.25
C_MAX = 4.0
CODEBOOK_K = 256
INITIAL_CURVATURE = 1.0

_CURRENT_CURVATURE = INITIAL_CURVATURE


def curvature_from_counts(counts, k: int = CODEBOOK_K, c_min: float = C_MIN,
                          c_max: float = C_MAX) -> float:
    """Normalised-usage-entropy curvature, clamped to ``[c_min, c_max]``."""
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError("counts must not be empty")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("counts must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("counts sum to zero")
    probabilities = values / total
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * positive.log()).sum())
    normaliser = math.log(k) if k > 1 else 1.0
    normalised = entropy / normaliser
    if not math.isfinite(normalised):
        raise ValueError("non-finite normalised entropy")
    curvature = c_min + (c_max - c_min) * (1.0 - min(max(normalised, 0.0), 1.0))
    return float(min(max(curvature, c_min), c_max))


def current_curvature() -> float:
    """The curvature every assignment closure reads at call time."""
    return float(_CURRENT_CURVATURE)


def set_curvature(value: float) -> float:
    """Set the module-level curvature after validating it."""
    global _CURRENT_CURVATURE
    if not math.isfinite(float(value)):
        raise ValueError("curvature must be finite")
    _CURRENT_CURVATURE = float(min(max(float(value), C_MIN), C_MAX))
    return float(_CURRENT_CURVATURE)


def update_curvature_from_tokens(tokens, k: int = CODEBOOK_K) -> float:
    """Refresh the curvature from an ``(N, 3)`` SID matrix's L1 usage histogram."""
    level1 = torch.as_tensor(tokens)[:, 0].to(torch.long)
    counts = torch.bincount(level1, minlength=k)
    return set_curvature(curvature_from_counts(counts, k))


def install_adaptive_assignment(layer, level: int) -> None:
    """Geodesic nearest-codeword assignment whose curvature is read at call time.

    A closure over the module-level curvature is used instead of a captured float
    precisely so the value can change without reinstalling the patch. ``indices``
    is replaced by the plain argmin of those distances, which is the same rule
    ``scripts/hyp_losses.install_nearest_assignment`` applies.
    """
    if getattr(layer, "_adaptive_curvature_active", False):
        return

    def distances(self, latent):
        c = current_curvature()
        points = encode_points(latent, c)
        codes = encode_points(self.get_code_embs(), c)
        return pair_poincare_distance(points, codes, c)

    def indices(self, distance_matrix, infer_use_sk):
        return distance_matrix.argmin(dim=-1)

    layer._distances = MethodType(distances, layer)
    layer._indices = MethodType(indices, layer)
    layer._adaptive_curvature_active = True
    layer._adaptive_curvature_level = int(level)
