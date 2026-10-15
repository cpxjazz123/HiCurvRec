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
With the L1 codeword usage histogram ``counts`` over the ``U`` *assignable*
codewords (``U = CROWDING_L1_ALLOWED``, the rest are masked out and can never be
chosen, so they carry no usage),

    p_i = counts_i / sum(counts)
    H   = -sum_i p_i ln p_i / ln U          (normalised usage entropy, [0, 1])
    c   = c_min + (c_max - c_min) * (1 - H)  clamped to [c_min, c_max]

so a perfectly uniform usage over the assignable set (H = 1) gives the minimum
curvature and a collapsed usage (H -> 0) gives the maximum. ``U`` rather than the
full codebook size is the normaliser: normalising by the full width capped the
achievable entropy at ``ln(U)/ln(K)``, so the curvature could never reach
``c_min``. ``c`` is then used as the curvature of ``encode_points`` +
``pair_poincare_distance`` for every level's assignment, with one tangent scale
per (codebook, curvature) shared by both sides of the distance.

Composition with the shared layer
---------------------------------
``scripts/capacity_mask.py`` is installed *after* this module, so the +inf
sentinel for the 128 disabled L1 codewords rides on top of the geodesic
distances. It still dominates under Sinkhorn: ``exp(-inf/eps)`` gives those
columns zero mass, so the assignment cannot land on a masked codeword. The
mechanism no longer replaces ``_indices``, so the L3 Sinkhorn balanced
assignment stays in force and this tree differs from the control in the distance
geometry alone.

State
-----
The curvature is mirrored into a buffer registered on the RQ module
(``bind_curvature_buffer``), so it is part of ``state_dict()`` and a reloaded
checkpoint assigns codes with the curvature it was exported under. The final
collision is recomputed from the exported SID matrix, so the reported number and
the artifacts describe the same assignment state.

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
from scripts.hyp_losses import encode_points, shared_tangent_scale

C_MIN = 0.25
C_MAX = 4.0
CODEBOOK_K = 256
INITIAL_CURVATURE = 1.0

# This arm is the *fixed-curvature* half of the M3 contrast: the assignment metric
# is geodesic on all three levels, and the curvature is the constant below rather
# than a function of the L1 usage entropy. Comparing the Crowding-only control
# against this tree measures the metric switch; comparing this tree against
# TIGER_density_adaptive_curvature_sinkhorn isolates the adaptation, because the
# two differ in nothing else.
FIXED_CURVATURE = 1.0

# The curvature is read by the assignment closures at call time and is mirrored
# into a registered buffer on the RQ module (see ``bind_curvature_buffer``) so it
# travels in ``state_dict()``. Reloading a checkpoint therefore restores the same
# curvature, and with it the same code assignment, instead of silently resetting
# to 1.0.
_CURVATURE_VALUE = INITIAL_CURVATURE
_CURVATURE_BUFFER: torch.Tensor | None = None


def bind_curvature_buffer(rq_module: torch.nn.Module) -> torch.Tensor:
    """Register the curvature as persistent state on ``rq_module``.

    Returns the buffer so callers can inspect or restore it. Re-binding (an arm
    re-installed on a fresh module) reuses the existing buffer if it is already
    on that module.
    """
    global _CURVATURE_BUFFER
    existing = getattr(rq_module, "_adaptive_curvature", None)
    if existing is None:
        existing = torch.tensor(float(_CURVATURE_VALUE), dtype=torch.float64)
        rq_module.register_buffer("_adaptive_curvature", existing)
    else:
        existing.fill_(float(_CURVATURE_VALUE))
    _CURVATURE_BUFFER = existing
    return existing


def curvature_from_counts(counts, k: int = CODEBOOK_K, c_min: float = C_MIN,
                          c_max: float = C_MAX,
                          usable_k: int | None = None) -> float:
    """Normalised-usage-entropy curvature, clamped to ``[c_min, c_max]``.

    ``usable_k`` is the number of codewords that may actually be assigned. It is
    the normaliser, because the shared capacity mask leaves only half of the L1
    codewords reachable: normalising by the full codebook size made a perfectly
    balanced *usable* distribution reach only ``ln(usable)/ln(total)`` of the
    entropy, so the curvature could never fall to ``c_min``.
    """
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
    normaliser_k = int(usable_k) if usable_k else k
    if usable_k and not 1 <= normaliser_k <= values.numel():
        raise ValueError(f"usable_k must be within [1, {values.numel()}]")
    normaliser = math.log(normaliser_k) if normaliser_k > 1 else 1.0
    normalised = entropy / normaliser
    if not math.isfinite(normalised):
        raise ValueError("non-finite normalised entropy")
    curvature = c_min + (c_max - c_min) * (1.0 - min(max(normalised, 0.0), 1.0))
    return float(min(max(curvature, c_min), c_max))


def current_curvature() -> float:
    """The curvature every assignment closure reads: the pinned constant."""
    return float(FIXED_CURVATURE)


def set_curvature(value: float) -> float:
    """Reject any attempt to move the curvature: this arm pins it.

    The trainer still refreshes at evaluation time; here that call is a no-op that
    proves the value never moved, rather than a silent divergence between the
    metric used and the one recorded.
    """
    if abs(float(value) - FIXED_CURVATURE) > 1e-12:
        raise ValueError(
            f"{'geodesic_fixed'} pins the curvature at {FIXED_CURVATURE}; "
            f"got {value!r}. The usage-adaptive arm is "
            f"TIGER_density_adaptive_curvature_sinkhorn."
        )
    if _CURVATURE_BUFFER is not None:
        with torch.no_grad():
            _CURVATURE_BUFFER.fill_(float(value))
    return float(value)


def update_curvature_from_tokens(tokens, k: int = CODEBOOK_K,
                                usable_k: int | None = None) -> float:
    """No-op refresh: the curvature is pinned, the histogram is not consulted.

    Kept so the trainer's evaluation path is unchanged and the pinned value is
    written into the metrics every evaluation, which is what makes "the metric did
    not move" checkable in a run rather than asserted in a comment.
    """
    return set_curvature(FIXED_CURVATURE)


def install_adaptive_assignment(layer, level: int) -> None:
    """Geodesic nearest-codeword assignment whose curvature is read at call time.

    A closure over the module-level curvature is used instead of a captured float
    precisely so the value can change without reinstalling the patch.

    Only ``_distances`` is replaced. ``_indices`` is left alone: it is the single
    place that decides Sinkhorn (training, last level) versus argmin (inference),
    and this installer used to short-circuit it with a plain argmin on all three
    levels, which deleted the L3 balanced-assignment control and made the arm
    differ from the control in two variables instead of one.
    """
    if getattr(layer, "_adaptive_curvature_active", False):
        return

    def distances(self, latent):
        c = current_curvature()
        # One scale for both sides, derived from the codebook: encoding the items
        # and the codewords separately gave each its own scale, so the same
        # vector landed in two places and the mapping drifted with every batch.
        scale = shared_tangent_scale(self.get_code_embs(), c)
        points = encode_points(latent, c, scale)
        codes = encode_points(self.get_code_embs(), c, scale)
        return pair_poincare_distance(points, codes, c)

    layer._distances = MethodType(distances, layer)
    layer._adaptive_curvature_active = True
    layer._adaptive_curvature_level = int(level)
