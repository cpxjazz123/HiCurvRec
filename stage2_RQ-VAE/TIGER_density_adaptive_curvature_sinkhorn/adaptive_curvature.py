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

# The curvature is instance-local state: a registered buffer on each patched layer,
# read by that layer's own assignment closure. It used to be a module-level pointer,
# so binding a second model in the same process silently changed the curvature the
# first model assigned with, and re-binding overwrote the buffer from that global.


def ensure_curvature_buffers(module) -> list:
    """Register ``_adaptive_curvature`` on each layer of ``module`` if absent.

    Idempotent and non-destructive: an existing buffer is left untouched, so
    installing twice cannot reset a curvature that was already trained. Returns the
    buffers the caller writes to when the curvature changes.
    """
    layers = getattr(module, "vq_layers", None)
    if layers is None:
        layers = module.rq.vq_layers
    buffers = []
    for layer in layers:
        buffer = getattr(layer, "_adaptive_curvature", None)
        if buffer is None:
            buffer = torch.tensor(float(INITIAL_CURVATURE), dtype=torch.float64)
            layer.register_buffer("_adaptive_curvature", buffer)
        buffers.append(buffer)
    return buffers


def current_curvature(layer=None, buffers=None) -> float:
    """The curvature this layer assigns with, or the first bound buffer's."""
    if layer is not None and hasattr(layer, "_adaptive_curvature"):
        return float(layer._adaptive_curvature.item())
    if buffers:
        return float(buffers[0].item())
    return float(INITIAL_CURVATURE)


def set_curvature(value: float, buffers) -> float:
    """Validate the curvature and write it into this model's own buffers."""
    if not math.isfinite(float(value)):
        raise ValueError("curvature must be finite")
    clamped = float(min(max(float(value), C_MIN), C_MAX))
    for buffer in buffers:
        with torch.no_grad():
            buffer.fill_(clamped)
    return clamped



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


def update_curvature_from_tokens(tokens, buffers, k: int = CODEBOOK_K,
                                usable_k: int | None = None) -> float:
    """Refresh the curvature from an ``(N, 3)`` SID matrix's L1 usage histogram.

    Only the assignable prefix contributes: a codeword the capacity mask has made
    unreachable carries no usage information, and counting it would inflate the
    histogram's width without adding any mass.
    """
    level1 = torch.as_tensor(tokens)[:, 0].to(torch.long)
    limit = int(usable_k) if usable_k else int(k)
    if limit > k:
        raise ValueError("usable_k cannot exceed the codebook size")
    level1 = level1[level1 < limit]
    counts = torch.bincount(level1, minlength=limit)[:limit]
    return set_curvature(curvature_from_counts(counts, limit, usable_k=limit), buffers)


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
    if not hasattr(layer, "_adaptive_curvature"):
        layer.register_buffer(
            "_adaptive_curvature",
            torch.tensor(float(INITIAL_CURVATURE), dtype=torch.float64),
        )

    def distances(self, latent):
        # This layer's own buffer: two models in one process must not be able
        # to change each other's curvature.
        c = float(self._adaptive_curvature.item())
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
