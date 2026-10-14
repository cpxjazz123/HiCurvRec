"""M1: Hyperbolic Fan-out Quantization.

Mechanism (the single variable relative to TIGER_CROWDING_CTRL)
--------------------------------------------------------------
Levels 2 and 3 are assigned *parent-conditioned*: a candidate child codeword is
penalised when it continues in the same direction as the anchor, measured from
the anchor's own parent codeword. Under L1 crowding, items that share an L1
prefix are forced apart inside that prefix instead of piling onto one child.

Math
----
With ``x`` the anchor's residual entering the level, ``p`` the codeword its
parent level assigned to that row (L1 for level 2, L2 for level 3), and ``c`` a
candidate codeword of the level:

    u_x = (x - p) / ||x - p||        direction of the anchor inside its parent
    u_c = (c - p) / ||c - p||        direction of the candidate inside that parent

    d_fanout(x, c) = d_P(x, c) + strength * (1 - cos(u_x, u_c))

``d_P`` is the curvature-``c`` Poincare distance between ``encode_points(x, c)``
and ``encode_points(c, c)`` -- the same geodesic used by the geodesic-assignment
patch in ``scripts/hyp_losses.py`` -- so the mechanism is a hyperbolic one and
composes with that patch instead of replacing it. ``cos`` is expanded so no
``(N, K, d)`` tensor is materialised:

    <x - p, c - p> = <x, c> - <x, p> - <p, c> + <p, p>
    ||c - p||      = sqrt(||c||^2 - 2<p, c> + ||p||^2)

The penalty costs one extra ``(N, d) @ (d, K)`` product per level. It is
computed from the *raw* residual and the *raw* codewords (the angular term is a
Euclidean direction comparison, exactly as specified); only the distance term is
mapped into the ball. Parent codewords are detached: the term changes which
child is selected, never the codebook gradient, which stays exactly the
commitment + reconstruction objective the control has.

Composition with the shared layer
---------------------------------
``scripts/capacity_mask.py`` is installed *after* this module, so the wrapper it
puts on L1 sits on top and the +inf sentinel it writes for the 128 disabled L1
codewords always wins the argmin; this mechanism only ever re-ranks the
codewords that the capacity constraint leaves selectable.

What is completely unchanged relative to the control
---------------------------------------------------
codebook sizes and parameters, level-1 assignment, all losses (reconstruction,
commitment, behaviour contrastive), pair sampling, optimiser, schedule, seed,
epochs, evaluation interval.

Fixed parameters (no sweep): ``c = 1.0``, ``strength = 0.5``.
"""
from __future__ import annotations

from types import MethodType

import torch

from hyperbolic import pair_poincare_distance
from scripts.hyp_losses import encode_points

FANOUT_C = 1.0
FANOUT_STRENGTH = 0.5
COS_EPS = 1e-12


def _angular_penalty(latent: torch.Tensor, codes: torch.Tensor,
                    parent_codes: torch.Tensor) -> torch.Tensor:
    """``1 - cos`` between ``(latent - parent)`` and every ``(code - parent)``.

    ``latent``/``parent_codes`` are ``(N, d)`` row-aligned; ``codes`` is
    ``(K, d)``. Returns ``(N, K)`` with the row's own parent used for every
    candidate of that row.
    """
    x_dot_c = latent @ codes.t()
    x_dot_p = (latent * parent_codes).sum(dim=-1)
    p_dot_p = parent_codes.pow(2).sum(dim=-1)
    p_dot_c = parent_codes @ codes.t()

    numerator = x_dot_c - x_dot_p.unsqueeze(1) - p_dot_c + p_dot_p.unsqueeze(1)
    offset_norm = (latent - parent_codes).norm(dim=-1)
    code_norm = (
        codes.pow(2).sum(dim=-1).unsqueeze(0) - 2.0 * p_dot_c + p_dot_p.unsqueeze(1)
    ).clamp_min(0.0).sqrt()
    denominator = (offset_norm.unsqueeze(1) * code_norm).clamp_min(COS_EPS)
    return 1.0 - (numerator / denominator).clamp(-1.0, 1.0)


def _fanout_distance(latent: torch.Tensor, codes: torch.Tensor,
                    parent_codes: torch.Tensor, c: float = FANOUT_C,
                    strength: float = FANOUT_STRENGTH) -> torch.Tensor:
    """Geodesic distance plus the parent-conditioned angular penalty."""
    points = encode_points(latent, c)
    code_points = encode_points(codes, c)
    geodesic = pair_poincare_distance(points, code_points, c)
    return geodesic + float(strength) * _angular_penalty(latent, codes, parent_codes)


def install_fanout(child_layer, parent_layer, level: int, c: float = FANOUT_C,
                   strength: float = FANOUT_STRENGTH) -> None:
    """Make ``child_layer``'s assignment parent-conditioned on ``parent_layer``.

    ``RQLayer.forward`` hands each level the previous level's indices through
    ``VQLayer.forward(..., parent_indices=...)``; the layer stores them for the
    duration of its distance computation and the wrapper installed here reads
    them. With no parent indices -- which is how ``init_codebook`` and every
    non-fan-out caller invoke the layer -- the wrapper delegates to the distance
    function it wrapped, so those paths are bit-identical to the control's.
    """
    if getattr(child_layer, "_fanout_active", False):
        return
    original = child_layer._distances

    def distances(self, latent):
        parents = getattr(self, "_fanout_parent_indices", None)
        if parents is None:
            return original(latent)
        parent_codes = parent_layer.get_code_embs().detach()[parents]
        return _fanout_distance(latent, self.get_code_embs(), parent_codes, c, strength)

    child_layer._distances = MethodType(distances, child_layer)
    child_layer._fanout_active = True
    child_layer._fanout_level = int(level)
    child_layer._fanout_strength = float(strength)
    child_layer._fanout_curvature = float(c)


def _trace_with_fanout(module, embeddings: torch.Tensor, with_tokens: bool = False):
    """Per-level pre-quantization residuals, parent-conditioned like training.

    ``train_rqvae._residuals_with_trace`` walks the quantizer stack itself, so it
    has to hand the parent indices down the same way ``RQLayer.forward`` does;
    otherwise the behaviour contrastive term would be trained on residuals taken
    from a different assignment path than the one the mechanism changes. Set
    ``with_tokens`` to also return the per-level index stack (level-major).
    """
    residual = module.encoder(embeddings)
    traces = [residual]
    tokens = []
    parents = None
    for level, vq_layer in enumerate(module.rq.vq_layers):
        quant, _, _, indices = vq_layer(residual, parent_indices=parents if level else None)
        tokens.append(indices)
        residual = residual - quant
        traces.append(residual)
        parents = indices
    stacked = torch.stack(traces[: module.rq.codebook_num], dim=0)
    if with_tokens:
        return stacked, torch.stack(tokens, dim=0)
    return stacked
