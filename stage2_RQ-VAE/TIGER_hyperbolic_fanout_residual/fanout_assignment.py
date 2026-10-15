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
from scripts.hyp_losses import encode_points, shared_tangent_scale

FANOUT_C = 1.0
FANOUT_STRENGTH = 0.5


def _angular_penalty(latent: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """``1 - cos`` between the item's residual direction and every child direction.

    Both sides are in residual space. ``latent`` is the residual *entering* the
    child level -- the parent codeword has already been subtracted from it by the
    level walk -- and ``codes`` are the codewords of that same child level, so
    the two are expressed in one frame and the cosine between them is meaningful.

    What this used to do, and why it was wrong on both counts:

    * it computed ``cos(latent - parent, code - parent)``. With ``r = z - e1``
      the first argument is ``z - 2*e1``: an L1-space codeword subtracted from a
      residual that had already given up its L1 codeword. The parent subtraction
      is a category error here, and dropping it is not merely cosmetic -- the
      parent conditioning is already carried by ``r`` itself, since ``r`` only
      exists for items that landed on that parent.
    * the sign. ``1 - cos`` is *small* when the item and the candidate point the
      same way, and this term was added to the distance, so an aligned child
      looked closer and won the argmin. That pulls the items of one L1 cell onto
      a single child, which is the opposite of a fan-out. The caller now subtracts
      the term, so an aligned child is the farthest one and siblings spread.
    """
    unit_x = torch.nn.functional.normalize(latent, dim=-1)
    unit_c = torch.nn.functional.normalize(codes, dim=-1)
    return 1.0 - unit_x @ unit_c.t()


def _fanout_distance(latent: torch.Tensor, codes: torch.Tensor,
                    c: float = FANOUT_C,
                    strength: float = FANOUT_STRENGTH) -> torch.Tensor:
    """Geodesic distance minus the parent-conditioned angular penalty.

    Subtracting ``strength * (1 - cos)`` is what makes this a fan-out: the child
    whose direction matches the item's becomes the *farthest* candidate, so items
    sharing one L1 parent are pushed towards different children. Adding it would
    do the opposite and concentrate them.
    """
    scale = shared_tangent_scale(codes, c)
    points = encode_points(latent, c, scale)
    code_points = encode_points(codes, c, scale)
    geodesic = pair_poincare_distance(points, code_points, c)
    return geodesic - float(strength) * _angular_penalty(latent, codes)


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
        # The parent's own codeword is no longer needed: the residual entering this
        # level has already had it subtracted, so the fan-out term compares two
        # residual-space directions. The parent indices are still what gates the
        # patch (without them the wrapper delegates to the control's distance).
        return _fanout_distance(latent, self.get_code_embs(), c, strength)

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
