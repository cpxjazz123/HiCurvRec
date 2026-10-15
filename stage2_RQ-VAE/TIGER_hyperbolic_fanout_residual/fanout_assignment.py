"""M1: Hyperbolic Fan-out Quantization.

Mechanism (the single variable relative to TIGER_CROWDING_CTRL)
--------------------------------------------------------------
A candidate child codeword that continues in the same direction as the anchor is
made *farther*, so a row is not drawn onto the child it already points at.

Math -- this is the formula the code executes
--------------------------------------------
With ``r`` the anchor's residual entering the level (the parent level's codeword has
already been subtracted), ``C`` the level's candidate codewords and ``s`` the
strength:

    u_r = r / ||r||                direction of the anchor, in residual space
    u_c = C / ||C||                direction of the candidate, in residual space

    d_fanout(r, C) = d_P(r, C) - s * (1 - cos(u_r, u_c))

``d_P`` is the curvature-``c`` Poincare distance between ``encode_points(r, c)`` and
``encode_points(C, c)`` -- the same geodesic the assignment patch in
``scripts/hyp_losses.py`` uses, so the mechanism composes with it rather than
replacing it -- and both sides share one tangent scale, derived from the codebook.

Sign: ``1 - cos`` is smallest when the directions agree, and the term is
*subtracted*, so an aligned child is the farthest candidate. Adding it would pull
the rows of one parent onto a single child, which is the opposite of a fan-out.
The same choice is pinned by a test that measures the margin against the aligned
child with and without the term.

How the parent conditioning works: **implicitly, through the residual.**
``install_fanout`` checks only that the layer carries ``_fanout_parent_indices``;
``_fanout_distance`` never reads a parent index or any other row of the same
parent bucket. The residual ``r`` exists only for items that landed on that
parent, so the conditioning is carried by the geometry of ``r`` rather than by an
explicit sibling comparison.

What this does *not* guarantee -- measured, not asserted
--------------------------------------------------------
The term is an itemwise angular re-ranking, not an occupancy constraint, and
``scripts/check_within_parent.py`` measures what that is worth inside one parent
bucket (12 candidate children, 48 rows):

  * identical residuals            -> 1 child used, entropy 0.0
  * near-identical residuals       -> 1 child used, entropy 0.0 (0.01 perturbation)
  * separable residuals            -> 8 of 12 children, entropy 0.830
  * the same rows without the term -> 10 of 12 children, entropy 0.846

So the arm does **not** deliver within-parent spread: it slightly *reduces* child
occupancy against the plain geodesic (8 vs 10 of 12) while still making an aligned
child the farthest candidate. Both facts are recorded because the second is easy to
mistake for success if only the aligned-child margin is measured.

Guaranteeing sibling separation would need an explicit occupancy or repulsion
term computed against the other rows of the same bucket -- a different mechanism,
not a fix to this one -- and it is not claimed here.

Cost and gradients
------------------
The penalty costs one ``(N, d) @ (d, K)`` product per level and is computed from
the raw residual and raw codewords; only the distance term is mapped into the
ball. The codebook gradient stays exactly the commitment plus reconstruction
objective the control has.

Composition with the shared layer
---------------------------------
``scripts/capacity_mask.py`` is installed *after* this module, so the wrapper it
puts on L1 sits on top and the +inf sentinel it writes for the 128 disabled L1
codewords keeps them unreachable.
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
