"""M2: Behaviour-Conflict Hyperbolic Separation.

Mechanism (the single variable relative to TIGER_CROWDING_CTRL)
--------------------------------------------------------------
One additive loss term that pushes each anchor's geodesic distance to its true
successor below its geodesic distance to any *admissible* negative, by a fixed
margin, measured inside the Poincare ball. Under L1 crowding the behaviour term's
batched cross-entropy is dominated by the crowded positive/negative pool; this
term is a per-row margin objective on the same rows, so it separates the pairs
that share a crowded prefix instead of only re-weighting the pool.

Math
----
Per level ``l`` with residuals ``r`` mapped into the curvature-``c`` ball by
``geom_losses.map_to_ball``:

    d_pos(i) = d_P(B(r_i^anc), B(r_i^suc))
    d_neg(i) = min_{j != i, j admissible} d_P(B(r_i^anc), B(r_j^suc))
    L = mean_i softplus(d_pos(i) - d_neg(i) + margin)

"Admissible" reuses the behaviour term's masking semantics exactly (see
``train_rqvae._contrastive_ce``): a candidate that repeats the anchor's own target
is not a negative, and -- when the false-negative mask is on -- neither is a
candidate that is itself a known successor of that anchor. The hardest admissible
negative is therefore selected under the same rules the existing contrastive term
uses. The term is the mean over the three residual levels.

Composition with the shared layer
---------------------------------
Purely loss-side: it reads the residual trace the trainer already computes and
never touches ``_distances``, so the L1 capacity mask is untouched by
construction. No new sampler, no change to ``behavior_contrastive_loss``, no
change to codebooks or to the Stage3 vocabulary.

What is completely unchanged relative to the control
---------------------------------------------------
L1 capacity mask and diagnostics, pair sampling, ``behavior_contrastive_loss``,
the ramp (the term is active exactly where the behaviour term is active, i.e.
epochs above ``RAMP_START_EPOCH``), codebook sizes/parameters, optimiser,
schedule, seed, epochs, evaluation interval.

Fixed parameters (no sweep): ``weight = 0.005``, ``c = 1.0``, ``margin = 0.05``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from geom_losses import map_to_ball, poincare_distance

CONFLICT_WEIGHT = 0.005
CONFLICT_CURVATURE = 1.0
CONFLICT_MARGIN = 0.05


def admissible_mask(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    successor_indicator=None,
) -> torch.Tensor:
    """True where a candidate must not be used as a negative for that anchor.

    Same construction as ``train_rqvae._contrastive_ce``: repeat targets are
    blocked, plus the anchor's other real successors when the false-negative mask
    is enabled. The diagonal is *not* blocked here -- the caller excludes the
    positive separately -- because this mask is also used as-is for inspection.
    """
    blocked = target_ids[:, None].eq(target_ids[None, :])
    blocked.fill_diagonal_(False)
    if successor_indicator is not None:
        rows = successor_indicator[source_ids.cpu().numpy()]
        extra = torch.from_numpy(
            np.asarray(rows[:, target_ids.cpu().numpy()].todense())
        ).to(device=blocked.device, dtype=torch.bool)
        extra.fill_diagonal_(False)
        blocked = blocked | extra
    return blocked


def hyperbolic_conflict_loss(
    residuals: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    successor_indicator=None,
    c: float = CONFLICT_CURVATURE,
    margin: float = CONFLICT_MARGIN,
) -> torch.Tensor:
    """Margin separation of true successors from the hardest admissible negatives."""
    batch_size = int(source_ids.shape[0])
    valid = source_ids.ne(target_ids)
    if not bool(valid.any()):
        return residuals.sum() * 0.0
    blocked = admissible_mask(source_ids, target_ids, successor_indicator)
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=blocked.device)
    blocked = blocked | diagonal

    losses: list[torch.Tensor] = []
    for level in range(residuals.shape[0]):
        sources = map_to_ball(residuals[level, :batch_size], c)
        candidates = map_to_ball(residuals[level, batch_size:], c)
        distances = poincare_distance(sources[:, None, :], candidates[None, :, :], c)
        positive = distances.diagonal()
        negative = distances.masked_fill(blocked, float("inf")).min(dim=1).values
        losses.append(F.softplus(positive - negative + float(margin))[valid].mean())
    return torch.stack(losses).mean()
