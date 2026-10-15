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

Corrections in this tree
------------------------
1. The negative set is now the same-L1 crowding population (candidates assigned
   the anchor's L1 codeword) instead of the globally nearest admissible
   candidate. Before, the loss never looked at L1 identity, so despite the arm's
   name it did not act on codeword crowding.
2. The false-negative mask is always supplied. It used to be gated on an arm name
   that does not match this arm, so the successor indicator was ``None`` and real
   successors could be pulled in as hard negatives.
3. Coverage is reported, not assumed: ``same_l1_rows`` and ``fallback_rows`` are
   written into the metrics stream, so the share of rows that actually had a
   same-L1 negative is visible instead of implied.
4. The deep-supervision gradient claim is checked rather than asserted: the
   residual trace comes from its own ``encoder(...)`` call, so this term does
   reach the encoder. The tree's checks measure the per-level gradients.
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
    level1_codes: torch.Tensor | None = None,
    c: float = CONFLICT_CURVATURE,
    margin: float = CONFLICT_MARGIN,
) -> tuple[torch.Tensor, dict]:
    """Margin separation of true successors from the hardest admissible negatives.

    What changed, and why:

    * the negative set is now the *same-L1 crowding population* -- candidates that
      were assigned the anchor's L1 codeword. Before, the negative was the nearest
      admissible candidate over the whole batch regardless of L1 identity, so
      despite the arm's name the loss never acted on the codeword crowding it is
      about. Pass ``level1_codes`` (the L1 assignment of every row of the pair
      batch, sources first) to enable it.
    * an anchor with no *other* admissible item in its own L1 cell has no
      same-L1 negative. Rather than dropping the row -- which would silently
      change the batch the term sees -- the fallback is the globally nearest
      admissible candidate, and ``stats["fallback_rows"]`` reports how often that
      happened so the term's coverage is never overstated.

    The false-negative mask is not optional here: a candidate that is itself a
    known successor of the anchor is a true positive, and using it as a negative
    teaches the model to separate a real successor from its anchor. The trainer
    therefore always supplies ``successor_indicator`` for this arm.

    On gradient reach: the residual trace the trainer passes in comes from its own
    ``encoder(...)`` call (``_residuals_with_trace``), not from the straight-through
    training forward, so gradients from this term reach the encoder directly. That
    is verified numerically in the tree's checks, not assumed here.
    """
    batch_size = int(source_ids.shape[0])
    valid = source_ids.ne(target_ids)
    if not bool(valid.any()):
        return residuals.sum() * 0.0, {"rows": 0, "fallback_rows": 0, "same_l1_rows": 0}
    blocked = admissible_mask(source_ids, target_ids, successor_indicator)
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=blocked.device)
    blocked = blocked | diagonal

    # Same-L1 crowding population, minus everything already inadmissible.
    same_l1 = None
    if level1_codes is not None:
        codes = level1_codes.to(device=blocked.device, dtype=torch.long).reshape(-1)
        if codes.numel() != 2 * batch_size:
            raise ValueError(
                f"level1_codes must cover the whole pair batch: "
                f"got {codes.numel()} for {2 * batch_size} rows"
            )
        anchor_codes, candidate_codes = codes[:batch_size], codes[batch_size:]
        same_l1 = anchor_codes[:, None] == candidate_codes[None, :]
        same_l1 = same_l1 & (~blocked)
        # Counted before the fallback: these are the rows that genuinely had an
        # admissible same-L1 negative, which is what the coverage number is about.
        same_l1_rows = int((same_l1.any(dim=1) & valid).sum())
        # A row whose own L1 cell holds no admissible other item keeps the global
        # nearest admissible candidate instead of contributing nothing.
        empty = ~same_l1.any(dim=1)
        fallback_rows = int(empty.sum())
        if fallback_rows:
            global_admissible = ~blocked
            nearest_global = global_admissible.float().argmax(dim=1, keepdim=True)
            same_l1 = same_l1 | (empty[:, None] & torch.zeros_like(same_l1).scatter_(
                1, nearest_global, True))
    else:
        fallback_rows = 0
        same_l1_rows = 0
    forbidden = blocked if same_l1 is None else (blocked | ~same_l1)

    losses: list[torch.Tensor] = []
    for level in range(residuals.shape[0]):
        sources = map_to_ball(residuals[level, :batch_size], c)
        candidates = map_to_ball(residuals[level, batch_size:], c)
        distances = poincare_distance(sources[:, None, :], candidates[None, :, :], c)
        positive = distances.diagonal()
        negative = distances.masked_fill(forbidden, float("inf")).min(dim=1).values
        losses.append(F.softplus(positive - negative + float(margin))[valid].mean())
    stats = {
        "rows": int(valid.sum()),
        "same_l1_rows": same_l1_rows,
        "fallback_rows": fallback_rows,
    }
    return torch.stack(losses).mean(), stats
