"""E1: conditional behaviour supervision on the cumulative L1+L2 quantised representation.

The measurement that motivates this. On the baseline, 7.94% of behaviour positive
pairs share their L1 codeword against 0.50% of negatives, a sixteen-fold enrichment,
so L1 genuinely cannot separate that slice of the real behaviour signal. Behaviour AUC
by level, split on whether L1 can tell the pair apart:

  residual     all      L1-ambiguous   L1-distinguishable
  L1           0.7767   0.9941          0.7579
  L2           0.5218   0.6880          0.5074
  L3           0.5056   0.6735          0.4911

Two things follow. The L1 *representation* is not the problem -- on ambiguous pairs its
AUC is 0.9941, because items that land on the same codeword are precisely the ones the
encoder already placed together. And L2's behaviour contribution is concentrated
exactly in the ambiguous slice: 0.6880 there against 0.5074, which is chance, on the
distinguishable ones.

So the loss is applied where it is missing and nowhere else: only to pairs whose L1
codeword matches, only on ``q2 = e1 + e2`` rather than on the L2 residual, and never to
the codebooks.

The straight-through term is what keeps the forward value exact while routing the
gradient to one place. ``latent + (target - latent).detach()`` evaluates to ``target``,
its gradient with respect to the encoder latent is the identity, and the embeddings it
reads are detached, so no codebook receives anything. The quantiser's own forward is
untouched: this module never calls it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Weights the plan asks for. Zero is the control: the arm exists and every line of the
# code path runs, the term simply contributes nothing.
ARM_WEIGHT = {
    "L2COND_CTRL": 0.0,
    "L2COND_W001": 0.001,
    "L2COND_W005": 0.005,
    "L2COND_W01": 0.01,
}
# Ranking form and temperature follow the family's existing behaviour term so that the
# only thing being varied is which representation is scored.
TEMPERATURE = 0.07


def cumulative_two_level(
    latent: torch.Tensor,
    tokens: torch.Tensor,
    vq_layers,
) -> torch.Tensor:
    """``q2 = e1 + e2`` in value, identity gradient into ``latent``, none into codebooks."""
    e1 = F.embedding(tokens[0], vq_layers[0].get_code_embs().detach())
    e2 = F.embedding(tokens[1], vq_layers[1].get_code_embs().detach())
    target = e1 + e2
    return latent + (target - latent).detach()


def ambiguous_mask(tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Anchor rows whose L1 codeword is the same as their behaviour target's."""
    return tokens[0, :batch_size] == tokens[0, batch_size:]


def l2_conditional_ranking_loss(
    latent: torch.Tensor,
    tokens: torch.Tensor,
    vq_layers,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy ranking on ``q2``, for L1-ambiguous pairs only.

    The form is the same one L1's own behaviour term uses, so the loss shape, the
    temperature, the candidate set and the duplicate mask are all held fixed; what
    changes is the representation being scored and which rows are scored at all.
    """
    device = latent.device
    batch_size = int(source_ids.shape[0])
    ambiguous = ambiguous_mask(tokens, batch_size)
    rows: dict[str, float] = {
        "l2cond_ambiguous_fraction": float(ambiguous.float().mean()),
        "l2cond_rows": float(int(ambiguous.sum())),
    }
    if weight == 0.0 or not bool(ambiguous.any()):
        return torch.zeros((), device=device), rows

    q2 = cumulative_two_level(latent, tokens, vq_layers)
    ambiguous_rows = ambiguous.nonzero(as_tuple=True)[0]
    anchor_rows = q2[:batch_size][ambiguous_rows]
    candidates = q2[batch_size:]

    # Same candidate set, same duplicate mask and the same positive indexing as
    # _contrastive_ce, because the batch is laid out as [sources | targets] and the
    # true successor of anchor i is candidate i. Nothing about the negatives differs
    # from what L1's own term sees; only the representation and the row filter do.
    logits = -torch.cdist(anchor_rows, candidates, p=2) / TEMPERATURE
    duplicate = target_ids[:, None].eq(target_ids[None, :])
    # Select the rows of the duplicate mask belonging to the ambiguous anchors, so the
    # block is (ambiguous rows x candidates). The true successor of anchor r is
    # candidate r, because the batch is laid out as [sources | targets]. Once the rows
    # have been filtered, that column is no longer the block's diagonal, so the block's
    # own diagonal must NOT be what gets un-masked: doing so scores a different item as
    # the positive while leaving the real successor masked out.
    duplicate = duplicate[ambiguous_rows].clone()
    duplicate[
        torch.arange(anchor_rows.shape[0], device=device), ambiguous_rows
    ] = False
    logits = logits.masked_fill(duplicate, -torch.inf)
    labels = ambiguous_rows
    loss = float(weight) * F.cross_entropy(logits, labels)
    rows["l2cond_loss"] = float(loss.detach())
    rows["l2cond_rows"] = float(anchor_rows.shape[0])
    return loss, rows
