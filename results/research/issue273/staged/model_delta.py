"""Issue #273: same-centre difference protection for the sectional-curvature prediction.

The pointwise curvature loss is blind to a common offset: adding the same constant to every
prediction changes the pointwise loss but leaves every same-centre difference unchanged.  This
module adds the term that is sensitive to the differences while staying blind to the offset, so the
two terms are complementary (see ``results/research/issue273/CONTRACT.md`` section 3):

    L_delta = mean over the centre's pairs (t, s), t != s of ((p_t - p_s) - (y_t - y_s))^2

The per-centre normalisation by the centre's valid triplet count makes the term centre-balanced:
no centre dominates the loss because every centre is drawn once per step and contributes its own
share.  With ``e = p - y`` the numerator is ``(e_t - e_s)^2``; for independent pairs its
expectation is ``2 Var_c(e)``, i.e. the term is twice the within-centre error variance.
"""
from __future__ import annotations

import torch


def same_centre_difference_loss(predicted, target, centre_population):
    """Centre-balanced same-centre difference loss of one sampled centre.

    ``predicted`` and ``target`` are the per-triplet curvature prediction and target of the
    sampled triplets (all of them share one centre); ``centre_population`` is the number of valid
    triplets of that centre in the frozen target table.  The value is

        sum over the unordered pairs of ((p_t - p_s) - (y_t - y_s))^2
        ------------------------------------------------------------------
        centre_population * number_of_pairs

    and is exactly zero when fewer than two triplets are given.
    """
    if predicted.shape != target.shape:
        raise ValueError('predicted and target must have the same shape')
    count = int(predicted.numel())
    if count < 2:
        return predicted.new_zeros(())
    population = int(centre_population)
    if population < count:
        raise ValueError('centre population is smaller than the sampled triplets')
    difference_p = predicted[:, None] - predicted[None, :]
    difference_y = target[:, None] - target[None, :]
    squared = (difference_p - difference_y) ** 2
    index = torch.triu_indices(count, count, offset=1, device=predicted.device)
    pairs = squared[index[0], index[1]]
    return pairs.sum() / (float(population) * float(pairs.numel()))
