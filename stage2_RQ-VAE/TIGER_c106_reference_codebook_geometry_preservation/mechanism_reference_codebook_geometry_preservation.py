"""Preserve pairwise geometry between valid source and target vectors."""

import torch


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Return the masked, weighted pairwise-distance preservation penalty.

    The source distances define the reference geometry; target distances are
    penalized for deviating from it. ``codebook`` is accepted by the common
    mechanism interface but is not needed for this pairwise objective.
    """
    eps_geom = float(parameters.get("eps_geom", 1e-6))
    lambda_geom = float(parameters.get("lambda_geom", 1e-4))
    valid = mask.to(device=source.device, dtype=torch.bool)
    valid_pairs = valid[:, None] & valid[None, :]
    upper = torch.triu(
        torch.ones_like(valid_pairs, dtype=torch.bool), diagonal=1
    )
    selected = valid_pairs & upper

    source_distance = torch.cdist(source, source)
    target_distance = torch.cdist(target, target)
    weights = 1.0 / (source_distance.square() + eps_geom)
    pair_penalty = weights * (target_distance - source_distance).square()
    # A zero-valued, differentiable result when fewer than two valid examples
    # are present; otherwise normalize over valid unordered pairs.
    denominator = selected.sum().clamp_min(1).to(dtype=pair_penalty.dtype)
    geometry = pair_penalty.masked_select(selected).sum() / denominator
    return lambda_geom * geometry
