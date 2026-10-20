"""Behavior-conditioned pairwise margin on continuous encoder outputs."""

import torch
import torch.nn.functional as F


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Compute the encoder-only pair margin using detached interaction affinity.

    ``affinity`` is a directed source-by-target matrix; alternatively interaction
    histories can be supplied as ``source_history`` and ``target_history``.
    The common-interface codebook argument is intentionally unused.
    """
    del codebook
    mask = mask.to(device=source.device, dtype=source.dtype)
    affinity = parameters.get("affinity")
    if affinity is None:
        source_history = parameters.get("source_history")
        target_history = parameters.get("target_history")
        if source_history is None or target_history is None:
            affinity = source.new_zeros((source.shape[0], target.shape[0]))
        else:
            a = F.normalize(source_history.detach().to(source), p=2, dim=-1)
            b = F.normalize(target_history.detach().to(target), p=2, dim=-1)
            affinity = (a @ b.transpose(-1, -2)).clamp(0.0, 1.0)
    affinity = affinity.detach().to(device=source.device, dtype=source.dtype)
    # The trainer supplies one directed (source_k, target_k) pair per row.  Keep
    # that paired interpretation instead of broadcasting a length-N vector over
    # an N x N Cartesian product.  Matrix inputs remain useful for callers that
    # explicitly request all cross-pairs.
    paired = affinity.ndim == 1
    if paired:
        if source.shape[0] != target.shape[0] or affinity.shape[0] != source.shape[0]:
            raise ValueError("1-D affinity requires equally sized paired source/target rows")
        affinity = affinity.reshape(-1)
        if mask.ndim > 1:
            mask = torch.diagonal(mask)
        else:
            mask = mask.reshape(-1)
    positive_margin = float(parameters.get("m_pos", 0.25))
    negative_margin = float(parameters.get("m_neg", 1.0))
    if paired:
        distance = (source - target).square().sum(dim=-1)
    else:
        if affinity.ndim == 1:
            affinity = affinity[:, None].expand(source.shape[0], target.shape[0])
        if mask.ndim == 1:
            mask = mask[:, None].expand(source.shape[0], target.shape[0])
        distance = (source[:, None, :] - target[None, :, :]).square().sum(dim=-1)
    positive = affinity * F.relu(distance - positive_margin).square()
    negative = (1.0 - affinity) * F.relu(negative_margin - distance).square()
    return ((positive + negative) * mask).sum() / mask.sum().clamp_min(1.0)
