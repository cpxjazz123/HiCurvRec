"""Same-centre relative curvature ranking for Issue #267.

Pure tensor function: it never reads the graph, assets or model state. The target
is detached by the caller. Only unordered within-centre pairs with a non-tied
target difference enter the mean.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TAU = 1.0516836255159168
TIE_EPS = 1e-6
RELATIVE_SHARE = 0.2


def relative_rank_loss(predicted: torch.Tensor, target: torch.Tensor, centres,
                       *, tau: float = TAU, tie_eps: float = TIE_EPS) -> dict:
    """Return the softplus sign-ranking loss and auditable pair counts.

    `predicted[i]` and `target[i]` are the predicted and frozen curvature values
    for the same sample row. Pairs are upper-triangular, same-centre and have
    `abs(target_i-target_j) > tie_eps`.
    """
    if predicted.ndim != 1 or target.ndim != 1:
        raise ValueError("predicted and target must be one-dimensional")
    if predicted.shape != target.shape:
        raise ValueError("predicted and target shapes must match")
    if not math.isfinite(float(tau)) or float(tau) <= 0:
        raise ValueError("tau must be finite and positive")
    if not math.isfinite(float(tie_eps)) or float(tie_eps) < 0:
        raise ValueError("tie_eps must be finite and nonnegative")
    if predicted.numel() != len(centres):
        raise ValueError("centre identities must match the sample")
    if predicted.numel() < 2:
        zero = predicted.sum() * 0
        return {"loss": zero, "all_pairs": 0, "valid_pairs": 0, "tied_pairs": 0,
                "cross_centre_pairs": 0}
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("nonfinite predicted/target curvature")
    device = predicted.device
    centre = torch.as_tensor(centres, dtype=torch.long, device=device)
    if centre.ndim != 1 or centre.numel() != predicted.numel():
        raise ValueError("centres must be a one-dimensional identity vector")
    left, right = torch.triu_indices(predicted.numel(), predicted.numel(), offset=1,
                                     device=device)
    delta_target = (target[left] - target[right]).to(predicted.dtype)
    delta_predicted = predicted[left] - predicted[right]
    same = centre[left] == centre[right]
    finite = torch.isfinite(delta_target) & torch.isfinite(delta_predicted)
    tied = same & (delta_target.abs() <= float(tie_eps))
    valid = same & finite & (~tied)
    valid_pairs = int(valid.sum())
    tied_pairs = int(tied.sum())
    cross = int((~same).sum())
    if valid_pairs:
        sign = torch.sign(delta_target[valid])
        loss = F.softplus(-sign * delta_predicted[valid] / float(tau)).mean()
    else:
        loss = predicted.sum() * 0
    return {"loss": loss, "all_pairs": int(left.numel()), "valid_pairs": valid_pairs,
            "tied_pairs": tied_pairs, "cross_centre_pairs": cross}
