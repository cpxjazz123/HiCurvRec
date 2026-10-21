"""Behaviour-conditioned directed residual-displacement codebook loss."""

import torch


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Match level-one codeword displacement to detached residual displacement.

    Assignments are inferred from the supplied codebook using nearest-codeword
    Euclidean distance. The observed source-to-target displacement also
    provides the behaviour similarity weight, so no undeclared inputs are
    required.
    """
    sigma = float(parameters.get("sigma_beh", 1.0))
    weight = float(parameters.get("lambda_res", 1e-5))

    source = source.to(device=codebook.device, dtype=codebook.dtype)
    target = target.to(device=codebook.device, dtype=codebook.dtype)
    valid = mask.to(device=codebook.device, dtype=codebook.dtype).reshape(-1)
    if not torch.any(valid > 0):
        return codebook.sum() * 0.0
    source_assignment = torch.cdist(source.detach(), codebook.detach()).argmin(dim=-1)
    target_assignment = torch.cdist(target.detach(), codebook.detach()).argmin(dim=-1)
    displacement = (
        codebook[target_assignment] - codebook[source_assignment]
        - (target.detach() - source.detach())
    )
    behaviour_delta = target.detach() - source.detach()
    q = torch.exp(-behaviour_delta.square().sum(dim=-1) / (2.0 * sigma * sigma))
    per_pair = displacement.square().sum(dim=-1)
    return weight * (per_pair * q * valid).sum() / valid.sum().clamp_min(1.0)
