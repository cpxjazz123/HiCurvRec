"""Behavior-conditioned pairwise curvature ranking for the level-1 codebook."""

import torch
import torch.nn.functional as F


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Compute the directed curvature-conditioned quantization ranking loss.

    Nearest-row choices are Euclidean and detached. Target residuals only
    condition pair curvature and target assignment, both detached by contract.
    """
    eps = parameters.get("eps", 1e-8)
    beta = parameters.get("beta", 2.0)
    similarity_offset = parameters.get("similarity_offset", 0.0)
    kappa_min = parameters.get("kappa_min", 0.1)
    kappa_max = parameters.get("kappa_max", 2.0)
    margin = parameters.get("margin", 0.1)

    source_unit = source / (source.norm(dim=-1, keepdim=True) + eps)
    target_unit = target / (target.norm(dim=-1, keepdim=True) + eps)
    similarity = (source_unit * target_unit).sum(dim=-1)
    kappa = kappa_min + (kappa_max - kappa_min) * torch.sigmoid(
        beta * (similarity - similarity_offset)
    )
    kappa = kappa.detach()

    source_indices = torch.cdist(source.detach(), codebook.detach()).square().argmin(dim=-1)
    target_indices = torch.cdist(target.detach(), codebook.detach()).square().argmin(dim=-1)
    source_rows = codebook[source_indices]
    target_rows = codebook[target_indices]

    def chordal_squared(a, b, curvature):
        delta = a - b
        radius = delta.norm(dim=-1)
        z = curvature.sqrt() * radius
        # 2(1-cos(z))/kappa, evaluated stably at z near zero.
        numerator = 4.0 * torch.sin(0.5 * z).square()
        return numerator / curvature

    positive = chordal_squared(source, source_rows, kappa)
    negative = chordal_squared(source, target_rows, kappa)
    per_pair = F.relu(negative - positive + margin)
    weights = mask.to(dtype=per_pair.dtype, device=per_pair.device)
    loss = (per_pair * weights).sum() / (weights.sum() + eps)
    # Keep the declared target gradient contract observable to callers that
    # request gradients for every input: targets condition detached curvature
    # and assignments, so their derivative is exactly zero, not absent.
    return loss + target.sum() * 0.0
