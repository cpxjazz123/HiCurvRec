"""Behavior-conditioned pairwise curvature ranking for the level-1 codebook."""

import torch


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Return the masked directed residual-to-codeword ranking loss.

    Nearest-row decisions are ordinary squared-Euclidean decisions and detached;
    target residuals only condition curvature and select the target's row.
    """
    eps = parameters.get("eps", 1e-8)
    beta = parameters.get("beta", 2.0)
    kappa_min = parameters.get("kappa_min", 0.1)
    kappa_max = parameters.get("kappa_max", 2.0)
    offset = parameters.get("similarity_offset", 0.0)
    margin = parameters.get("margin", 0.1)

    # Keep an exact zero connected to all differentiable inputs for empty masks.
    valid = mask.to(device=source.device, dtype=source.dtype).reshape(-1)
    if not bool(torch.any(valid != 0)):
        return (source.sum() + target.sum() + codebook.sum()) * 0.0

    target_condition = target.detach()
    source_idx = (source.detach().square().sum(dim=1, keepdim=True)
                  - 2.0 * source.detach() @ codebook.detach().t()
                  + codebook.detach().square().sum(dim=1).unsqueeze(0)).argmin(dim=1)
    target_idx = (target_condition.square().sum(dim=1, keepdim=True)
                  - 2.0 * target_condition @ codebook.detach().t()
                  + codebook.detach().square().sum(dim=1).unsqueeze(0)).argmin(dim=1)

    # Curvature is a conditioning signal, not a second route for the auxiliary
    # gradient into the encoder.  Keeping the source-side curvature input
    # detached leaves the source gradient supplied by the ranking distances,
    # while the target remains fully gradient-free as specified.
    u = source.detach() / (source.detach().norm(dim=1, keepdim=True) + eps)
    v = target_condition / (target_condition.norm(dim=1, keepdim=True) + eps)
    similarity = (u * v).sum(dim=1)
    kappa = kappa_min + (kappa_max - kappa_min) * torch.sigmoid(
        beta * (similarity - offset)
    )

    near_source = codebook.index_select(0, source_idx)
    near_target = codebook.index_select(0, target_idx)

    def chordal_squared(a, b):
        delta = a - b
        # Use the algebraically equivalent 4*sin(z/2)^2/kappa form.  Apart
        # from avoiding cancellation for small z, this gives a well-behaved
        # autograd path at coincident vectors (where 1-cos(z) loses precision).
        z = kappa.sqrt() * delta.norm(dim=1)
        return 4.0 * torch.sin(0.5 * z).square() / kappa

    ranking = (chordal_squared(source, near_target)
               - chordal_squared(source, near_source) + margin).relu()
    return (ranking * valid).sum() / (valid.sum() + eps)
