"""Host-occupancy-constrained directed transition assignment loss."""

import torch


def _project(logits, occupancy, iterations):
    """KL project row probabilities onto row sums 1 and column sums B*h."""
    batch, codes = logits.shape
    # Work in log space; alternating dual updates retain gradients through the
    # finite Sinkhorn solver while avoiding underflow at the small temperature.
    log_x = logits
    log_h = occupancy.clamp_min(torch.finfo(logits.dtype).tiny).log()
    log_rows = torch.zeros(batch, device=logits.device, dtype=logits.dtype)
    log_cols = (batch * occupancy).clamp_min(torch.finfo(logits.dtype).tiny).log()
    u = torch.zeros_like(log_rows)
    v = torch.zeros_like(log_h)
    for _ in range(max(1, int(iterations))):
        u = log_rows - torch.logsumexp(log_x + v.unsqueeze(0), dim=1)
        v = log_cols - torch.logsumexp(log_x + u.unsqueeze(1), dim=0)
    return torch.exp(log_x + u.unsqueeze(1) + v.unsqueeze(0))


def mechanism_loss(source, target, codebook, mask, **parameters):
    """Directed CE between assignments with a fixed occupancy column marginal.

    ``occupancy`` is required in training and is the frozen host snapshot. If
    omitted by a standalone caller, the proposal's uniform checker marginal is
    used.
    """
    tau = float(parameters.get("tau", 0.2))
    epsilon = float(parameters.get("epsilon", 1e-8))
    iterations = int(parameters.get("sinkhorn_iterations", 50))
    scale = float(parameters.get("lambda", 1.0))
    valid = mask.to(device=source.device).reshape(-1) > 0
    if not bool(valid.any()):
        return (source.sum() + target.sum() + codebook.sum()) * 0.0
    source = source[valid]
    target = target[valid]
    if "occupancy" in parameters:
        occupancy = torch.as_tensor(parameters["occupancy"], device=codebook.device,
                                    dtype=codebook.dtype)
    else:
        occupancy = torch.full((codebook.shape[0],), 1.0 / codebook.shape[0],
                               device=codebook.device, dtype=codebook.dtype)
    floor = float(parameters.get("occupancy_floor", 1e-4))
    occupancy = occupancy.clamp_min(floor)
    occupancy = occupancy / occupancy.sum()
    source_logits = -torch.cdist(source, codebook).pow(2) / tau
    target_logits = -torch.cdist(target, codebook).pow(2) / tau
    assignments = _project(source_logits, occupancy, iterations)
    target_assignments = _project(target_logits, occupancy, iterations)
    # Directed source-to-target cross entropy; target remains attached.
    return -scale * (assignments * (target_assignments + epsilon).log()).sum(dim=1).mean()
