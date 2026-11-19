"""Behaviour-edge contrastive objective for B1.

One positive per anchor and K degree/semantic-matched negatives. The loss is
symmetric (anchor -> positive and positive -> anchor) and operates on the
straight-through quantized latent, so its gradient reaches the encoder while the
codebook still receives the ordinary quantization objective. The test/validation
split is never read.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NAME = "behaviour_edge_contrastive"
DESCRIPTION = "symmetric InfoNCE over train-only behaviour edges with matched negatives"


def behaviour_contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE with one positive and K negatives per anchor.

    Shapes: anchor/positive `(B, D)`, negatives `(B, K, D)`. Both directions share
    the same candidate set, so the positive cannot get gradient only from one side.
    """
    if anchor.ndim != 2 or positive.shape != anchor.shape:
        raise ValueError("anchor and positive must have the same (B, D) shape")
    if negatives.ndim != 3 or negatives.shape[0] != anchor.shape[0] or negatives.shape[2] != anchor.shape[1]:
        raise ValueError("negatives must have shape (B, K, D) matching anchor")
    if negatives.shape[1] < 1:
        raise ValueError("at least one negative is required")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    n = F.normalize(negatives, dim=-1)

    def direction(first: torch.Tensor, second: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        positive_logit = (first * second).sum(dim=-1, keepdim=True)
        negative_logits = torch.einsum("bd,bkd->bk", first, other)
        logits = torch.cat([positive_logit, negative_logits], dim=1) / temperature
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)

    return 0.5 * (direction(a, p, n) + direction(p, a, n))


def contrastive_gradient_norms(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    parameters: list[torch.Tensor],
) -> tuple[float, float]:
    """Diagnostic helper: structural and reconstruction gradient norms.

    Kept separate from the training loop; the candidate's test suite uses it to
    prove that the loss reaches both sides of the pair rather than only the
    anchor.
    """
    loss = behaviour_contrastive_loss(anchor, positive, negatives)
    grad = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
    struct_norm = float(torch.sqrt(sum((g.detach().pow(2).sum() for g in grad if g is not None),
                                       torch.zeros((), device=anchor.device))))
    return float(loss.detach()), struct_norm
