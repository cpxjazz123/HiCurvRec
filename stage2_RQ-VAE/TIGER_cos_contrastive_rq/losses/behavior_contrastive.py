"""Behaviour-edge contrastive objective whose similarity is a curve operation.

CLAUDE.md 19h asks for a training-only behaviour signal *and* an explicit curve operation
coupled on one connected path, with both single-sided versions out of scope.  This loss is the
coupling:

    positives  = edges of the canonical behaviour graph, built from train interactions only
    similarity = -d^2_{-1}(u, v)   (squared geodesic on the curvature -1 ball), /temperature

Turning the behaviour side off shuffles the pairs, so the same vectors are contrasted against
neighbours that carry no graph edge; turning the curve side off returns the archived cosine
similarity.  Either single-sided version is the thing 19h rules out, which is why both switches
exist and why a test asserts each one changes the value.

The archived arm (TIGER_behavior_edge_contrastive_rq) is this loss with the curve side off; the
only two arms above the baseline in the repository are behaviour terms, and the only robust
finding of the previous round is that curvature works when it acts on a metric -- this arm is the
union, and that union had not been tried.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NAME = "behaviour_edge_contrastive"
DESCRIPTION = ("symmetric InfoNCE over train-only behaviour edges; similarity is the curvature "
               "-1 squared geodesic instead of the cosine")

#: Radius cap that keeps the ball metric conditioned (a self-distance of 14 instead of 0 without it).
RADIUS_CAP = 0.9
EPS = 1e-6


def _to_ball(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return RADIUS_CAP * torch.tanh(norm / tau) * x / norm


def _negative_sq_geodesic(first: torch.Tensor, second: torch.Tensor, kappa: float = -1.0):
    """`(B, D)` against `(B, D)` or `(B, K, D)` -> `(B,)` or `(B, K)`, all pairwise element-wise."""
    tau = torch.cat([first.reshape(-1, first.shape[-1]), second.reshape(-1, second.shape[-1])], 0)
    tau = tau.norm(dim=-1).mean().clamp_min(EPS)
    u = _to_ball(first, tau)
    v = _to_ball(second, tau)
    if v.dim() == 3:                       # negatives are (B, K, D): give the anchor a middle axis
        u = u.unsqueeze(1)
    u2 = (u * u).sum(-1).clamp(max=RADIUS_CAP ** 2)
    v2 = (v * v).sum(-1).clamp(max=RADIUS_CAP ** 2)
    diff2 = (u - v).pow(2).sum(-1).clamp_min(0.0)
    inner = 1.0 + 2.0 * diff2 / ((1.0 - u2) * (1.0 - v2))
    geodesic = torch.acosh(inner.clamp_min(1.0 + EPS)) / (abs(kappa) ** 0.5)
    return -geodesic.pow(2)                       # similarity: larger is closer


def _logits(first, second, other, temperature, access_curve, kappa):
    if access_curve:
        positive_logit = _negative_sq_geodesic(first, second, kappa).unsqueeze(-1)
        negative_logits = _negative_sq_geodesic(first, other, kappa)
    else:
        first = F.normalize(first, dim=-1)
        positive_logit = (first * F.normalize(second, dim=-1)).sum(dim=-1, keepdim=True)
        negative_logits = torch.einsum("bd,bkd->bk", first, F.normalize(other, dim=-1))
    return torch.cat([positive_logit, negative_logits], dim=1) / temperature


def behaviour_contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float = 0.07,
    access_behavior: bool = True,
    access_curve: bool = True,
    kappa: float = -1.0,
) -> torch.Tensor:
    """Symmetric InfoNCE with one positive and K negatives per anchor.

    Shapes: anchor/positive `(B, D)`, negatives `(B, K, D)`.
    """
    if anchor.ndim != 2 or positive.shape != anchor.shape:
        raise ValueError("anchor and positive must have the same (B, D) shape")
    if negatives.ndim != 3 or negatives.shape[0] != anchor.shape[0] \
            or negatives.shape[2] != anchor.shape[1]:
        raise ValueError("negatives must have shape (B, K, D) matching anchor")
    if negatives.shape[1] < 1:
        raise ValueError("at least one negative is required")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # Behaviour switch: with it off the same vectors are paired with a different batch's
    # neighbours, i.e. no graph edge connects them any more.
    if not access_behavior:
        order = torch.roll(torch.arange(anchor.shape[0], device=anchor.device), 1)
        positive = positive[order]
        negatives = negatives[order]

    def direction(first, second, other):
        logits = _logits(first, second, other, temperature, access_curve, kappa)
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)

    return 0.5 * (direction(anchor, positive, negatives) + direction(positive, anchor, negatives))


def contrastive_gradient_norms(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    parameters: list[torch.Tensor],
) -> tuple[float, float]:
    """Diagnostic helper: structural and reconstruction gradient norms."""
    raise NotImplementedError
