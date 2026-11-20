"""Issue #224: distill the Teacher's *relations* into the quantiser, no classifier.

The Student head in #222/#223 first turned `q` into Type/Orbit logits and matched
those to a label (or to the Teacher's posterior).  This module drops that step: the
target is the structure the Teacher's node states *have*, not a class name.

    h_T = teacher(items, adjacency)          (B,k,128), stop-gradient
    P_T(i,j) = softmax_j( cos(h_i,h_j) / tau )        relation distribution
    q   = RQ-VAE(items)                       (B,k,32), real quantiser
    P_Q(i,j) = softmax_j( cos(q_i,q_j) / tau )
    L_rel = mean_i KL( P_T(i,:) || P_Q(i,:) )

The dimensions never have to agree: only the *ordering* over the k nodes is
compared.  Two limits are deliberate and recorded, not hidden: the similarity is
symmetric, so A->B and B->A are the same target here; and the Teacher state is a
function of the instance's adjacency, so a fixed SID can only match the part of
the relation that is stable across the instances the item appears in (measured by
`relation_stability.py` before this loss is trusted).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def relation_distribution(representations: torch.Tensor, temperature: float = 0.1,
                          eps: float = 1e-8) -> torch.Tensor:
    """Row-wise softmax over `cos(z_i, z_j)` with the self-similarity excluded."""
    if temperature <= 0:
        raise ValueError("temperature must be positive, got %r" % (temperature,))
    normalised = F.normalize(representations, dim=-1, eps=eps)
    similarity = normalised @ normalised.transpose(-1, -2)
    nodes = similarity.shape[-1]
    if nodes < 2:
        raise ValueError("a relation needs at least two nodes")
    eye = torch.eye(nodes, dtype=torch.bool, device=similarity.device)
    similarity = similarity.masked_fill(eye, float("-inf"))
    return F.softmax(similarity / float(temperature), dim=-1)


def adjacency_distribution(adjacency: torch.Tensor, uniform: float = 0.0) -> torch.Tensor:
    """The behaviour-graph control target: uniform over a node's neighbours.

    `adjacency` is (B,k,k) with 1 at an edge; a row with no edge falls back to
    uniform over the other nodes so the KL is defined.
    """
    nodes = adjacency.shape[-1]
    eye = torch.eye(nodes, dtype=torch.bool, device=adjacency.device)
    weights = adjacency.masked_fill(eye, 0.0).clamp(min=0.0)
    totals = weights.sum(-1, keepdim=True)
    uniform_row = torch.full_like(weights, 1.0 / max(nodes - 1, 1)).masked_fill(eye, 0.0)
    mixed = torch.where(totals > 0, weights / totals.clamp(min=1e-12), uniform_row)
    return (1.0 - uniform) * mixed + uniform * uniform_row


def relational_loss(student_representation: torch.Tensor, target: torch.Tensor,
                    temperature: float = 0.1) -> dict:
    """KL(P_T || P_Q) averaged over instances; P_Q is built inside the graph."""
    student = relation_distribution(student_representation, temperature)
    log_student = torch.log(student.clamp_min(1e-12))
    per_instance = (target * (torch.log(target.clamp_min(1e-12)) - log_student)).sum(-1)
    loss = per_instance.mean()
    with torch.no_grad():
        top1_target = target.argmax(-1)
        top1_student = student.argmax(-1)
        agreement = (top1_target == top1_student).float().mean()
    return {"loss": loss, "kl": float(loss.detach()), "top1_agreement": float(agreement)}


def teacher_relation(teacher, items: torch.Tensor, adjacency: torch.Tensor,
                     temperature: float = 0.1) -> torch.Tensor:
    """Stop-gradient relation target from the frozen Teacher's node states."""
    with torch.no_grad():
        _, _, hidden = teacher(items, adjacency)
        return relation_distribution(hidden, temperature)


def relation_fidelity(representation: torch.Tensor, target: torch.Tensor,
                      temperature: float = 0.1) -> dict:
    """KL and top-1 agreement of a representation against a relation target."""
    with torch.no_grad():
        return relational_loss(representation, target, temperature)
