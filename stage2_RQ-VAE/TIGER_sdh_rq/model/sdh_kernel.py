"""The same-code support and the curvature kernel of issue #194.

#193 measured where the #181 distillation actually put its weight: items sharing no code level with the anchor are 94.95% of the teacher's support, so
nineteen pairs in twenty taught the student nothing it had not already separated.  This module changes exactly two things and nothing else.

``support_mask`` keeps a candidate only when it shares at least one code level with the anchor (the issue's 同码支撑).  ``affinity`` replaces the
teacher's row values with ``A_k(i, j) = 1 / (1 + d_k(e_i, e_j))``, where ``d_k`` is the hyperbolic distance under ``k = sqrt(k_i k_j)``: dimensionless,
strictly decreasing, and carrying no temperature, margin or any other free constant.

Both are switches rather than conditionals scattered through the step: ``SUPPORT_OFF`` opens the support to every candidate (the repaired #181, for
off-line reconciliation only), ``GEOMETRY_OFF`` falls back to the cosine affinity, and ``BEHAVIOUR_OFF`` lives in :mod:`model.struct_distill` because
it closes the term itself.
"""
from __future__ import annotations

import torch

from . import r48k_geometry

#: Open the support to every candidate pair, i.e. the repaired #181.  Off by default: the mechanism is the restriction.
SUPPORT_OFF = False
#: Fall back to the cosine affinity instead of the curvature kernel.  Off by default.
GEOMETRY_OFF = False


def support_mask(anchor_codes: torch.Tensor, candidate_codes: torch.Tensor, candidates: torch.Tensor | None = None) -> torch.Tensor:
    """``[anchors, candidates]`` boolean: the pair shares at least one code level.

    ``anchor_codes`` is ``[A, L]``, ``candidate_codes`` is ``[A, C, L]``; padding slots (``candidates < 0``) are never in the support.
    """
    shared = (anchor_codes.unsqueeze(1) == candidate_codes).sum(dim=-1)
    mask = shared >= 1
    if SUPPORT_OFF:
        mask = torch.ones_like(mask)
    if candidates is not None:
        mask = mask & (candidates >= 0).to(mask.device)
    return mask


def affinity(anchor_embeddings: torch.Tensor, candidate_embeddings: torch.Tensor,
             anchor_kappa: torch.Tensor, candidate_kappa: torch.Tensor) -> torch.Tensor:
    """``A_k(i, j) = 1 / (1 + d_k(e_i, e_j))`` on ``[A, C, D]`` candidate tensors, detached: the teacher is frozen.

    ``k`` is the pair's geometric mean ``sqrt(k_i k_j)``, which is what :func:`model.r48k_geometry.pair_kappa` defines.  Both the embeddings and the
    curvature come from the frozen table, so no parameter of this quantity is learned and no constant was chosen.
    """
    a = anchor_embeddings.detach().to(torch.float64).unsqueeze(1)
    b = candidate_embeddings.detach().to(torch.float64)
    if a.dim() == 3 and a.shape[1] == 1:
        a = a.expand(b.shape[0], b.shape[1], a.shape[-1])
    if GEOMETRY_OFF:
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        return (0.5 * (cos + 1.0)).to(anchor_embeddings.dtype)
    k = r48k_geometry.pair_kappa(anchor_kappa.detach().reshape(-1), candidate_kappa.detach().reshape(-1))
    k = k.to(torch.float64)
    p = r48k_geometry.project_to_ball(a, k.unsqueeze(-1))
    q = r48k_geometry.project_to_ball(b, k.unsqueeze(-1))
    return (1.0 / (1.0 + r48k_geometry.distance(p, q, k.unsqueeze(-1)))).to(anchor_embeddings.dtype)


def teacher_distribution(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalise the affinity inside each anchor's support; an empty support stays all-zero and is dropped by the caller."""
    values = scores.masked_fill(~mask, 0.0).clamp_min(0.0)
    total = values.sum(dim=-1, keepdim=True)
    return values / total.clamp_min(1e-12)


def kl_within_support(student_logits: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``KL(teacher || softmax(student))`` averaged over the anchors that have a support at all.

    The student distribution is normalised inside the same support, so the two sides describe the same pairs; an anchor whose support is empty
    contributes nothing rather than a NaN.
    """
    log_p = torch.log_softmax(student_logits.masked_fill(~mask, float("-inf")), dim=-1)
    per_anchor = (teacher * (torch.log(teacher.clamp_min(1e-12)) - log_p)).sum(dim=-1)
    live = (teacher > 0).any(dim=-1)
    if not bool(live.any()):
        return student_logits.sum() * 0.0
    return per_anchor[live].mean()
