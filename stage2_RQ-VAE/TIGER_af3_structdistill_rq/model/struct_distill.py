"""The relational structure-distillation loss of issue #181 -- no positive/negative samples anywhere.

The student is read from the *quantised* SID: ``z_hat = z + stopgrad(q - z)``, so the forward value equals the readout the
exported codes produce while the gradient still reaches the encoder. For each anchor's candidate set the teacher's relational
scores ``T^G`` and the student's cosines ``S^Q`` are z-scored inside the set and softmaxed, and the loss is the KL from the
frozen teacher distribution to the student's.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: Switches of the issue: SD_OFF reproduces the #180 arm with the contrast closed and no structural term; GEOMETRY_OFF is the
#: Euclidean degeneration and is a separate switch on purpose.
SD_OFF = False
GEOMETRY_OFF = False

#: Candidate-set composition, fixed by the issue.
ONE_HOP, TWO_HOP, GLOBAL = 16, 16, 32
#: Variance floor of the per-row z-score.
VAR_FLOOR = 1e-6


def ste_readout(z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``z + stopgrad(q - z)``: forward equals ``q``, gradient flows to ``z``."""
    return z + (q - z).detach()


def candidate_sets(anchors, neighbours, two_hop, n_items: int, seed: int = 42) -> list[list[int]]:
    """Deterministic per-anchor candidate sets: up to 16 one-hop, 16 two-hop, 32 global, deduplicated, anchor excluded."""
    generator = torch.Generator().manual_seed(seed)
    sets: list[list[int]] = []
    for anchor in anchors:
        chosen: list[int] = []
        for pool, limit in ((neighbours.get(anchor, []), ONE_HOP), (two_hop.get(anchor, []), TWO_HOP)):
            for item in list(pool)[:limit]:
                if int(item) != int(anchor) and int(item) not in chosen:
                    chosen.append(int(item))
        draw = torch.randperm(n_items, generator=generator)[: GLOBAL * 4].tolist()
        added = 0
        for item in draw:
            if int(item) != int(anchor) and int(item) not in chosen:
                chosen.append(int(item))
                added += 1
                if added >= GLOBAL:
                    break
        sets.append(chosen)
    return sets


def zscore(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(VAR_FLOOR ** 0.5)
    return (x - mean) / std


def relational_loss(student_cos: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
    """KL(stopgrad(softmax(zscore(teacher))) || softmax(zscore(student))), averaged over anchors."""
    p_teacher = F.softmax(zscore(teacher_scores.detach()), dim=-1)
    p_student = F.log_softmax(zscore(student_cos), dim=-1)
    return F.kl_div(p_student, p_teacher, reduction="batchmean")


def student_cosines(z_hat: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """``[anchors, candidates]`` cosine between each STE readout and its set.

    ``candidates`` is either one shared ``[C, D]`` pool or a per-anchor ``[A, C, D]`` tensor.
    """
    a = F.normalize(z_hat, dim=-1)
    b = F.normalize(candidates if candidates.dim() == 3 else candidates.unsqueeze(0), dim=-1)
    return torch.einsum("ad,acd->ac", a, b.expand(a.shape[0], -1, -1))
