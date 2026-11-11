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


def two_hop_of(anchor: int, neighbours: dict) -> list[int]:
    """Two-hop neighbours of one anchor, computed on demand.

    Precomputing this for every node costs |N| x deg^2 on a graph with hubs in the thousands, which is why it is derived per anchor
    instead: a step only ever asks for its 63 anchors.
    """
    # membership in a set (a list scan is O(n^2) on hub anchors) and an early stop once enough two-hop candidates exist: the
    # candidate set only ever keeps TWO_HOP of them, so expanding a hub's whole neighbourhood buys nothing
    seen: set[int] = set()
    out: list[int] = []
    for mid in neighbours.get(anchor, []):
        for far in neighbours.get(mid, []):
            if far != anchor and far not in seen:
                seen.add(far)
                out.append(far)
                if len(out) >= TWO_HOP * 4:
                    return out
    return out


def candidate_sets(anchors, neighbours, two_hop, n_items: int, seed: int = 42) -> list[list[int]]:
    """Deterministic per-anchor candidate sets: up to 16 one-hop, 16 two-hop, 32 global, deduplicated, anchor excluded.

    ``two_hop`` may be ``None``, in which case each anchor's two-hop pool is derived from ``neighbours`` as it is needed.
    """
    generator = torch.Generator().manual_seed(seed)
    sets: list[list[int]] = []
    for anchor in anchors:
        chosen: list[int] = []
        second = two_hop_of(anchor, neighbours) if two_hop is None else two_hop.get(anchor, [])
        for pool, limit in ((neighbours.get(anchor, []), ONE_HOP), (second, TWO_HOP)):
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


#: Time window of the structural term, taken from #179: nothing before epoch 1500, linear to lambda* at 3000.
WARMUP_EPOCH, FULL_EPOCH = 1500, 3000
#: The probe's fixed scale, from the issue's formula.
LAMBDA_SCALE = 0.1


def lambda_schedule(epoch: int) -> float:
    """``0`` before the window, then linear from ``0`` at ``WARMUP_EPOCH`` to ``1`` at ``FULL_EPOCH``."""
    if epoch <= WARMUP_EPOCH:
        return 0.0
    if epoch >= FULL_EPOCH:
        return 1.0
    return (epoch - WARMUP_EPOCH) / float(FULL_EPOCH - WARMUP_EPOCH)


def calibrate(grad_rq: float, grad_sd: float) -> float:
    """``lambda* = 0.1 * ||grad_enc L_RQ|| / (||grad_enc L_SD|| + 1e-12)``, measured once on the frozen probe."""
    return LAMBDA_SCALE * float(grad_rq) / (float(grad_sd) + 1e-12)
