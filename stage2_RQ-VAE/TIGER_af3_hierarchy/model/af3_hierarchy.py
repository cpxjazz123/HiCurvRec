"""AF3-guided hierarchical supervision for the three-level RQ-VAE.

The baseline learns three residual codes per item from its semantic embedding
alone. This module adds supervision that acts on **each level's own assignment**
and on the prefix it forms, driven by the training behaviour graph and its AF3
curvature — with no external teacher model.

The four terms, and what each is for:

``af3_level1_loss``
    Level 1 learns the coarse behaviour communities. Only the edges the frozen
    AF3 convention calls locally dense (above the median) drive it, and they are
    weighted by how dense: ``w = clip((AF3 - median)/(p90 - median), 0, 1)``. The
    agreement is measured as a **Poincaré kernel** between the two items'
    expected level-1 codewords, so the coarse grouping is asked to be close in
    the contract's hyperbolic metric rather than in the Euclidean one. An edge
    with a *low* AF3 is not a negative: it simply does not drive this term, and
    the cross-community edges it represents stay positive in the level-2 term.

``behaviour_level2_loss``
    Level 2 refines inside the level-1 group. Pairs that already agree at level
    1 AND are a behaviour edge are pushed to agree at level 2; pairs that agree
    at level 1 and are NOT an edge are pushed apart. That is what makes the
    second level do work the first cannot.

``discrimination_level3_loss``
    Level 3 keeps items distinguishable. Only *unrelated* pairs are used, and
    only as a hinge: their level-3 agreement is pushed below a small margin and
    no further. No term asks related items to agree at level 3, which is the
    difference between refinement and over-aggregation.

``balance_loss``
    The mean assignment of each level is kept near uniform, so no level can
    collapse onto a few codes while the other terms push items together.

Everything here is a **read-out**: the assignments used for the loss are soft
and differentiable, and the identifiers are still produced by the baseline's own
balanced assignment, untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .poincare_contract import distance as contract_distance
from .poincare_contract import expmap0

#: The temperature of the soft read-out. It is a read-out, not the assignment:
#: the identifiers come from the baseline's own rule.
READOUT_TEMPERATURE = 0.1

#: The temperature of the hyperbolic kernel of the level-1 term.
CURVE_TEMPERATURE = 0.5

#: The curvature the level-1 kernel is measured at.
CURVE = 1.0

#: The margin the level-3 term pushes unrelated pairs below.
DISCRIMINATION_MARGIN = 0.05

#: The cap the level-3 term keeps RELATED pairs under. Two items that share a
#: level-1 and a level-2 code must still be told apart by the third: without
#: this, "refinement" turns into over-aggregation, which is the collapse the
#: earlier behaviour arms hit.
LEVEL3_RELATED_CAP = 0.5

#: How far into the ball the expected level-1 codeword is mapped before the
#: distance is taken: a codeword is a direction, and the map fixes its radius.
BALL_SCALE = 0.6


def soft_assignments(latent: torch.Tensor, codebooks: list[torch.Tensor],
                     temperature: float = READOUT_TEMPERATURE
                     ) -> tuple[torch.Tensor, ...]:
    """The soft assignment of every item at every level, along the residual chain.

    ``latent`` is the encoder's output. Level ``l`` assigns the residual that is
    left after the previous levels' expected codewords, which is the same
    structure the quantiser uses; the difference is that this read-out is soft
    and differentiable, so a loss can act on it.
    """
    assignments = []
    residual = latent
    for codebook in codebooks:
        distances = torch.cdist(residual, codebook, compute_mode="donot_use_mm_for_euclid_dist")
        weights = torch.softmax(-distances.pow(2) / max(temperature, 1e-6), dim=-1)
        assignments.append(weights)
        residual = residual - weights @ codebook
    return tuple(assignments)


def expected_codewords(assignment: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """The expected codeword vector of every item at one level."""
    return assignment @ codebook


def af3_weights(af3: torch.Tensor, median: float, p90: float) -> torch.Tensor:
    """How much each edge drives the level-1 aggregation.

    Above the median the weight rises linearly to one at the ninetieth
    percentile and saturates. An edge below the median carries zero: it is not
    a negative, it simply does not ask for coarse aggregation.
    """
    span = max(p90 - median, 1e-6)
    return ((af3 - median) / span).clamp(min=0.0, max=1.0)


def af3_level1_loss(assignment1: torch.Tensor, codebook1: torch.Tensor,
                    left: torch.Tensor, right: torch.Tensor,
                    weight: torch.Tensor, temperature: float = CURVE_TEMPERATURE,
                    curvature: float = CURVE) -> torch.Tensor:
    """Level 1 aggregates the edges AF3 calls locally dense, in the curved metric.

    The agreement of two items is the Poincare kernel of their expected level-1
    codewords: high when the codes they are drawn to sit close in the ball. The
    AF3 weight decides how much each edge asks for that, so an edge with a low
    AF3 contributes nothing while a dense one asks for it fully.
    """
    if left.numel() == 0 or float(weight.sum()) <= 0.0:
        return torch.zeros((), dtype=assignment1.dtype, device=assignment1.device)
    points = expmap0(expected_codewords(assignment1, codebook1) * BALL_SCALE, curvature)
    separation = contract_distance(points[left].double(), points[right].double(),
                                   curvature).float()
    kernel = torch.exp(-separation / max(temperature, 1e-6))
    return -(weight * torch.log(kernel.clamp_min(1e-6))).sum() / weight.sum()


#: Level 2 keeps the pairs whose level-1 agreement is in the top of the sampled
#: pool, not the pairs above an absolute value. The assignment is close to a
#: decision, so two items either share a code (agreement near one) or do not
#: (agreement near zero); an absolute threshold therefore either selects
#: everything or nothing, and in practice selected nothing at every step.
LEVEL2_KEEP_FRACTION = 0.25


def behaviour_level2_loss(assignment1: torch.Tensor, assignment2: torch.Tensor,
                          positive_left: torch.Tensor, positive_right: torch.Tensor,
                          negative_left: torch.Tensor, negative_right: torch.Tensor,
                          keep_fraction: float = LEVEL2_KEEP_FRACTION) -> torch.Tensor:
    """Level 2 refines inside the level-1 group.

    A positive pair is a behaviour edge whose endpoints already share a level-1
    group; a negative pair shares the level-1 group without being an edge. The
    positives are pulled together at level 2 and the negatives pushed apart
    (hinged, so the term stops once they are separated).
    """
    device = assignment2.device
    terms = []

    def in_the_same_group(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Which pairs currently fall in the same level-1 group.

        The decision, not the continuous agreement: with a nearly one-hot
        assignment two items either share a code or they do not, and a
        continuous threshold selected nothing at every step of the first run.
        """
        return assignment1[left].argmax(dim=-1) == assignment1[right].argmax(dim=-1)

    if positive_left.numel():
        chosen = torch.nonzero(in_the_same_group(positive_left, positive_right)).flatten()
        if chosen.numel():
            agree2 = (assignment2[positive_left[chosen]]
                      * assignment2[positive_right[chosen]]).sum(dim=-1)
            terms.append(-torch.log(agree2.clamp_min(1e-6)).mean())
    if negative_left.numel():
        chosen = torch.nonzero(in_the_same_group(negative_left, negative_right)).flatten()
        if chosen.numel():
            agree2 = (assignment2[negative_left[chosen]]
                      * assignment2[negative_right[chosen]]).sum(dim=-1)
            terms.append(F.softplus(20.0 * agree2).mean())
    if not terms:
        return torch.zeros((), dtype=assignment2.dtype, device=device)
    return torch.stack(terms).mean()


def discrimination_level3_loss(assignment3: torch.Tensor,
                               left: torch.Tensor, right: torch.Tensor,
                               related_left: torch.Tensor | None = None,
                               related_right: torch.Tensor | None = None,
                               margin: float = DISCRIMINATION_MARGIN,
                               related_cap: float = LEVEL3_RELATED_CAP) -> torch.Tensor:
    """Level 3 keeps unrelated items apart, and stops once they are.

    Only unrelated pairs are used. The hinge means the term cannot buy
    separation by scattering the codebook: past the margin it contributes
    nothing, so the reconstruction objective keeps the last word on the code.
    """
    terms = []
    if left.numel():
        agreement = (assignment3[left] * assignment3[right]).sum(dim=-1)
        terms.append(F.relu(agreement - margin).mean())
    if related_left is not None and related_right is not None and related_left.numel():
        related = (assignment3[related_left] * assignment3[related_right]).sum(dim=-1)
        terms.append(F.relu(related - related_cap).mean())
    if not terms:
        return torch.zeros((), dtype=assignment3.dtype, device=assignment3.device)
    return torch.stack(terms).mean()


def balance_loss(assignments: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Every level's mean assignment is kept near uniform."""
    terms = []
    for assignment in assignments:
        usage = assignment.mean(dim=0)
        uniform = torch.full_like(usage, 1.0 / usage.numel())
        terms.append((usage * (usage.clamp_min(1e-9).log()
                               - uniform.clamp_min(1e-9).log())).sum())
    return torch.stack(terms).mean()


def hierarchical_loss(latent: torch.Tensor, codebooks: list[torch.Tensor],
                      batch: dict,
                      weights: dict) -> tuple[torch.Tensor, dict]:
    """All four terms, combined, with the individual values reported.

    ``batch`` carries the sampled edges: ``af3_left``/``af3_right``/
    ``af3_weight`` for level 1, ``positive_left``/``positive_right`` and
    ``negative_left``/``negative_right`` for level 2, and ``unrelated_left``/
    ``unrelated_right`` for level 3.
    """
    assignments = soft_assignments(latent, codebooks)
    level1 = af3_level1_loss(assignments[0], codebooks[0],
                             batch.get("af3_left", torch.empty(0, dtype=torch.long)),
                             batch.get("af3_right", torch.empty(0, dtype=torch.long)),
                             batch.get("af3_weight", torch.empty(0)))
    level2 = behaviour_level2_loss(assignments[0], assignments[1],
                                   batch.get("positive_left", torch.empty(0, dtype=torch.long)),
                                   batch.get("positive_right", torch.empty(0, dtype=torch.long)),
                                   batch.get("negative_left", torch.empty(0, dtype=torch.long)),
                                   batch.get("negative_right", torch.empty(0, dtype=torch.long)))
    level3 = discrimination_level3_loss(
        assignments[2],
        batch.get("unrelated_left", torch.empty(0, dtype=torch.long)),
        batch.get("unrelated_right", torch.empty(0, dtype=torch.long)),
        batch.get("positive_left"), batch.get("positive_right"))
    balance = balance_loss(assignments)
    total = (weights["level1"] * level1 + weights["level2"] * level2
             + weights["level3"] * level3 + weights["balance"] * balance)
    return total, {
        "level1": float(level1.detach()),
        "level2": float(level2.detach()),
        "level3": float(level3.detach()),
        "balance": float(balance.detach()),
        "assignments": assignments,
    }


def gradient_norm(loss: torch.Tensor, parameters) -> float:
    """The L2 norm of the gradient of ``loss`` over ``parameters``.

    A read-only statistic: it is measured with ``autograd.grad`` and never enters
    the backward pass, so it cannot change what the optimiser applies.
    """
    selected = [p for p in parameters if p.requires_grad]
    if not selected:
        return 0.0
    grads = torch.autograd.grad(loss, selected, retain_graph=True, allow_unused=True)
    total = torch.zeros((), dtype=torch.float64)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().double().pow(2).sum()
    return float(total.sqrt())


@dataclass
class GradientBudget:
    """The cap on the new terms' combined share of the gradient.

    The four terms are added together first; this object keeps that sum's
    contribution to the encoder and to the codebook at or below the configured
    share of the original objective's, and applies the stricter of the two
    scalings. It is the same rule the previous arm used, so the comparison
    between mechanisms is a comparison of what they ask for and not of how
    loudly they ask.
    """

    share: float = 0.10
    scale: float = 1.0
    encoder_scale: float = 1.0
    codebook_scale: float = 1.0

    def observe(self, base_encoder: float, new_encoder: float,
                base_codebook: float, new_codebook: float) -> float:
        self.encoder_scale = self._ratio(base_encoder, new_encoder)
        self.codebook_scale = self._ratio(base_codebook, new_codebook)
        self.scale = min(self.encoder_scale, self.codebook_scale)
        return self.scale

    def _ratio(self, base: float, new: float) -> float:
        if new <= 0.0 or base <= 0.0:
            return 1.0
        return min(1.0, self.share * base / new)


# --------------------------------------------------------------------------
# the surfaces the reference contract declares
# --------------------------------------------------------------------------

def poincare_kernel(left: torch.Tensor, right: torch.Tensor,
                    curvature, temperature) -> torch.Tensor:
    """The curved agreement kernel of one pair, row by row.

    ``exp(-d_H(left, right) / temperature)`` with ``d_H`` the contract's
    Poincare distance. This is the explicit curve operation the level-1 term is
    built on: at ``curvature -> 0`` the kernel becomes the Euclidean one.
    """
    separation = contract_distance(left, right, curvature)
    return torch.exp(-separation / temperature)


def agreement(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The overlap of two soft assignments, row by row."""
    return (left * right).sum(dim=-1)


def weighted_curve_term(weight: torch.Tensor, left: torch.Tensor,
                        right: torch.Tensor, curvature, temperature) -> torch.Tensor:
    """The level-1 term of one pair, row by row.

    ``-w * log kernel``: the AF3 weight ``w`` decides how much the edge asks for
    coarse agreement and the kernel supplies the curved metric. Both are on the
    same path, so removing either changes the value.
    """
    kernel = poincare_kernel(left, right, curvature, temperature)
    # the weight arrives as a (n,) vector from the loss and as a (n, 1) column
    # from a per-row contract call; both mean the same thing
    flat = weight.reshape(-1)
    return -flat * torch.log(kernel.clamp_min(1e-6))
