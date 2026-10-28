"""The differentiable proxy of the Issue #126 hard prefix score.

The score asks whether a first-level prefix keeps enough behavioural information
to order a real next item above fixed negatives. The proxy asks the same question
of the assignment the quantizer actually makes, with the same frozen reference and
the same distance, so that improving the proxy is an attempt to improve the score
rather than a second objective that happens to look similar.

Reading rule (frozen, recorded in the candidate's registration):

* ``p_ib`` is the softmax over the level-1 assignment cost the quantizer already
  computes; no second assignment head exists.
* the centre ``mu_b`` of a bucket is the weighted hyperbolic Frechet mean of the
  frozen target representations under the soft weights, refreshed from the hard
  assignment every ``REFRESH_STEPS`` steps and held fixed in between. The
  approximation is stated, not hidden: gradients reach the assignment through
  ``p_ib`` and the distances, and they do NOT reach the centre, which is a cached
  stop-gradient quantity between refreshes.
* ``D_P(a, i) = sum_b p_ib * d(u_a, mu_b)^2`` with the frozen source
  representation ``u_a`` (stop-gradient: the reference is a teacher, not a
  parameter).
* the loss term is ``softplus(-(D_P(a, j-) - D_P(a, i+)) / tau)`` with the frozen
  temperature, averaged over the training pairs.
* ``LAMBDA`` is derived once, from the ratio of the initial gradient norms of the
  original objective and the new term on a fixed calibration sample, and then
  frozen. It is not tuned against any result.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from model.frechet_poincare import clamp_ball, exp_at, exp_0, log_at, poincare_distance

#: The frozen reference of Issue #126 and its reading rule.
REFERENCE_FILE = "results/research/issue126/reference/reference.npz"
REFERENCE_RECORD = "results/research/issue126/reference/reference.json"

#: How the score is turned into a training signal.
TEMPERATURE = None          # read from the frozen record, never chosen here
ASSIGN_TEMPERATURE = 0.1    # scale of the level-1 costs of the frozen recipe
REFRESH_STEPS = 50          # steps between two centre refreshes (stated approximation)
CENTRE_STEPS = 20           # Karcher iterations per refresh
LAMBDA = None               # derived once by calibrate(), then frozen in the record


class FrozenReference:
    """The teacher: frozen source and target representations plus its temperature."""

    def __init__(self, root: Path, items: int):
        arrays = torch.load  # noqa: F841  (documented: the file is a .npz, not a pickle)
        import numpy as np
        payload = np.load(root / REFERENCE_FILE)
        record = json.loads((root / REFERENCE_RECORD).read_text(encoding="utf-8"))
        self.source = torch.from_numpy(payload["source"].astype("float32"))
        self.target = torch.from_numpy(payload["target"].astype("float32"))
        self.temperature = float(record["temperature"])
        if self.source.shape[0] < items or self.target.shape[0] < items:
            raise ValueError("the frozen reference does not cover every item")


def soft_assignment(cost: torch.Tensor, temperature: float = ASSIGN_TEMPERATURE) -> torch.Tensor:
    """``softmax(-D/T)`` over the level-1 codes, from the quantizer's own cost."""
    return torch.softmax(-cost / float(temperature), dim=-1)


def frechet_centres(targets: torch.Tensor, weights: torch.Tensor,
                    steps: int = CENTRE_STEPS) -> torch.Tensor:
    """Weighted hyperbolic Frechet mean of the targets under ``weights``.

    ``weights`` is ``[items, codes]``: the membership of every item in every
    bucket. Returns ``[codes, dim]`` centres inside the ball.
    """
    codes = weights.shape[1]
    mass = weights.sum(dim=0).clamp_min(1e-9)                     # [codes]
    initial = (weights.t() @ targets) / mass.unsqueeze(-1)        # coordinate average
    centre = clamp_ball(exp_0(initial))
    for _ in range(int(steps)):
        # mean of log_centre(target_i) weighted by the membership
        logs = log_at(centre.unsqueeze(0).expand(targets.shape[0], -1, -1),
                      targets.unsqueeze(1).expand(-1, codes, -1))  # [items, codes, dim]
        step = (weights.unsqueeze(-1) * logs).sum(dim=0) / mass.unsqueeze(-1)
        centre = exp_at(centre, step)
    return clamp_ball(centre)


def proxy_distances(assignment: torch.Tensor, source: torch.Tensor,
                    centres: torch.Tensor) -> torch.Tensor:
    """``D_P(a, i) = sum_b p_ib * d(u_a, mu_b)^2`` for a batch of (a, i) pairs."""
    pair_distance = poincare_distance(source.unsqueeze(1), centres.unsqueeze(0)) ** 2
    return (assignment * pair_distance).sum(dim=-1)


def ranking_term(assignment_pos: torch.Tensor, assignment_neg: torch.Tensor,
                 source: torch.Tensor, centres: torch.Tensor,
                 temperature: float) -> torch.Tensor:
    """The average ``softplus(-delta/tau)`` over one batch of pairs."""
    d_pos = proxy_distances(assignment_pos, source, centres)
    d_neg = proxy_distances(assignment_neg, source, centres)
    return torch.nn.functional.softplus(-(d_neg - d_pos) / float(temperature)).mean()


def calibrate(original_grad_norm: float, term_grad_norm: float) -> float:
    """``lambda`` from the ratio of the two initial gradient norms.

    A single derivation, recorded and frozen: the new term starts at the same
    order of magnitude as the objective it is added to, so neither is drowned.
    """
    if not (term_grad_norm > 0) or not (original_grad_norm > 0):
        raise ValueError("a zero gradient norm cannot calibrate the weight")
    return float(original_grad_norm) / float(term_grad_norm)
