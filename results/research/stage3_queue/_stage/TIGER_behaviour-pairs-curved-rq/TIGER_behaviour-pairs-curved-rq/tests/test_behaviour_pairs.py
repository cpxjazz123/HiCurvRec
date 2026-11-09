"""Tests for the behaviour-pair supervision.

They pin what the mechanism claims: the positives are the training split's own
transitions and are redrawn every step, a candidate that is also a successor is
masked out as an unlabelled positive rather than pushed away, the kernel is the
contract's Poincare distance, and the bound keeps the term from dominating the
reconstruction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import behaviour_pairs as behaviour  # noqa: E402


class _Frame:
    """A stand-in for the training parquet: history plus target."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, column):
        index = {"history": 0, "target": 1}[column]
        return [row[index] for row in self.rows]


def test_the_successors_come_from_the_sequences():
    frame = _Frame([([1, 2, 3], 4), ([5, 6], None)])
    successors = behaviour.successor_rows(frame)
    assert successors[1] == [2]
    assert successors[2] == [3]
    assert successors[3] == [4]
    assert successors[5] == [6]


def test_the_pairs_are_redrawn():
    frame = _Frame([([1, 2, 3, 4], 5)])
    pairs = behaviour.BehaviourPairs(frame, 8)
    first, second = pairs.draw(4)
    assert first.size == 4 and second.size == 4
    for anchor, target in zip(first.tolist(), second.tolist()):
        assert target in pairs.successors[anchor]


def test_the_mask_marks_the_unlabelled_positives():
    frame = _Frame([([1, 2, 3], 4)])
    pairs = behaviour.BehaviourPairs(frame, 8)
    anchors = torch.tensor([1])
    candidates = torch.tensor([2, 3, 4])
    mask = pairs.successor_mask(anchors, candidates)
    # the sequence is 1 -> 2 -> 3 -> 4, so only 2 is a successor of 1
    assert mask[0].tolist() == [True, False, False]


def test_the_loss_excludes_the_masked_candidates():
    """A masked candidate leaves the denominator, so it is not pushed away."""
    logits = torch.tensor([[5.0, 5.0, 0.0]])
    excluded = torch.tensor([[False, True, False]])
    masked = float(behaviour.contrastive_loss(logits, excluded))
    unmasked = float(behaviour.contrastive_loss(logits, torch.zeros(1, 3, dtype=torch.bool)))
    assert masked < unmasked


def test_the_kernel_is_the_contract_distance():
    from model.poincare_contract import distance as contract_distance, expmap0

    source = torch.tensor([[0.3, -0.2]])
    positive = torch.tensor([[0.1, 0.4]])
    scale = 1.0
    logits = behaviour.curved_logit(source, positive, 1.0, scale, 1.0)
    expected = -float(contract_distance(expmap0(source.double(), 1.0),
                                       expmap0(positive.double(), 1.0), 1.0))
    assert float(logits[0]) == pytest.approx(expected, rel=1e-9)


def test_the_bound_never_exceeds_one():
    assert behaviour.bounded_scale(2.0, 0.5) == pytest.approx(1.0)
    assert behaviour.bounded_scale(1.0, 4.0) == pytest.approx(0.25)
    assert behaviour.bounded_scale(1.0, 0.0) == pytest.approx(1.0)


def test_the_lift_scale_is_the_median_of_the_norms():
    """The median, so the lifted points stay inside the ball rather than on it."""
    latent = torch.tensor([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0]])
    assert behaviour.latent_scale(latent) == pytest.approx(1.5, rel=1e-6)


def test_the_gradient_norm_does_not_touch_the_parameters():
    net = torch.nn.Linear(3, 2)
    term = net(torch.ones(4, 3)).pow(2).sum()
    before = [p.grad for p in net.parameters()]
    norm = behaviour.encoder_gradient_norm(term, list(net.parameters()))
    assert norm > 0.0
    assert all(a is b for a, b in zip(before, [p.grad for p in net.parameters()]))
