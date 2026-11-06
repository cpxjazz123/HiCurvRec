"""Tests for the community-pair supervision.

They pin what the mechanism claims: the positive really comes from the anchor's
own community and never is the anchor, the mask excludes exactly the unlabelled
positives and keeps the drawn target, the loss is a cross-entropy that rises
when the positives are pushed away, and both scales are measured from the data
rather than fixed.
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

from model import community_pairs as community  # noqa: E402


def _partition(tmp_path: Path, n_items: int = 12, communities: int = 3) -> Path:
    probabilities = np.zeros((n_items, communities), dtype=np.float32)
    for item in range(n_items):
        probabilities[item, item % communities] = 1.0
    path = tmp_path / "pi.npy"
    np.save(path, probabilities)
    return path


def test_the_positive_comes_from_the_same_community_and_is_never_the_anchor():
    path = _partition(Path("/tmp"))
    partition = community.CommunityPartition(path, 12)
    anchors = torch.arange(12)
    for _ in range(50):
        positive = partition.positive(anchors)
        left = partition.labels[anchors.numpy()]
        right = partition.labels[positive.numpy()]
        assert np.array_equal(left, right)
        assert bool((positive != anchors).all())


def test_the_same_community_mask_marks_the_partition():
    path = _partition(Path("/tmp"))
    partition = community.CommunityPartition(path, 12)
    anchors = torch.tensor([0, 1, 2])
    candidates = torch.tensor([0, 1, 3])
    mask = partition.same_community(anchors, candidates)
    # 0 is with 0 and 3, 1 is with 1, 2 is with 2
    assert mask[0].tolist() == [True, False, True]
    assert mask[1].tolist() == [False, True, False]


def test_the_mask_excludes_the_unlabelled_positives_and_keeps_the_target():
    logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    # every pair shares a community: only the targets remain, so the loss is
    # zero -- the model is not asked to separate items the partition groups
    same = torch.ones(2, 2, dtype=torch.bool)
    assert float(community.contrastive_loss(logits, same)) == pytest.approx(0.0, abs=1e-6)


def test_the_loss_rises_when_the_positives_are_pushed_away():
    good = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    bad = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    same = torch.zeros(2, 2, dtype=torch.bool)
    assert float(community.contrastive_loss(good, same)) < float(
        community.contrastive_loss(bad, same))


def test_the_temperature_is_the_median_pairwise_distance():
    latent = torch.tensor([[0.0], [1.0], [3.0]])
    # the pairwise distances are 1, 3 and 2 -> the median is 2
    assert community.measure_temperature(latent) == pytest.approx(2.0)


def test_the_declared_surface_is_the_row_wise_distance_over_the_temperature():
    source = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[0.0, 0.0]])
    assert float(community.pair_distance(source, positive)) == pytest.approx(1.0)
    assert float(community.pair_logit(source, positive, 2.0)) == pytest.approx(-0.5)


def test_the_matrix_logit_is_the_same_kernel_over_every_pair():
    source = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    matrix = community.logit_matrix(source, positive, 1.0)
    assert matrix.shape == (2, 2)
    assert float(matrix[0, 0]) == pytest.approx(-1.0)
    assert float(matrix[1, 1]) == pytest.approx(0.0)
