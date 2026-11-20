"""Issue #227: the pairing bugs that made #226's numbers unreliable.

Four assertions, each one a defect the audit found:

  * the negative sampler must check the pair that is actually scored `(a, d)`,
    not `(b, d)` -- and never return the anchor or the positive target;
  * a failed sample must not shift the positive/negative pairing;
  * the majority baseline must be the true-label frequency, not the predicted one;
  * the split SHA must hash the full pair content, not a summary of it;
  * the shuffled control must not recreate a held-out pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import graph_teacher2 as gt  # noqa: E402


def _degrees(n_items: int, edges: list[tuple[int, int]]) -> np.ndarray:
    degree = np.zeros(n_items)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    return degree


def test_the_sampler_checks_the_pair_that_is_scored():
    # positive (0 -> 1); the only candidate with 1's degree is 2, and (0, 2) is real
    edges = [(0, 1), (0, 2)]
    degrees = _degrees(6, edges)
    forbidden = set(edges) | {(b, a) for a, b in edges}
    rng = np.random.default_rng(3)
    candidate = gt.degree_matched_negative(0, 1, degrees, forbidden, rng)
    assert candidate != 2, "a true edge was returned as a negative"
    assert candidate is None or candidate not in (0, 1)


def test_the_sampler_never_returns_the_anchor_or_the_positive():
    edges = [(0, 1), (3, 4)]
    degrees = _degrees(8, edges)
    forbidden = set(edges) | {(b, a) for a, b in edges}
    rng = np.random.default_rng(5)
    for _ in range(50):
        candidate = gt.degree_matched_negative(0, 1, degrees, forbidden, rng)
        assert candidate is None or candidate not in (0, 1)


def test_a_failed_sample_does_not_shift_the_pairing():
    pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64)
    degrees = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    forbidden = {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)}

    def matcher(anchor, positive, degrees_, forbidden_, rng):
        # fail at the first, a middle and the last row
        # rows are [0,1], [2,3], [4,5], [6,7]: fail the first, the middle, the last
        return {0: None, 4: None, 6: None}.get(int(anchor), 8)

    positives, negatives, failures = gt.sample_negatives(
        pairs, degrees, forbidden, np.random.default_rng(7), matcher=matcher)
    assert failures == [0, 2, 3]
    assert positives.tolist() == [[2, 3]]
    assert negatives.tolist() == [8]


def test_the_majority_baseline_is_the_true_label_frequency():
    # balanced truth, skewed predictions: the old formula returned ~0.99
    confusion = np.array([[23000, 100], [22700, 200]])
    metrics = gt.direction_metrics(confusion)
    assert abs(metrics["majority_baseline"] - 0.5032) < 0.01
    assert abs(metrics["prediction_frequency_max"] - 0.9869) < 0.01
    assert metrics["majority_baseline"] < metrics["prediction_frequency_max"]


def test_the_split_sha_covers_the_pair_content():
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 40, 200)
    cols = rng.integers(0, 40, 200)
    keep = rows != cols
    first = gt.split_pairs(rows[keep], cols[keep])
    swapped = cols[keep].copy()
    swapped[:2] = swapped[:2][::-1]
    second = gt.split_pairs(rows[keep], swapped)
    assert first["test"].shape == second["test"].shape
    assert first["split_sha"] != second["split_sha"], \
        "the SHA ignores the pair content"


def test_the_shuffled_graph_does_not_recreate_a_held_out_pair():
    index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    adjacency = torch.sparse_coo_tensor(index, torch.ones(4), size=(6, 6)).coalesce()
    degrees = np.array([1, 2, 2, 2, 1, 0], dtype=np.float64)
    forbidden = {(0, 2), (2, 0), (1, 3), (3, 1)}
    shuffled, skipped = gt.shuffled_adjacency(adjacency, degrees, seed=11,
                                              forbidden=forbidden)
    pairs = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in shuffled.indices().t().tolist()}
    assert not (pairs & {(0, 2), (1, 3)}), "the rewire recreated a held-out pair"
    assert skipped >= 0


def test_unordered_pairs_are_oriented_along_the_existing_edge():
    """The split stores min/max pairs; the positive must follow the real direction."""
    pairs = np.array([[2, 5]], dtype=np.int64)          # stored as (min, max) = (2, 5)
    directed = {(5, 2)}                                   # the edge runs 5 -> 2
    forbidden = {(5, 2), (2, 5)}
    degrees = np.array([1, 1, 2, 1, 1, 2], dtype=np.float64)
    positives, negatives, failures = gt.sample_negatives(
        pairs, degrees, forbidden, np.random.default_rng(9), directed=directed)
    assert failures == []
    assert positives.tolist() == [[5, 2]], "the positive was left in the wrong direction"
