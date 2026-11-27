"""Known-answer checks for the issue-270 linkage logic (no production import)."""
from __future__ import annotations
import math
import numpy as np


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def ndcg10_from_ranks(ranks):
    return sum(1.0 / math.log2(r + 1) for r in ranks if 0 < r <= 10) / len(ranks)


def hit_and_rank(top, target):
    pos = top.index(target) + 1 if target in top else 0
    return int(0 < pos <= 5), int(0 < pos <= 10), pos


def test_lcp_known_cases():
    assert lcp([1, 2, 3], [1, 2, 3]) == 3
    assert lcp([1, 2, 3], [1, 9, 3]) == 1
    assert lcp([1, 2, 3], [4, 2, 3]) == 0
    assert lcp([], []) == 0


def test_hit_rank_and_ndcg():
    assert hit_and_rank([9, 8, 7, 6, 5, 4], 4) == (0, 1, 6)
    assert hit_and_rank([1, 2, 3], 1) == (1, 1, 1)
    assert hit_and_rank([1, 2, 3], 99) == (0, 0, 0)
    # one hit at rank 1, one miss => 0.5
    assert abs(ndcg10_from_ranks([1, 0]) - 0.5) < 1e-15
    # rank 2 => 1/log2(3)
    assert abs(ndcg10_from_ranks([2]) - 1.0 / math.log2(3)) < 1e-15


def test_group_sizes_and_purity():
    codes = np.array([[1, 1], [1, 1], [1, 2], [2, 1]])
    keys = codes[:, 0] * 10 + codes[:, 1]
    uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    assert cnt.tolist() == [2, 1, 1]
    roles = np.array([0, 0, 1, 2])
    # L1 group {0,1,2}: purity by majority = 2/3
    group = codes[:, 0] == 1
    vals, counts = np.unique(roles[group], return_counts=True)
    assert abs(counts.max() / group.sum() - 2 / 3) < 1e-15


def test_cluster_bootstrap_deterministic():
    rng = np.random.default_rng(7)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    draws = [rng.choice(a, len(a), replace=True).mean() for _ in range(5)]
    rng2 = np.random.default_rng(7)
    draws2 = [rng2.choice(a, len(a), replace=True).mean() for _ in range(5)]
    assert np.allclose(draws, draws2)


def test_collision_definition():
    triples = [(1, 2, 3), (1, 2, 3), (4, 5, 6)]
    assert abs((1 - len(set(triples)) / len(triples)) - 1 / 3) < 1e-15
