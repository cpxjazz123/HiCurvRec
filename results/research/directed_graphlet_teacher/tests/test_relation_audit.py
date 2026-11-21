"""Self-tests for the relation audit's measurement code.

Each test pins a property the audit relies on, so a broken metric cannot silently
produce a conclusion: AUC convention, prefix distance, ridge recovery, kNN precision,
code purity against its chance level, and deterministic sampling.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "relation"))

import audit_relation as ar  # noqa: E402


def test_l2norm_rows_unit():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = ar.l2norm(x)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], 0.0)


def test_auc_perfect_and_chance():
    assert ar.auc_score(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert ar.auc_score(np.array([0.1, 0.2]), np.array([0.9, 0.8])) == 0.0
    assert ar.auc_score(np.array([0.5]), np.array([0.5])) == 0.5


def test_auc_counts_ties_as_half():
    # one positive tied with one negative, one clearly higher -> (1.5)/2
    assert ar.auc_score(np.array([0.5, 1.0]), np.array([0.5])) == pytest.approx(0.75)


def test_prefix_score_leading_only():
    a = np.array([[1, 2, 3], [1, 2, 3], [1, 5, 9], [4, 0, 0]])
    b = np.array([[1, 2, 3], [1, 7, 3], [1, 5, 8], [4, 0, 1]])
    assert ar.prefix_score(a, b).tolist() == [3, 1, 2, 2]


def test_ridge_recovers_exact_linear_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 20))
    w_true = rng.normal(size=(20, 5))
    y = x @ w_true
    w, xm, ym = ar.ridge_predict(x, y, lam_frac=1e-12)
    pred = (x - xm) @ w + ym
    assert np.abs(pred - y).max() < 1e-6


def test_top_k_excludes_self():
    row = np.array([0.9, 0.8, 0.7, 0.6])
    picks = ar.top_k(row, 2, 0)
    assert 0 not in picks and set(picks) == {1, 2}


def test_graph_precision_synthetic():
    neighbours = [{1}, {0, 2}, {1}, set()]
    probes = np.array([0])
    # row 0: item 1 is the only real neighbour; make item 1 top, item 3 second
    block = np.array([[0.0, 0.9, 0.1, 0.5]])
    assert ar.graph_precision(block, probes, neighbours, k=2) == 0.5
    block = np.array([[0.0, 0.1, 0.9, 0.5]])
    assert ar.graph_precision(block, probes, neighbours, k=1) == 0.0


def test_code_purity_beats_chance_when_groups_align():
    # 20 items, code 0 has 11 members; make the top neighbours stay inside it
    codes = np.zeros((20, 3), dtype=np.int64)
    codes[11:, 0] = 1
    probes = np.array([0])
    block = np.zeros((1, 20))
    block[0, 1:11] = 1.0     # every top neighbour shares first code 0
    purity, chance = ar.code_purity(block, probes, codes, k=10)
    assert purity == 1.0
    assert chance == pytest.approx(0.55)


def test_stable_sample_deterministic_and_disjoint_of_excluded_range():
    index = np.arange(100)
    a = ar.stable_sample(index, 10, seed=7)
    b = ar.stable_sample(index, 10, seed=7)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 10
    assert ar.stable_sample(index, 200, seed=7).tolist() == index.tolist()


def test_r2_multi_perfect_and_zero():
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert ar.r2_multi(y, y) == pytest.approx(1.0)
    assert ar.r2_multi(np.zeros_like(y), y) < 0.0


def test_procrustes_undoes_rotation():
    rng = np.random.default_rng(1)
    target = ar.l2norm(rng.normal(size=(200, 8)))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    rotated = target @ q
    aligned = ar.procrustes_align(rotated, target)
    assert np.abs(aligned - target).max() < 1e-8


# ---------------------------------------------------------------- triplet scoring
def _one_hot_cosine_states(rows: dict, n: int = 8, dim: int = 8):
    """Unit vectors whose cosine is 1 iff two ids share a group letter."""
    import numpy as np
    states = np.zeros((n, dim))
    for item, (group, value) in rows.items():
        states[item, ord(group) - ord("a")] = 1.0
        states[item] = ar.l2norm(states[item])
    return states


def test_triplet_scores_uses_the_anchor_negative_pair():
    import numpy as np
    # item 0 anchor, item 1 positive (shares "a"), item 2 negative (shares "b")
    states = _one_hot_cosine_states({0: ("a", 1), 1: ("a", 2), 2: ("b", 3)})
    pos, neg = ar.triplet_scores(states, np.array([[0, 1]]), np.array([2]))
    assert pos.tolist() == [1.0]
    assert neg.tolist() == [0.0]


def test_triplet_scores_reads_the_negative_item_not_an_array_index():
    import numpy as np
    # The superseded reading was `positive_scores[negative_id]`, i.e. it indexed the
    # positive-score array with an *item id*.  Pick negatives that collide with other
    # positive rows so that reading returns a plausible but wrong number.
    states = _one_hot_cosine_states({0: ("a", 1), 1: ("a", 2), 2: ("b", 3),
                                     3: ("c", 4), 4: ("d", 5), 5: ("e", 6)})
    pairs = np.array([[0, 1], [1, 3], [2, 4], [3, 5]])
    negatives = np.array([1, 0, 3, 2])
    pos, neg = ar.triplet_scores(states, pairs, negatives)
    assert pos.tolist() == [1.0, 0.0, 0.0, 0.0]
    # cos(h[0],h[1]) = 1, cos(h[1],h[0]) = 1, cos(h[2],h[3]) = 0, cos(h[3],h[2]) = 0
    assert neg.tolist() == [1.0, 1.0, 0.0, 0.0]
    superseded = pos[negatives]
    assert superseded.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert not np.allclose(superseded, neg), "the array-index reading must differ here"


def test_triplet_scores_rejects_mismatched_shapes():
    import numpy as np
    states = np.eye(4)
    with pytest.raises(ValueError):
        ar.triplet_scores(states, np.array([[0, 1], [1, 2]]), np.array([3]))


def test_usable_triplets_requires_anchor_positive_and_negative():
    import numpy as np
    pairs = np.array([[0, 1], [2, 3], [4, 5]])
    negatives = np.array([6, 7, 8])
    covered = np.ones(9, dtype=bool)
    assert ar.usable_triplets(pairs, negatives, covered).all()
    covered[4] = False                     # anchor of the third triplet
    assert ar.usable_triplets(pairs, negatives, covered).tolist() == [True, True, False]
    covered[4] = True
    covered[8] = False                     # negative of the third triplet
    got = ar.usable_triplets(pairs, negatives, covered).tolist()
    assert got == [True, True, False]


def test_paired_auc_difference_matches_the_point_estimate():
    import numpy as np
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 0.5, 400)
    neg = rng.normal(0.0, 0.5, 400)
    got = ar.paired_auc_difference(pos, neg, pos + 0.6, neg, rounds=20)
    expected = ar.auc_score(pos + 0.6, neg) - ar.auc_score(pos, neg)
    assert got["delta_auc"] == pytest.approx(expected)
    assert got["p05"] <= got["delta_auc"] <= got["p95"]


def test_prefix_triplet_score_pairs_the_anchor_with_its_negative():
    import numpy as np
    codes = np.array([[1, 2, 3], [1, 2, 9], [1, 5, 0], [4, 4, 4]])
    pos, neg = ar.prefix_triplet_score(codes, np.array([[0, 1]]), np.array([2]))
    assert pos.tolist() == [2]
    assert neg.tolist() == [1]


# ---------------------------------------------------------- neighbour agreement
def test_overlap_fraction_is_not_jaccard():
    import numpy as np
    a = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    b = np.array([[0, 1, 2, 3, 4, 10, 11, 12, 13, 14]])
    got = ar.jaccard_overlap_statistics(a, b, 10)
    assert got["shared"].tolist() == [5]
    assert got["jaccard"].tolist() == [pytest.approx(5 / 15)]
    assert got["overlap_fraction"].tolist() == [pytest.approx(0.5)]


def test_overlap_fraction_edge_cases():
    import numpy as np
    same = np.arange(10)[None]
    got = ar.jaccard_overlap_statistics(same, same, 10)
    assert got["jaccard"][0] == pytest.approx(1.0)
    assert got["overlap_fraction"][0] == pytest.approx(1.0)
    disjoint = np.arange(10, 20)[None]
    got = ar.jaccard_overlap_statistics(same, disjoint, 10)
    assert got["jaccard"][0] == 0.0 and got["overlap_fraction"][0] == 0.0


def test_within_anchor_contrast_separates_direction():
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "relation"))
    import audit_relation as ar
    import numpy as np
    # item 0 is the anchor, items 1..5 are all its held-out partners, item 5 also
    # appears as a held-out partner of item 1 (so the control pair 5-1 exists)
    states = np.array([
        [1.0, 0.0],   # 0 anchor
        [0.6, 0.8],   # 1 partner, similar to 0
        [0.6, 0.8],
        [0.6, 0.8],
        [0.6, 0.8],
        [0.0, 1.0],   # 5 partner of 0 and of 1, but orthogonal to 0
    ])
    positives = np.array([1, 2, 3, 4, 5])
    anchors = np.zeros(5, dtype=np.int64)
    negatives = np.array([5, 5, 5, 5, 5])
    got = ar.within_anchor_contrast("x", {"x": states}, positives, negatives, anchors)
    assert got["forward"] >= 1
    # forward: cos(0,5)=0 beats cos(0,b)=0.6 -> loses, so the win rate is not 1
    assert got["forward_win_rate"] < 1.0
    # the reverse direction is the control: it uses item 5's own held-out partner, and
    # for a candidate that is orthogonal to the anchor the control loses
    assert got["reverse_win_rate"] == 0.0
    assert got["mean_forward_margin"] > got["mean_reverse_margin"]


def test_graph_precision_reports_mean_degree_not_a_fake_chance():
    import numpy as np
    neighbours = [{1, 2}, {0}, {0}]
    probes = np.array([0])
    block = np.array([[0.0, 0.9, 0.1]])
    assert ar.graph_precision(block, probes, neighbours, k=1) == 1.0
