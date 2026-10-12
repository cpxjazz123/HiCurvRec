import numpy as np

from pair_prefix import (classify_prefix, lcp_max_auc, lcp_pairwise_auc,
                         longest_common_prefix)
from quant_bridge import (
    anchor_bootstrap_rows,
    auc_from_distances,
    pairwise_continuous_correct,
    safe_er,
)


def test_positive_prefix_advantage_is_success_not_loss():
    result = classify_prefix(np.array([True]), np.array([[False, False]]))
    assert result["positive_prefix_advantage"].tolist() == [True]
    assert result["negative_prefix_advantage"].tolist() == [False]


def test_negative_prefix_advantage_is_failure():
    result = classify_prefix(np.array([False]), np.array([[True, False]]))
    assert result["negative_prefix_advantage"].tolist() == [True]


def test_both_share_and_neither_share_are_separate_ties():
    result = classify_prefix(np.array([True, False]), np.array([[True, False], [False, False]]))
    assert result["both_share"].tolist() == [True, False]
    assert result["neither_shares"].tolist() == [False, True]
    assert result["tie_rate"] == 1.0
    assert result["both_share_rate"] == result["neither_shares_rate"] == 0.5


def test_four_classes_partition_every_triple():
    result = classify_prefix(
        np.array([True, True, False, False]),
        np.array([[False, False], [True, False], [True, False], [False, False]]),
    )
    assert sum(int(result[key].sum()) for key in (
        "positive_prefix_advantage", "negative_prefix_advantage", "both_share", "neither_shares"
    )) == 4


def test_pairwise_continuous_criterion_differs_from_anchor_relative():
    positive = 2.0
    nearest_successor = 1.0
    matched_negative = 3.0
    old_positive_predicate = positive > nearest_successor
    pairwise = bool(pairwise_continuous_correct([positive], [[matched_negative]])[0, 0])
    assert positive > nearest_successor and positive < matched_negative
    assert old_positive_predicate and pairwise

    # The old predicate labels a farther-than-nearest positive "correct" even
    # when its matched negative is actually closer.
    matched_negative = 1.5
    pairwise = bool(pairwise_continuous_correct([positive], [[matched_negative]])[0, 0])
    assert old_positive_predicate is True
    assert pairwise is False


def test_lcp_ignores_collision_extension_token():
    left = np.array([[4, 8, 2, 0]])
    right = np.array([[4, 8, 2, 9]])
    assert longest_common_prefix(left, right).tolist() == [3]


def test_pairwise_lcp_convention_on_hand_checkable_tokens():
    anchor = np.array([[1, 2, 3]])
    positive = np.array([[1, 2, 8]])
    negatives = np.array([[4, 5, 6], [1, 9, 7]])
    positive_lcp = longest_common_prefix(anchor, positive)
    negative_lcp = longest_common_prefix(np.repeat(anchor, 2, axis=0), negatives)
    assert positive_lcp.tolist() == [2]
    assert negative_lcp.tolist() == [0, 1]
    assert lcp_pairwise_auc(positive_lcp, negative_lcp.reshape(1, -1)) == 1.0


def test_max_based_lcp_convention_is_distinct_from_pairwise():
    positive_lcp = np.array([1])
    negative_lcp = np.array([[0, 2]])
    assert lcp_pairwise_auc(positive_lcp, negative_lcp) == 0.5
    assert lcp_max_auc(positive_lcp, negative_lcp) == 0.0


def test_auc_direction_and_tie_handling():
    assert auc_from_distances(np.array([1.0]), np.array([[2.0, 3.0]])) == 1.0
    assert auc_from_distances(np.array([2.0]), np.array([[1.0, 1.5]])) == 0.0
    assert auc_from_distances(np.array([1.0]), np.array([[1.0, 2.0]])) == 0.75
    assert auc_from_distances(np.array([1.0]), np.array([[1.0]])) == 0.5


def test_anchor_bootstrap_resamples_whole_clusters():
    anchors = np.array([10, 10, 20, 20, 20])
    samples = anchor_bootstrap_rows(anchors, replicates=20, seed=7)
    for rows in samples:
        for anchor in np.unique(anchors):
            group = np.flatnonzero(anchors == anchor)
            assert (np.all(np.isin(group, rows)) or not np.any(np.isin(group, rows)))


def test_er_undefined_only_for_zero_negative_share():
    assert safe_er(0.4, 0.0) is None
    assert safe_er(0.0, 0.0) is None
    assert safe_er(0.4, 0.2) == 2.0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} passed")
