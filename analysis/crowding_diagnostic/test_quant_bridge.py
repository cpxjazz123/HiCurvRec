import numpy as np

from quant_bridge import (
    anchor_bootstrap_rows,
    cross_tabulate,
    decision_verdict,
    safe_er,
)


def test_four_cells_partition_comparisons():
    correct = np.array([True, True, False, False, True])
    shared = np.array([True, False, True, False, True])
    table = cross_tabulate(correct, shared)
    assert sum(cell["count"] for cell in table["cells"].values()) == len(correct)
    assert table["n_comparisons"] == len(correct)


def test_perfect_continuous_and_separated_prefix_has_no_quantization_loss():
    table = cross_tabulate(np.array([True, True]), np.array([False, False]))
    assert table["cells"]["quantization_loss"]["count"] == 0


def test_unseparable_continuous_and_shared_prefix_is_crowding():
    table = cross_tabulate(np.array([False, False, False]), np.array([True, True, True]))
    assert table["cells"]["continuous_crowding"]["count"] == 3
    assert all(cell["count"] == 0 for name, cell in table["cells"].items() if name != "continuous_crowding")


def test_er_undefined_only_for_zero_negative_share():
    assert safe_er(0.4, 0.0) is None
    assert safe_er(0.0, 0.0) is None
    assert safe_er(0.4, 0.2) == 2.0


def test_decision_verdict_branches_and_boundary():
    assert decision_verdict([3, 2, 1], [0.8, 0.8, 0.8], [0.1, 0.1, 0.1]) == "NO EVIDENCE FOR INCREASED CROWDING"
    assert decision_verdict([3, 2, 1], [0.8, 0.7, 0.6], [0.1, 0.2, 0.3]) == "EVIDENCE FOR INCREASED CROWDING"
    assert decision_verdict([1, 1, 1], [0.8, 0.8, 0.8], [0.1, 0.1, 0.1]) == "NO EVIDENCE FOR INCREASED CROWDING"


def test_anchor_bootstrap_resamples_whole_clusters():
    anchors = np.array([10, 10, 20, 20, 20])
    samples = anchor_bootstrap_rows(anchors, replicates=20, seed=7)
    for rows in samples:
        for anchor in np.unique(anchors):
            group = np.flatnonzero(anchors == anchor)
            assert (np.all(np.isin(group, rows)) or not np.any(np.isin(group, rows)))
