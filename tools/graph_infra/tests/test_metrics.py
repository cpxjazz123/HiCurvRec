"""Hand-checked production metrics (issue #244 §2)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.graph_infra import metrics  # noqa: E402


def approx(actual, expected, tol=1e-12, label=""):
    assert actual is not None, f"{label}: got None"
    assert abs(float(actual) - float(expected)) <= tol, f"{label}: {actual} != {expected}"


def test_average_ranks_with_ties():
    ranks = metrics.average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0], ranks


def test_spearman_perfect_and_tied():
    approx(metrics.spearman([1, 2, 3, 4], [10, 20, 60, 120]), 1.0, 1e-12, "monotone")
    approx(metrics.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0, 1e-12, "reversed")
    # hand value for x=[1,2,2,3], y=[1,2,3,4]: ranks [1,2.5,2.5,4] vs [1,2,3,4]
    rx = np.array([1.0, 2.5, 2.5, 4.0]); ry = np.array([1.0, 2.0, 3.0, 4.0])
    rx -= rx.mean(); ry -= ry.mean()
    expected = float((rx * ry).sum() / (np.linalg.norm(rx) * np.linalg.norm(ry)))
    approx(metrics.spearman([1, 2, 2, 3], [1, 2, 3, 4]), expected, 1e-12, "ties")


def test_spearman_separates_from_pearson():
    x = [1, 2, 3, 4]
    y = [10, 20, 60, 120]
    approx(metrics.spearman(x, y), 1.0, 1e-12, "spearman")
    pearson = float(np.corrcoef(x, y)[0, 1])
    assert abs(pearson - 1.0) > 1e-3, "the case must separate the two"


def test_spearman_constant_is_na():
    assert metrics.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert metrics.spearman([1, 2, 3], [5, 5, 5]) is None


def test_jaccard_real_intersection_over_union():
    approx(metrics.jaccard({1, 2, 3}, {2, 3, 4}), 0.5, 1e-12, "jaccard")
    approx(metrics.jaccard(set(), set()), 0.0, 1e-12, "empty")
    approx(metrics.jaccard({1, 2}, {1, 2}), 1.0, 1e-12, "identical")


def test_neighbour_jaccard_on_a_known_graph():
    # 0 -> {1,2}, 1 -> {2,3}: undirected neighbourhoods {1,2} vs {0,2,3}
    rows = [0, 0, 1, 1]; cols = [1, 2, 2, 3]
    data = np.ones(4, dtype=np.int64)
    out_csr = sp.csr_matrix((data, (rows, cols)), shape=(4, 4)); out_csr.sort_indices()
    in_csr = out_csr.T.tocsr()
    # undirected N(0)={1,2}, N(1)={0,2,3}: intersection {2}, union size 4
    approx(metrics.neighbour_jaccard(out_csr, in_csr, 0, 1), 0.25, 1e-12, "undirected")
    # directed: out(0)={1,2} vs out(1)={2,3} -> 1/3 ; in(0)={} vs in(1)={0} -> 0
    approx(metrics.neighbour_jaccard(out_csr, in_csr, 0, 1, directed=True), (1.0 / 3.0) / 2.0, 1e-12, "directed")


def test_prefix_share():
    assert metrics.prefix_share([1, 2, 3, 9], [1, 2, 7, 9], 3) == 2
    assert metrics.prefix_share([1, 2, 3], [4, 2, 3], 3) == 0
    assert metrics.prefix_share([1, 2, 3], [1, 2, 3], 3) == 3


def test_pearson_and_constant_na():
    value = metrics.pearson([1, 2, 3], [2, 4, 6])
    assert value is not None and abs(value - 1.0) < 1e-12, value
    assert metrics.pearson([1, 1, 1], [1, 2, 3]) is None


def test_distortion_uses_the_unit_and_masks_unreachable():
    # unreachable means the GRAPH distance is infinite; a model distance of inf with a
    # finite graph distance is a different failure and must not be counted as unreachable
    report = metrics.distortion([1.0, 2.0, 5.0], [1.0, 2.0, np.inf])
    assert report["n"] == 2
    assert report["mean"] == 0.0
    assert report["unreachable"] == 1
    assert report["unit"] == "graph hops"


def test_prefix_at_k_is_positional():
    pairs = [([1, 2, 3], [1, 2, 9]), ([1, 2, 3], [1, 9, 3])]
    assert metrics.prefix_at_k(pairs, 2) == 0.5
    assert metrics.prefix_at_k(pairs, 1) == 1.0


def test_collision_and_duplicate_are_different_quantities():
    sids = [[1, 2, 3, 9], [1, 2, 3, 9], [1, 2, 4, 9]]
    assert abs(metrics.collision_rate(sids) - (1 - 2 / 3)) < 1e-12
    assert abs(metrics.duplicate_item_fraction(sids) - 2 / 3) < 1e-12


def test_codebook_stats_hand_values():
    report = metrics.codebook_stats(np.array([[0, 0], [0, 0], [1, 0]]))
    assert report[0]["used"] == 2
    assert abs(report[0]["max_share"] - 2 / 3) < 1e-12


def test_reconstruction_error_and_sid_comparison():
    assert abs(metrics.reconstruction_error([0, 0], [1, 1])["mse"] - 1.0) < 1e-12
    comparison = metrics.compare_sid_tables({0: [1, 2], 1: [3, 4]}, {0: [1, 2], 2: [5, 6]})
    assert comparison["differing"] == 0 and comparison["only_left"] == 1 and comparison["only_right"] == 1




def test_subgraph_summary_separates_the_denominators():
    report = metrics.subgraph_summary([np.array([1.0, 0.0]), np.array([1.0, 1.0])], threshold=0.5,
                                      direction="maximize")
    assert report["centers"] == 2 and report["subgraphs"] == 4
    assert abs(report["subgraph_pass_fraction"] - 0.75) < 1e-12
    assert abs(report["center_all_pass_fraction"] - 0.5) < 1e-12
    assert abs(report["center_any_pass_fraction"] - 1.0) < 1e-12
    assert abs(report["worst"] - 0.5) < 1e-12, "maximize records the per-centre minimum"


def test_stratified_absent_strata_are_absent():
    report = metrics.stratified([1.0, 2.0, 3.0], ["a", "a", "b"])
    assert report["a"]["n"] == 2 and abs(report["a"]["mean"] - 1.5) < 1e-12
    assert report["b"]["n"] == 1 and abs(report["b"]["mean"] - 3.0) < 1e-12
    assert set(report) == {"a", "b"}


def main() -> int:
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001
            failures.append((name, str(error)))
            print(f"FAIL {name}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


