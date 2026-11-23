"""v2 pair-bank acceptance tests (issue #246 C6-C9, C20-C22)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.graph_infra import pairs  # noqa: E402


def build(edges, n):
    rows = np.array([a for a, _ in edges], dtype=np.int64)
    cols = np.array([b for _, b in edges], dtype=np.int64)
    csr = sp.coo_matrix((np.ones(len(edges), dtype=np.int64), (rows, cols)), shape=(n, n)).tocsr()
    return csr, csr.T.tocsr()


def test_bhattacharyya_distinguishes_missing_from_zero():
    assert pairs.bhattacharyya({0: 1.0}, {}) is None, "an empty distribution is missing, not 0"
    assert pairs.bhattacharyya({0: 1.0}, {1: 1.0}) == 0.0, "disjoint support is a real zero"


def test_jaccard_missing_when_the_union_is_empty():
    assert pairs.jaccard(set(), set()) is None
    assert abs(pairs.jaccard({1, 2}, {2, 3}) - 1 / 3) < 1e-12


def test_degree_matching_is_verified_pair_by_pair():
    # 0,1,2 share (in=1, out=1); 3 has a different degree
    edges = [(3, 0), (0, 4), (3, 1), (1, 5), (3, 2), (2, 6)]
    out_csr, in_csr = build(edges, 8)
    produced = pairs.degree_matched_pairs(out_csr, in_csr, 8, np.random.default_rng(0), 4)
    report = pairs.verify_degree_match(out_csr, in_csr, produced)
    assert report["violations"] == 0
    assert report["pairs"] == len(produced) and len(produced) > 0


def test_label_evidence_names_a_witness():
    edges = [(0, 7), (1, 7), (2, 3), (3, 4)]
    out_csr, in_csr = build(edges, 8)
    assert pairs.label_evidence(out_csr, in_csr, (0, 1), "shared_successor")["witness"] == [7]
    assert pairs.label_evidence(out_csr, in_csr, (2, 4), "two_hop_chain")["witness"] == [3]
    assert pairs.label_evidence(out_csr, in_csr, (0, 1), "direct")["witness"] == [0, 1]


def test_unobserved_is_not_a_negative_label():
    edges = [(0, 1)]
    out_csr, in_csr = build(edges, 6)
    record = pairs.pair_record(out_csr, in_csr, (2, 3), group="degree_matched")
    assert record["label_kind"] == "degree_matched_unobserved"
    assert "negative" not in record


def test_two_hop_group_is_disjoint_from_direct_edges():
    edges = [(0, 1), (1, 2), (2, 0)]
    csr, _ = build(edges, 3)
    produced = pairs.two_hop_pairs(csr, 3, np.random.default_rng(0), per_group=10)
    for left, right in produced.tolist():
        assert csr[left, right] == 0 and csr[right, left] == 0


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
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
