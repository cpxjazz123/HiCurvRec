"""Hand-checked pair-bank tests for issue #244 §2."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.graph_infra import pairs as bank  # noqa: E402

# 0 -> 1, 0 -> 2 ; 3 -> 1, 3 -> 2 ; 1 -> 4, 2 -> 4 ; 1 -> 5
EDGES = [(0, 1), (0, 2), (3, 1), (3, 2), (1, 4), (2, 4), (1, 5)]
N = 6


def graph(edges=EDGES, n=N):
    rows = [a for a, _ in edges]
    cols = [b for _, b in edges]
    data = np.ones(len(edges), dtype=np.int64)
    out_csr = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    out_csr.sort_indices()
    return out_csr, out_csr.T.tocsr()


def as_set(pairs):
    return {tuple(int(x) for x in row) for row in pairs}


def test_direct_group_is_edges():
    out_csr, in_csr = graph()
    rng = np.random.default_rng(0)
    pairs = bank.direct_pairs(out_csr, rng)
    assert as_set(pairs) == {tuple(e) for e in EDGES}
    bank.verify_group(out_csr, in_csr, pairs, "direct")


def test_shared_successor_is_not_a_chain():
    """0 and 3 both point at 1 and 2, so they are a shared-successor pair."""
    out_csr, in_csr = graph()
    rng = np.random.default_rng(0)
    pairs = bank.shared_successor_pairs(out_csr, N, rng)
    found = as_set(pairs)
    assert (0, 3) in found or (3, 0) in found, found
    bank.verify_group(out_csr, in_csr, pairs, "shared_out")
    # the #243 bug: a chain 0->1->4 must never be reported as a shared successor
    assert (0, 4) not in found and (4, 0) not in found


def test_shared_predecessor():
    out_csr, in_csr = graph()
    rng = np.random.default_rng(0)
    pairs = bank.shared_predecessor_pairs(in_csr, N, rng)
    found = as_set(pairs)
    assert (1, 2) in found or (2, 1) in found, found
    bank.verify_group(out_csr, in_csr, pairs, "shared_in")


def test_two_hop_pairs_are_paths_without_a_direct_edge():
    out_csr, in_csr = graph()
    rng = np.random.default_rng(0)
    pairs = bank.two_hop_pairs(out_csr, N, rng)
    found = as_set(pairs)
    assert (0, 4) in found, found
    assert (0, 1) not in found, "a direct edge is not a two-hop pair"
    bank.verify_group(out_csr, in_csr, pairs, "two_hop")


def test_degree_matched_has_no_direct_edge():
    out_csr, in_csr = graph()
    rng = np.random.default_rng(0)
    pairs = bank.degree_matched_pairs(out_csr, in_csr, N, rng)
    bank.verify_group(out_csr, in_csr, pairs, "degree_matched")
    for i, j in pairs:
        assert not bank._has(out_csr, int(i), int(j))


def test_no_self_pairs_or_duplicates():
    out_csr, in_csr = graph()
    rng = np.random.default_rng(1)
    for name, produced in (
        ("shared_out", bank.shared_successor_pairs(out_csr, N, rng)),
        ("shared_in", bank.shared_predecessor_pairs(in_csr, N, rng)),
        ("two_hop", bank.two_hop_pairs(out_csr, N, rng)),
    ):
        stats = bank.group_stats(produced)
        assert stats["self_pairs"] == 0, name
        assert stats["duplicates"] == 0, name


def test_verifier_rejects_a_wrong_type():
    """The verifier itself must fail closed on a mislabelled pair."""
    out_csr, in_csr = graph()
    wrong = np.array([[0, 4]], dtype=np.int64)  # a two-hop pair, not a shared successor
    try:
        bank.verify_group(out_csr, in_csr, wrong, "shared_out")
    except AssertionError:
        pass
    else:
        raise AssertionError("verifier accepted a chain as a shared-successor pair")




def test_bhattacharyya_hand_value():
    assert abs(bank.bhattacharyya({0: 0.5, 1: 0.5}, {0: 0.5, 2: 0.5}) - 0.5) < 1e-12
    assert bank.bhattacharyya({}, {0: 1.0}) == 0.0


def test_split_banks_use_only_the_given_adjacency():
    first = sp.coo_matrix((np.ones(1), (np.array([0]), np.array([1]))), shape=(4, 4)).tocsr()
    second = sp.coo_matrix((np.ones(1), (np.array([2]), np.array([3]))), shape=(4, 4)).tocsr()
    output = bank.split_banks(first, first.T.tocsr(), 4, {"a": first, "b": second},
                              np.random.default_rng(5), per_group=3)
    direct_a = output["a"]["direct"].tolist()
    assert direct_a and all(pair == [0, 1] for pair in direct_a)
    direct_b = output["b"]["direct"].tolist()
    assert direct_b and all(pair == [2, 3] for pair in direct_b)


def test_probe_bank_is_frozen():
    csr = sp.coo_matrix((np.ones(3), (np.array([0, 0, 1]), np.array([1, 2, 2]))),
                        shape=(4, 4)).tocsr()
    first = bank.probe_bank(csr, points=6, seed=7)
    second = bank.probe_bank(csr, points=6, seed=7)
    assert np.array_equal(first["pairs"], second["pairs"])
    assert np.array_equal(first["graph_hops"], second["graph_hops"])




def test_two_hop_excludes_the_reverse_direction():
    # 0 -> 1 -> 2 -> 0: every two-hop candidate has a reverse edge, so nothing may survive
    edges = [(0, 1), (1, 2), (2, 0)]
    csr = sp.coo_matrix((np.ones(3), (np.array([a for a, _ in edges]), np.array([b for _, b in edges]))),
                        shape=(3, 3)).tocsr()
    produced = bank.two_hop_pairs(csr, 3, np.random.default_rng(3), per_group=10)
    assert len(produced) == 0


def test_exclude_undirected_drops_the_mirror_image():
    blocked = sp.coo_matrix((np.ones(1), (np.array([3]), np.array([4]))), shape=(6, 6)).tocsr()
    kept = bank.exclude_undirected(np.array([[4, 3], [0, 1]]), blocked)
    assert kept.tolist() == [[0, 1]]


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
