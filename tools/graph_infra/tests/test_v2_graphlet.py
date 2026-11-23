"""v2 exact graphlet / orbit acceptance tests (issue #246 C17-C19)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import reference_hand as ref  # noqa: E402
from tools.graph_infra import graphlet  # noqa: E402


def build(edges, n):
    rows = np.array([a for a, _ in edges], dtype=np.int64)
    cols = np.array([b for _, b in edges], dtype=np.int64)
    csr = sp.coo_matrix((np.ones(len(edges), dtype=np.int64), (rows, cols)), shape=(n, n)).tocsr()
    return csr, csr.T.tocsr()


def test_fork_relaxed_accepts_two_adjacent_plus_one_isolated_successor():
    # successors {1,2,3} with 1-2 adjacent and 3 adjacent to neither
    out_csr, in_csr = build([(0, 1), (0, 2), (0, 3), (1, 2)], 5)
    assert graphlet.fork_relaxed(0, out_csr, in_csr) is True
    assert ref.fork_relaxed({0: {1, 2, 3}, 1: {2}}, 0) is True


def test_fork_relaxed_is_false_when_every_successor_pair_is_adjacent():
    out_csr, in_csr = build([(0, 1), (0, 2), (1, 2), (2, 1)], 3)
    assert graphlet.fork_relaxed(0, out_csr, in_csr) is False
    assert ref.fork_relaxed({0: {1, 2}, 1: {2}, 2: {1}}, 0) is False


def test_merge_relaxed_is_the_mirror():
    out_csr, in_csr = build([(1, 0), (2, 0), (3, 0), (1, 2)], 5)
    assert graphlet.merge_relaxed(0, out_csr, in_csr) is True


def test_canonicalisation_matches_the_reference_on_all_64_patterns():
    for code in range(64):
        assert graphlet.canonical_code(code) == ref.canonical_code(code), code


def test_orbits_match_the_reference_on_all_64_patterns():
    for code in range(64):
        assert graphlet.orbits(code) == ref.orbits(code), code


def test_canonical_classes_are_the_sixteen_unlabeled_digraphs():
    classes = {graphlet.canonical_code(code) for code in range(64)}
    assert len(classes) == 16, len(classes)


def test_instance_enumeration_matches_brute_force_triples():
    edges = [(0, 1), (1, 2), (0, 2), (2, 0), (3, 0), (1, 3), (3, 2), (4, 4), (4, 0)]
    out_csr, in_csr = build(edges, 5)
    adjacency = {node: set(out_csr.indices[out_csr.indptr[node]:out_csr.indptr[node + 1]].tolist())
                 for node in range(5)}
    produced = graphlet.motif_instances(0, out_csr, in_csr)
    found = {tuple(sorted((inst["node_ids"][0], inst["node_ids"][1], inst["node_ids"][2])))
             for inst in produced}
    expected = set()
    for triple in __import__("itertools").combinations(range(5), 3):
        if 0 in triple:
            expected.add(tuple(sorted(triple)))
    assert found == expected, (sorted(found), sorted(expected))
    for instance in produced:
        assert instance["graphlet_id"] == graphlet.canonical_code(instance["code"])
        assert instance["graph_version"].startswith("v2")
        assert instance["center_role"] in {(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (0, 0)}


def test_strict_out_star_instance_is_exactly_two_edges():
    out_csr, in_csr = build([(0, 1), (0, 2), (1, 2)], 3)
    instances = graphlet.motif_instances(0, out_csr, in_csr)
    families = sorted(instance["family"] for instance in instances)
    assert "ff_triangle" in families
    out_csr2, in_csr2 = build([(0, 1), (0, 2)], 3)
    instances2 = graphlet.motif_instances(0, out_csr2, in_csr2)
    assert [instance["family"] for instance in instances2] == ["out_star"]
    assert len(instances2[0]["edges"]) == 2, "a strict out_star instance has exactly two edges"
    assert instances2[0]["kind"] == "strict_instance"


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
