"""Typed motif tests on hand-built graphs (issue #245 §3/§7).

Each expectation is computed by hand from the edge list, not read back from the implementation.
The last test states the contract the issue cares about: a type comes from the graph, so two
different samplings of the same ego cannot produce different labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.graph_infra import graphlet, topology  # noqa: E402

# hand-built directed graph, one root per motif
#   0 -> 1 -> 2                    chain
#   3 -> 4, 3 -> 5 (no 4-5)        fork
#   6 <- 7, 6 <- 8 (no 7-8)        merge
#   9 <-> 10                       reciprocal
#   11 -> 12 -> 13 -> 11           triangle
#   14 -> 15, 14 -> 16, 15 <-> 16  NOT a fork (the successors are adjacent)
#   18 -> 17, 19 -> 17, 18 <-> 19  NOT a merge
EDGES = [(0, 1), (1, 2), (3, 4), (3, 5), (7, 6), (8, 6), (9, 10), (10, 9),
         (11, 12), (12, 13), (13, 11),
         (14, 15), (14, 16), (15, 16), (16, 15),
         (18, 17), (19, 17), (18, 19), (19, 18)]
NODES = 20


def build():
    rows = np.array([a for a, _ in EDGES], dtype=np.int64)
    cols = np.array([b for _, b in EDGES], dtype=np.int64)
    out_csr = sp.coo_matrix((np.ones(len(EDGES), dtype=np.int64), (rows, cols)),
                            shape=(NODES, NODES)).tocsr()
    in_csr = out_csr.T.tocsr()
    undirected = ((out_csr + in_csr) > 0).astype(np.int64).tocsr()
    return out_csr, in_csr, undirected


def test_chain():
    out_csr, in_csr, undirected = build()
    labels = graphlet.root_motifs(0, out_csr, in_csr, undirected)
    assert labels["chain"] is True
    assert labels["fork"] is False and labels["merge"] is False
    assert labels["triangle"] is False and labels["reciprocal"] is False


def test_fork_requires_an_independent_successor_pair():
    out_csr, in_csr, undirected = build()
    labels = graphlet.root_motifs(3, out_csr, in_csr, undirected)
    assert labels["fork"] is True
    assert labels["chain"] is False and labels["merge"] is False


def test_fork_rejects_adjacent_successors():
    out_csr, in_csr, undirected = build()
    labels = graphlet.root_motifs(14, out_csr, in_csr, undirected)
    assert labels["degree_out"] == 2
    assert labels["fork"] is False


def test_merge_rejects_adjacent_predecessors():
    out_csr, in_csr, undirected = build()
    labels = graphlet.root_motifs(17, out_csr, in_csr, undirected)
    assert labels["degree_in"] == 2
    assert labels["merge"] is False


def test_merge_requires_an_independent_predecessor_pair():
    out_csr, in_csr, undirected = build()
    labels = graphlet.root_motifs(6, out_csr, in_csr, undirected)
    assert labels["merge"] is True
    assert labels["fork"] is False


def test_reciprocal_needs_edges_in_both_directions():
    out_csr, in_csr, undirected = build()
    assert graphlet.root_motifs(9, out_csr, in_csr, undirected)["reciprocal"] is True
    assert graphlet.root_motifs(10, out_csr, in_csr, undirected)["reciprocal"] is True
    assert graphlet.root_motifs(0, out_csr, in_csr, undirected)["reciprocal"] is False


def test_triangle_is_not_a_chain():
    out_csr, in_csr, undirected = build()
    labels = graphlet.root_motifs(11, out_csr, in_csr, undirected)
    assert labels["triangle"] is True
    assert labels["chain"] is False


def test_labels_come_from_the_graph_not_from_a_sampling():
    out_csr, in_csr, undirected = build()
    first = graphlet.root_motifs(0, out_csr, in_csr, undirected)
    # a different sampling of the same ego must not be able to change the type
    for seed in (1, 2, 3):
        topology.ego_induced(0, out_csr, in_csr, np.random.default_rng(seed), radius=1, cap=8)
    second = graphlet.root_motifs(0, out_csr, in_csr, undirected)
    assert {k: v for k, v in first.items()} == {k: v for k, v in second.items()}


def test_report_counts_and_overlaps():
    out_csr, in_csr, undirected = build()
    report = graphlet.label_nodes(out_csr, in_csr, undirected, range(NODES))
    assert report["per_type"]["fork"]["items"] == 1  # node 14 is adjacent, so not a fork
    assert report["per_type"]["merge"]["items"] == 1  # node 17 likewise
    assert report["per_type"]["triangle"]["items"] == 9  # three hand-built triangles
    assert report["n_items"] == NODES
    assert report["overlap"]["fork+merge"] == 0


def main() -> int:
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    if not tests:
        print("FAIL no tests collected")
        return 1
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
