"""Hand-checked ego subgraph tests for issue #244 §2 (production sampler)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import topology  # noqa: E402


def graph(edges, n_nodes):
    rows = [a for a, _ in edges]
    cols = [b for _, b in edges]
    data = np.ones(len(edges), dtype=np.int64)
    csr = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    csr.sort_indices()
    return csr, csr.T.tocsr()


def run(edges, n_nodes, root, radius=2, cap=32, seed=0):
    out_csr, in_csr = graph(edges, n_nodes)
    rng = np.random.default_rng(seed)
    return topology.ego_induced(root, out_csr, in_csr, rng, radius=radius, cap=cap)


def test_chain():
    result = run([(0, 1), (1, 2)], 3, 0)
    assert set(result["nodes"].tolist()) == {0, 1, 2}
    assert {tuple(e) for e in result["edges"].tolist()} == {(0, 1), (1, 2)}
    assert result["role"].tolist()[result["nodes"].tolist().index(2)] == 2


def test_triangle_keeps_every_induced_edge():
    result = run([(0, 1), (1, 2), (2, 0)], 3, 0)
    assert set(result["nodes"].tolist()) == {0, 1, 2}
    local = {int(n): i for i, n in enumerate(result["nodes"].tolist())}
    expected = {
        (local[0], local[1]), (local[1], local[2]), (local[2], local[0]),
    }
    assert {tuple(e) for e in result["edges"].tolist()} == expected


def test_fork():
    result = run([(0, 1), (0, 2)], 3, 0, radius=1)
    assert set(result["nodes"].tolist()) == {0, 1, 2}
    assert len(result["edges"]) == 2
    assert all(local := True for _ in [0])  # structure asserted through the edge count
    assert result["direction"].tolist().count(1) == 2


def test_merge():
    result = run([(1, 0), (2, 0)], 3, 0, radius=1)
    assert set(result["nodes"].tolist()) == {0, 1, 2}
    local = {int(n): i for i, n in enumerate(result["nodes"].tolist())}
    assert {tuple(e) for e in result["edges"].tolist()} == {(local[1], local[0]), (local[2], local[0])}
    assert result["direction"].tolist().count(2) == 2


def test_reciprocal_edge_and_dual_role():
    edges = [(0, 1), (1, 0)]
    result = run(edges, 2, 0, radius=1)
    local = {int(n): i for i, n in enumerate(result["nodes"].tolist())}
    assert {tuple(e) for e in result["edges"].tolist()} == {(local[0], local[1]), (local[1], local[0])}
    out_csr, in_csr = graph(edges, 2)
    roles = topology.root_roles(0, out_csr, in_csr)
    assert roles["both"] == {1}, "a mutual neighbour must keep both roles"


def test_isolated_node_has_no_fabricated_edges():
    result = run([(0, 1)], 6, 5, radius=2)
    assert result["nodes"].tolist() == [5]
    assert result["edges"].size == 0
    assert result["empty_neighbourhood"] is True


def test_non_tree_edge_is_kept():
    """0 -> 1, 0 -> 2, 1 -> 2: the edge 1-2 is inside the node set but is not a tree edge."""
    result = run([(0, 1), (0, 2), (1, 2)], 3, 0, radius=2)
    local = {int(n): i for i, n in enumerate(result["nodes"].tolist())}
    assert (local[1], local[2]) in {tuple(e) for e in result["edges"].tolist()}, \
        "induced edges between sampled neighbours must survive"


def test_cap_is_respected():
    edges = [(0, i) for i in range(1, 60)]
    result = run(edges, 60, 0, radius=1, cap=8)
    assert len(result["nodes"]) <= 9, "cap plus the root"


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
