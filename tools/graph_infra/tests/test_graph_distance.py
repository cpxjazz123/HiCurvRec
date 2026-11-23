"""Hand-checked graph-distance tests (issue #245 C12)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.graph_infra import graph_distance as gd  # noqa: E402


def graph(edges, n):
    rows = [a for a, _ in edges]
    cols = [b for _, b in edges]
    data = np.ones(len(edges), dtype=np.int64)
    csr = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    csr.sort_indices()
    return csr


def test_chain_distance():
    csr = graph([(0, 1), (1, 2), (2, 3)], 4)
    und = gd.undirected_projection(csr)
    d = gd.bfs_distances(und, [0])
    assert d[0, 0] == 0 and d[0, 3] == 3
    # direction is ignored: the undirected projection is symmetric
    assert gd.bfs_distances(und, [3])[0, 0] == 3


def test_unreachable_is_inf_and_masked():
    csr = graph([(0, 1)], 3)
    und = gd.undirected_projection(csr)
    d = gd.bfs_distances(und, [0])
    assert np.isinf(d[0, 2]), d
    assert gd.masked_mean(d[0], np.isfinite(d[0])) == 0.5


def test_depth_cap_marks_beyond_as_unreachable():
    csr = graph([(0, 1), (1, 2), (2, 3)], 4)
    und = gd.undirected_projection(csr)
    d = gd.bfs_distances(und, [0], depth=2)
    assert d[0, 2] == 2 and np.isinf(d[0, 3])


def test_subgraph_distance_ignores_outside_nodes():
    """The subgraph matrix only contains the requested nodes, and edges leaving it vanish."""
    csr = graph([(0, 1), (1, 2), (1, 3)], 4)
    inside = gd.subgraph_distances([0, 1, 2], csr)
    assert inside.shape == (3, 3), "node 3 must not appear in a {0,1,2} subgraph"
    assert inside[0, 2] == 2
    # {0,1,3} keeps 0->1->3 and drops 1->2
    other = gd.subgraph_distances([0, 1, 3], csr)
    assert other.shape == (3, 3)
    assert other[0, 2] == 2
    # a pair with no path inside the set stays unreachable
    lonely = gd.subgraph_distances([0, 2], graph([(0, 1), (1, 2)], 3), depth=0)
    assert np.isinf(lonely[0, 1])


def test_matches_the_reference_bfs():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    csr = graph(edges, 5)
    und = gd.undirected_projection(csr)
    for root in range(4):
        fast = gd.bfs_distances(und, [root])[0]
        reference = gd.reference_bfs(edges, 5, root)
        for node in range(4):
            assert fast[node] == reference[node], (root, node, fast[node], reference[node])
        assert np.isinf(fast[4]), "node 4 is isolated"




def test_ppr_converges_and_normalises_the_mass():
    edges = [(0, 1), (1, 2), (2, 3)]
    rows = np.array([a for a, _ in edges]); cols = np.array([b for _, b in edges])
    csr = sp.coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(4, 4)).tocsr()
    result = gd.ppr(csr, 0, restart=0.15, top_k=None)
    assert result["converged"] is True
    assert abs(result["scores"].sum() - 1.0) < 1e-6
    scores = dict(zip(result["nodes"].tolist(), result["scores"].tolist()))
    assert scores[0] > scores[1] > scores[2] > scores[3]
    assert result["unit"].startswith("propagation")


def test_pair_metrics_masks_and_reports_units():
    report = gd.pair_metrics([1.0, 3.0, np.inf], [1, 2, 2])
    assert report["n"] == 2 and report["masked_out"] == 1
    assert abs(report["bias"] - 0.5) < 1e-12
    assert report["unit"] == "graph hops"


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
