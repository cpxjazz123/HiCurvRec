"""v2 PPR acceptance tests (issue #246 C21)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.graph_infra import graph_distance, ppr  # noqa: E402


def build(edges, n):
    rows = np.array([a for a, _ in edges], dtype=np.int64)
    cols = np.array([b for _, b in edges], dtype=np.int64)
    return sp.coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n)).tocsr()


def test_mass_conserves_with_a_dangling_row():
    csr = build([(0, 1), (1, 2)], 4)          # 2 and 3 are dangling
    result = ppr.solve(csr, [0])
    assert abs(result["rows"][0].sum() - 1.0) < 1e-9, result["rows"][0].sum()
    assert result["rows"][0][0] >= 0.15


def test_matches_the_iterative_definition():
    csr = build([(0, 1), (1, 2), (2, 0), (0, 2)], 3)
    solved = ppr.solve(csr, [0], alpha=0.15)["rows"][0]
    iterative = graph_distance.ppr(csr, 0, restart=0.15, tolerance=1e-12, max_iterations=20000,
                                   top_k=None)
    dense = np.zeros(3)
    dense[iterative["nodes"]] = iterative["scores"]
    # the iterative v1 path returns mass to the source too; compare on the same convention
    assert abs(dense.sum() - 1.0) < 1e-6
    assert np.allclose(dense, solved, atol=2e-3), (dense, solved)


def test_two_sources_are_independent():
    csr = build([(0, 1), (1, 2), (2, 0)], 4)
    result = ppr.solve(csr, [0, 1])
    assert not np.allclose(result["rows"][0], result["rows"][1])
    assert result["rows"][0][0] > result["rows"][0][1]
    assert result["rows"][1][1] > result["rows"][1][0]


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
