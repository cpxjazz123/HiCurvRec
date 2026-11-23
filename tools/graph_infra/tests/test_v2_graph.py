"""v2 graph / split / isolation acceptance tests (issue #246 C1-C9, C35).

Each test names the production entry it exercises; the known answers come from
`reference_hand` or from explicit hand arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import reference_hand as ref  # noqa: E402
from tools.graph_infra import graphlet, metrics, topology  # noqa: E402

from tools.graph_infra import config  # noqa: E402

REPO = config.REPO


def test_held_out_support_uses_boolean_semantics():
    from tools.graph_infra import isolation

    # visible (7->8) weight 3; eval observes (7->8) once AND the unseen direction (8->7) once.
    # The v1 weighted formula gives 1 - 1*3 = -2 at (7,8); symmetrising then cancels the real
    # held-out direction (8,7).  The support formula must keep it.
    visible = sp.coo_matrix((np.array([3]), (np.array([7]), np.array([8]))),
                            shape=(12, 12)).tocsr()
    evaluation = sp.coo_matrix((np.array([1, 1]), (np.array([7, 8]), np.array([8, 7]))),
                               shape=(12, 12)).tocsr()
    held = isolation.held_out_support(visible, evaluation)
    assert held[8, 7] and held[7, 8], "the held-out direction must survive"
    v1 = evaluation - evaluation.multiply(visible)
    cancelled = int((v1 + v1.T)[8, 7])
    assert cancelled <= 0, f"v1 must be shown to cancel, got {cancelled}"


def test_three_way_split_disjoint_and_complete():
    from tools.graph_infra import splits

    users = np.arange(100, dtype=np.int64)
    result = splits.three_way(users, seed=246001)
    names = ["visible", "struct_val", "struct_test"]
    sets = [set(result[name].tolist()) for name in names]
    assert sum(len(s) for s in sets) == 100
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert set().union(*sets) == set(users.tolist())
    assert abs(len(sets[0]) / 100 - 0.8) < 0.02
    again = splits.three_way(users, seed=246001)
    assert np.array_equal(result["visible"], again["visible"]), "split must be reproducible"


def test_edge_counts_sum_to_the_full_graph():
    from tools.graph_infra import graph_build, splits

    sequences = [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]]
    users = np.array([0, 1, 2, 3], dtype=np.int64)
    result = splits.three_way(users, seed=246001)
    full = graph_build.count_edges(sequences)
    total = 0
    for name in ("visible", "struct_val", "struct_test"):
        subset = [sequences[i] for i in result[name].tolist()]
        total += graph_build.count_edges(subset).total()
    assert total == full.total()


def test_sequence_edges_match_the_csr_exactly():
    from tools.graph_infra import graph_build

    sequences = [[0, 1, 2, 1], [3, 1, 1, 2], [2, 0]]
    counts = graph_build.count_edges(sequences)
    csr = counts.to_csr(n_items=4)
    reference = ref.sequence_edges(sequences)
    coo = csr.tocoo()
    assert set(zip(coo.row.tolist(), coo.col.tolist())) == set(reference)
    for (left, right), weight in reference.items():
        assert int(csr[left, right]) == weight
    assert graph_build.self_loops(sequences) == ref.self_loops(sequences) == 1


def test_item_index_reads_the_actual_set():
    from tools.graph_infra import graph_build

    report = graph_build.item_index()
    ids = report["ids"]
    assert ids.dtype == np.int64
    assert len(ids) == len(set(ids.tolist()))
    assert ids.tolist() == list(range(len(ids))), "ids must be the actual contiguous set"
    assert report["embeddings"][0] == len(ids)


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
