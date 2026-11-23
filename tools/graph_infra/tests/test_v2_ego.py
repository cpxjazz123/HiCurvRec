"""v2 ego acceptance tests (issue #246 C10-C16)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.graph_infra import topology  # noqa: E402


def build(edges, n):
    rows = np.array([a for a, _ in edges], dtype=np.int64)
    cols = np.array([b for _, b in edges], dtype=np.int64)
    csr = sp.coo_matrix((np.ones(len(edges), dtype=np.int64), (rows, cols)), shape=(n, n)).tocsr()
    return csr, csr.T.tocsr()


def test_direction_reads_real_csr_membership_beyond_the_sample_budget():
    # root 0 has 12 out-neighbours (more than K1_OUT = 8) and no in-neighbour
    edges = [(0, k) for k in range(1, 13)]
    out_csr, in_csr = build(edges, 13)
    block = topology.ego_induced(0, out_csr, in_csr, np.random.default_rng(0), radius=1, cap=32)
    non_root = block["nodes"] != 0
    assert block["role_out"][non_root].all(), "every included out-neighbour must be marked out"
    assert not block["role_in"][non_root].any()
    assert (block["direction"][non_root] == 1).all()
    full = topology.ego_full(0, out_csr, in_csr, radius=1)
    assert int(full["role_out"].sum()) == 12, "the full ego must keep all 12 out-neighbours"


def test_reciprocal_neighbour_keeps_both_masks():
    out_csr, in_csr = build([(0, 1), (1, 0)], 3)
    block = topology.ego_induced(0, out_csr, in_csr, np.random.default_rng(0), radius=1, cap=8)
    index = int(np.flatnonzero(block["nodes"] == 1)[0])
    assert block["role_out"][index] and block["role_in"][index]
    assert block["direction"][index] == 3, "mutual is its own role, not 'out'"


def test_hop_is_the_shortest_hop_not_the_traversal_depth():
    # 0 - 1 - 2 - 3 and a sampled budget that reaches 3 through 1 then 2
    out_csr, in_csr = build([(0, 1), (1, 2), (2, 3)], 4)
    block = topology.ego_induced(0, out_csr, in_csr, np.random.default_rng(0), radius=2, cap=8)
    hops = dict(zip(block["nodes"].tolist(), block["hop"].tolist()))
    depths = dict(zip(block["nodes"].tolist(), block["traversal_depth"].tolist()))
    assert hops[0] == 0
    assert hops[1] == 1 and depths[1] == 1
    assert hops[2] == 2
    # every recorded hop must equal the reference BFS distance
    distance = topology.reference_hops(0, out_csr, in_csr, radius=2)
    for node, hop in hops.items():
        assert hop == distance[node], (node, hop, distance[node])


def test_traversal_depth_and_true_hop_can_differ():
    # 12 out-neighbours: the budget keeps 8 at depth 1, the rest are reached at depth 2
    edges = [(0, k) for k in range(1, 13)] + [(1, k) for k in range(9, 13)]
    out_csr, in_csr = build(edges, 13)
    block = topology.ego_induced(0, out_csr, in_csr, np.random.default_rng(0), radius=2, cap=32)
    hop = dict(zip(block["nodes"].tolist(), block["hop"].tolist()))
    depth = dict(zip(block["nodes"].tolist(), block["traversal_depth"].tolist()))
    differing = [node for node in hop if hop[node] != depth[node]]
    assert differing, "the sampled traversal depth must differ from the true hop somewhere"
    for node in differing:
        assert hop[node] == 1, "a node found at depth 2 that is a direct neighbour has true hop 1"


def test_root_local_is_explicit_when_the_root_has_the_largest_id():
    out_csr, in_csr = build([(9, 1), (9, 2)], 10)
    block = topology.ego_induced(9, out_csr, in_csr, np.random.default_rng(1), radius=1, cap=4)
    assert block["root_local"] == int(np.flatnonzero(block["nodes"] == 9)[0])
    assert block["nodes"][block["root_local"]] == 9


def test_cap_counts_neighbours_not_the_root():
    edges = [(0, k) for k in range(1, 6)]
    out_csr, in_csr = build(edges, 6)
    block = topology.ego_induced(0, out_csr, in_csr, np.random.default_rng(0), radius=1, cap=3)
    assert len(block["nodes"]) <= 4, "cap=3 must allow at most 3 neighbours plus the root"
    assert block["cap"] == 3 and block["cap_includes_root"] is False


def test_ego_full_is_complete_and_induced():
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 0)]
    out_csr, in_csr = build(edges, 4)
    block = topology.ego_full(0, out_csr, in_csr, radius=1)
    assert set(block["nodes"].tolist()) == {0, 1, 2, 3}
    local = {int(n): i for i, n in enumerate(block["nodes"])}
    expected = {(local[a], local[b]) for a, b in edges}
    assert {tuple(edge) for edge in block["edges"].tolist()} == expected
    assert int(block["role_out"].sum()) == 2 and int(block["role_in"].sum()) == 1


def test_batch_egos_returns_edges_masks_and_features():
    edges = [(0, 1), (0, 2), (1, 2)]
    out_csr, in_csr = build(edges, 3)
    first = topology.ego_induced(0, out_csr, in_csr, np.random.default_rng(0), radius=1, cap=8)
    second = topology.ego_induced(1, out_csr, in_csr, np.random.default_rng(0), radius=1, cap=8)
    batch = topology.batch_egos([first, second], [0, 1])
    for key in ("nodes", "node_mask", "edges", "edge_mask", "features", "root_row"):
        assert key in batch, key
    assert batch["edges"].shape[0] == 2 and batch["edges"].shape[2] == 2
    assert batch["features"].shape[:2] == batch["nodes"].shape
    assert batch["edge_mask"].dtype == bool
    assert int(batch["edge_mask"].sum()) == len(first["edges"]) + len(second["edges"])


def test_features_mark_root_and_mutual():
    out_csr, in_csr = build([(0, 1), (1, 0), (0, 2)], 3)
    block = topology.ego_induced(0, out_csr, in_csr, np.random.default_rng(0), radius=1, cap=8)
    names = list(block["feature_names"])
    features = block["features"]
    root_row = block["root_local"]
    assert features[root_row, names.index("is_root")] == 1
    row = {int(n): i for i, n in enumerate(block["nodes"])}[1]
    assert features[row, names.index("is_mutual")] == 1


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
