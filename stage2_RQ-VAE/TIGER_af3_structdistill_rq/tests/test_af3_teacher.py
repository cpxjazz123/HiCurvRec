"""Issue #181 acceptance 1: the frozen AF3 teacher on a toy graph, sparse against dense."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import af3_teacher as t  # noqa: E402

# a 4-node path 0-1-2-3 plus the closing edge 3-0, so every node has degree 2 and one triangle exists (0,1,...) no: it is a 4-cycle
CYCLE4 = [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_af3_on_a_cycle_is_minus_the_degree_plus_three():
    degrees = {0: 2, 1: 2, 2: 2, 3: 2}
    af3 = t.af3_values(CYCLE4, degrees)
    assert all(abs(v - 3.0) < 1e-12 for v in af3.values())     # 4 - 2 - 2 + 3 = 3


def test_quantiles_and_the_clipping_span():
    degrees = {0: 2, 1: 2, 2: 2, 3: 2}
    q05, q95 = t.quantiles(t.af3_values(CYCLE4, degrees))
    assert abs(q05 - 3.0) < 1e-12 and abs(q95 - 3.0) < 1e-12


def test_weights_keep_bridge_edges_strictly_positive():
    # a path 0-1-2-3 plus the chord 1-3, giving mixed degrees and therefore unequal AF3 values
    coo = [(0, 1), (1, 2), (2, 3), (1, 3)]
    degrees = {0: 1, 1: 3, 2: 2, 3: 2}
    af3 = t.af3_values(coo, degrees)
    q05, q95 = t.quantiles(af3)
    w = t.weights(coo, 4, q05, q95).to_dense()
    observed = {(i, j) for i, j in coo} | {(j, i) for i, j in coo}
    for i in range(4):
        for j in range(4):
            if (i, j) in observed:
                assert float(w[i, j]) >= 1.0          # never zero, even for the lowest-AF3 bridge
            elif i != j:
                assert float(w[i, j]) == 0.0
    assert torch.allclose(w, w.t())


def test_teacher_is_symmetric_and_reproduces_the_definition():
    coo = [(0, 1), (1, 2), (2, 3), (1, 3)]
    teacher, manifest = t.build(coo, 4)
    assert torch.allclose(teacher, teacher.t(), atol=1e-12)
    # the definition, rebuilt through the same public helpers the module exposes.
    # node 3 appears in (2,3) and (1,3), so its degree is 2, not 3 -- an earlier hand-written dict had 3 and made this test fail
    # against a correct teacher.
    degrees = {0: 1, 1: 3, 2: 2, 3: 2}
    q05, q95 = t.quantiles(t.af3_values(coo, degrees))
    p_dense = t.normalized(t.weights(coo, 4, q05, q95))
    assert torch.allclose(teacher, 0.5 * (p_dense + p_dense @ p_dense), atol=1e-12)
    assert manifest["n_items"] == 4 and manifest["q05"] == q95 or manifest["q05"] == q05


def test_an_isolated_node_keeps_a_zero_row():
    coo = [(0, 1)]
    teacher, manifest = t.build(coo, 3)
    assert float(teacher[2].abs().sum()) == 0.0
    assert float(teacher[:, 2].abs().sum()) == 0.0
    assert manifest["edges"] == 1


def test_the_teacher_is_deterministic():
    coo = [(0, 1), (1, 2), (2, 3), (1, 3)]
    first, m1 = t.build(coo, 4)
    second, m2 = t.build(coo, 4)
    assert torch.equal(first, second) and m1 == m2


def test_the_sparse_teacher_equals_the_dense_one():
    coo = [(0, 1), (1, 2), (2, 3), (1, 3)]
    dense, m1 = t.build(coo, 4)
    sparse, m2 = t.build_sparse(coo, 4)
    assert torch.allclose(sparse.to_dense().double(), dense, atol=1e-6)   # float32 storage, same definition
    assert m1["q05"] == m2["q05"] and m1["q95"] == m2["q95"]


def test_teacher_scores_read_the_requested_pairs():
    coo = [(0, 1), (1, 2), (2, 3), (1, 3)]
    sparse, _ = t.build_sparse(coo, 4)
    dense = sparse.to_dense()
    anchors = torch.tensor([0, 1])
    candidates = torch.tensor([[1, 2], [0, 3]])
    got = t.teacher_scores(sparse, anchors, candidates)
    for a, row in enumerate(anchors.tolist()):
        for c, item in enumerate(candidates[a].tolist()):
            assert abs(float(got[a, c]) - float(dense[row, item])) < 1e-12


def test_pack_and_unpack_round_trip_the_teacher_bit_for_bit():
    coo = [(0, 1), (1, 2), (2, 3), (1, 3)]
    sparse, manifest = t.build_sparse(coo, 4)
    payload = t.pack(sparse, manifest)
    restored, restored_manifest = t.unpack(payload)
    assert torch.equal(restored.to_dense(), sparse.to_dense())
    assert restored_manifest["q05"] == manifest["q05"] and restored_manifest["q95"] == manifest["q95"]
