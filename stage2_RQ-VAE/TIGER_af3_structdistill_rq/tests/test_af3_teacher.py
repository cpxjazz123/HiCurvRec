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
    # a path 0-1-2-3 plus a triangle 1-2-3, giving mixed degrees and therefore unequal AF3 values
    coo = [(0, 1), (1, 2), (2, 3), (1, 3)]
    degrees = {0: 1, 1: 3, 2: 2, 3: 3}
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


def test_teacher_is_symmetric_with_unit_free_rows_and_matches_a_dense_computation():
    coo = [(0, 1), (1, 2), (2, 3), (1, 3)]
    teacher, manifest = t.build(coo, 4)
    assert torch.allclose(teacher, teacher.t(), atol=1e-12)
    # hand computation through the definition
    degrees = {0: 1, 1: 3, 2: 2, 3: 3}
    q05, q95 = t.quantiles(t.af3_values(coo, degrees))
    w = t.weights(coo, 4, q05, q95).to_dense()
    d = w.sum(1)
    inv = 1.0 / torch.sqrt(d)
    p = w * inv.unsqueeze(1) * inv.unsqueeze(0)
    assert torch.allclose(teacher, 0.5 * (p + p @ p), atol=1e-12)
    assert manifest["n_items"] == 4 and manifest["q05"] == q05


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
