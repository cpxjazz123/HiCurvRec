"""The batch restriction of the AF3 supervision (written with the module, run immediately)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import item_index as ii  # noqa: E402
from model.structure_runtime import StructureSupervision  # noqa: E402

EMB = np.ascontiguousarray(np.arange(32, dtype=np.float32).reshape(8, 4))


def _runtime() -> StructureSupervision:
    edges = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    u = np.asarray([1.0, 0.5, 0.0], dtype=np.float64)
    return StructureSupervision(edges, u, np.zeros((0, 3), dtype=np.int64), 0.2,
                                ii.build(EMB), {index: index for index in range(8)})


def test_only_edges_inside_the_batch_are_kept():
    runtime = _runtime()
    rows = torch.from_numpy(EMB[[0, 1, 2]].copy())
    pairs, targets, triples = runtime.batch_edges(rows)
    # items 0, 1, 2 are inside: edge (0,1) yes, (1,2) yes, (2,3) no
    assert pairs.tolist() == [[0, 1], [1, 2]]
    assert targets.tolist() == [1.0, 0.5]
    assert triples.shape == (0, 3)


def test_a_batch_without_edges_returns_none():
    runtime = _runtime()
    rows = torch.from_numpy(EMB[[5]].copy())

    class _Model:
        class rq:  # noqa: N801
            last_prefixes = []

    assert runtime.terms(_Model(), rows) is None
