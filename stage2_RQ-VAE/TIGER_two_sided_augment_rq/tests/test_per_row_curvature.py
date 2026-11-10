"""Per-item curvature must reproduce the frozen scalar geometry (issue #167)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import per_row_curvature as prc  # noqa: E402
from model.poincare_contract import distance as frozen_distance  # noqa: E402


def test_constant_curvature_matches_the_frozen_distance():
    torch.manual_seed(0)
    x = torch.randn(16, 8) * 0.1
    y = torch.randn(16, 8) * 0.1
    c = torch.full((16, 1), 1.0)
    mine = prc.distance_rows(x, y, c).squeeze(-1)
    theirs = frozen_distance(x, y, 1.0)
    assert torch.allclose(mine, theirs, rtol=1e-9, atol=1e-10)


def test_zero_curvature_is_the_euclidean_limit():
    x = torch.randn(4, 5) * 0.2
    y = torch.randn(4, 5) * 0.2
    c = torch.zeros(4, 1)
    mine = prc.distance_rows(x, y, c).squeeze(-1)
    assert torch.allclose(mine, 2.0 * (x - y).norm(dim=-1), rtol=1e-12, atol=1e-12)


def test_curvature_rows_broadcast_independently():
    x = torch.randn(4, 5) * 0.1
    y = torch.randn(4, 5) * 0.1
    low = prc.distance_rows(x, y, torch.full((4, 1), 0.25)).squeeze(-1)
    high = prc.distance_rows(x, y, torch.full((4, 1), 4.0)).squeeze(-1)
    assert bool((low != high).all())


def test_curvature_column_shape_is_enforced():
    x = torch.zeros(3, 2)
    with torch.no_grad():
        column = prc.as_column(torch.tensor([0.5, 1.0, 2.0]), x)
    assert column.shape == (3, 1)
