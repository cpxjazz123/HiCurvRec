"""Issue #190: the curvature-conditioned target, its kappa = 1 agreement and the switch."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import hyperbolic_target as ht  # noqa: E402


def test_kappa_one_matches_the_frozen_contract():
    import importlib.util
    ref = TREE / "model" / "poincare_contract.py"
    spec = importlib.util.spec_from_file_location("frozen_contract", ref)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    a = torch.randn(6, 4, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 0.2
    b = torch.randn(6, 4, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.2
    mine = ht.distance(a, b, torch.ones(6, dtype=torch.float64))
    theirs = frozen.distance(a, b, 1.0)
    assert torch.allclose(mine, theirs, rtol=1e-6, atol=1e-9)


def test_the_pair_curvature_is_the_geometric_mean():
    k = ht.pair_kappa(torch.tensor([0.25, 4.0]), torch.tensor([1.0, 1.0]))
    assert [round(float(x), 6) for x in k.tolist()] == [0.5, 2.0]


def test_coincident_and_near_coincident_points_are_finite():
    base = torch.rand(1, 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 0.1
    for gap in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
        d = ht.distance(base, base + gap, torch.ones(1, dtype=torch.float64))
        assert bool(torch.isfinite(d).all()), gap


def test_the_squared_form_is_the_square_of_the_distance():
    a = torch.randn(4, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 0.2
    b = torch.randn(4, 3, generator=torch.Generator().manual_seed(4), dtype=torch.float64) * 0.2
    k = torch.full((4,), 1.3, dtype=torch.float64)
    assert torch.allclose(ht.squared(a, b, k), ht.distance(a, b, k).pow(2), rtol=1e-12)


def test_the_table_lookup_fills_and_floors():
    values = ht.kappa_of(torch.tensor([0, 1, 2]))
    assert values.shape == (3,) and bool((values >= 1e-3).all())


def test_the_switch_default_is_geometry_on():
    assert ht.GEOMETRY_OFF is False
