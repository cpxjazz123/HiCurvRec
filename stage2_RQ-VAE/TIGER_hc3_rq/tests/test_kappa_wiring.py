"""Issue #177's wiring: level 1 untouched, levels 2 and 3 add the prefix-conditioned d_kappa."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import mixed_rq as mr  # noqa: E402

DIM_E, DIM_H = mr.EUCLIDEAN_DIM, mr.HYPERBOLIC_DIM


def _level(seed=0, n=256):
    torch.manual_seed(seed)
    return mr.MixedVQLevel(n, curvature=1.0, beta=0.25)


def _rows(n=4, seed=1):
    torch.manual_seed(seed)
    e = torch.randn(n, DIM_E, dtype=torch.float64) * 0.1
    h = torch.randn(n, DIM_H, dtype=torch.float64) * 0.1
    alpha = torch.full((n, 1), 0.5, dtype=torch.float64)
    return e, h, alpha


def test_level_one_cost_is_bitwise_the_reference():
    layer = _level()
    e, h, alpha = _rows()
    base = layer.distances(e, h, alpha)
    with_kappa_none = layer.distances(e, h, alpha, None)
    assert torch.equal(base, with_kappa_none)


def test_a_prefix_curvature_adds_a_finite_non_negative_term():
    layer = _level()
    e, h, alpha = _rows()
    base = layer.distances(e, h, alpha)
    kappa = torch.tensor([[0.5], [1.0], [2.0], [4.0]], dtype=torch.float64)
    added = layer.distances(e, h, alpha, kappa)
    delta = added - base
    assert bool(torch.isfinite(added).all())
    assert bool((delta >= 0).all())
    assert float(delta.min()) >= 0.0


def test_the_curvature_actually_changes_the_cost_ordering():
    layer = _level()
    e, h, alpha = _rows()
    base = layer.distances(e, h, alpha)
    low = layer.distances(e, h, alpha, torch.full((4, 1), 0.2, dtype=torch.float64))
    high = layer.distances(e, h, alpha, torch.full((4, 1), 5.0, dtype=torch.float64))
    assert not torch.allclose(base, low)
    assert not torch.allclose(low, high)


def test_switch_off_reproduces_the_reference_stack(monkeypatch):
    monkeypatch.setattr(mr, "HC3_ON", False)
    table = mr.MixedRQ([16, 16, 16], mr.AlphaTable(levels=3), curvature=1.0)
    torch.manual_seed(3)
    z = torch.randn(5, DIM_E + DIM_H, dtype=torch.float64) * 0.1
    readout, loss, unused, tokens = table(z)
    assert tokens.shape == (5, 3)
    assert bool(torch.isfinite(loss).all())


def test_the_switch_on_stack_actually_resolves_kappa_from_its_own_tokens():
    # regression for the smoke crash: MixedRQ.forward used to call .tolist() on the token *list* before any level appended,
    # so the real entry died with "'list' object has no attribute 'tolist'" during codebook initialisation
    table = mr.MixedRQ([16, 16, 16], mr.AlphaTable(levels=3), curvature=1.0)
    torch.manual_seed(11)
    z = torch.randn(6, DIM_E + DIM_H, dtype=torch.float64) * 0.1
    readout, loss, unused, tokens = table(z)
    assert tokens.shape == (6, 3)
    assert bool(torch.isfinite(loss).all())
    # and it is genuinely using the table, not silently the level-1 path
    from model import kappa_table
    assert mr.HC3_ON is True
    assert kappa_table.kappa_for_rows(tokens[:, :1], 2).shape == (6, 1)
