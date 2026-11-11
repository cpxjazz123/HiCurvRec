"""Issue #184 acceptance 1, 3, 4: the per-code refresh, the floor, the empty-code rule and the frozen identity."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import r48k_geometry_learned as lk  # noqa: E402


def test_each_code_takes_the_mean_of_the_curvatures_it_received():
    table = lk.initial(4)
    item_kappa = torch.tensor([2.0, 4.0, 0.5])
    codes = torch.tensor([0, 0, 2])
    new, record = lk.update(table, item_kappa, codes)
    assert abs(float(new[0]) - 3.0) < 1e-12          # mean of 2 and 4
    assert abs(float(new[2]) - 0.5) < 1e-12
    assert record["moved"] == 2 and record["empty"] == 2


def test_empty_codes_keep_their_previous_value():
    table = torch.tensor([1.0, 5.0, 0.7])
    new, record = lk.update(table, torch.tensor([2.0]), torch.tensor([0]))
    assert abs(float(new[1]) - 5.0) < 1e-12 and abs(float(new[2]) - 0.7) < 1e-6   # float32 table entries keep ~1e-8
    assert record["empty"] == 2


def test_a_below_floor_value_is_replaced_and_counted():
    table = lk.initial(2)
    new, record = lk.update(table, torch.tensor([1e-9]), torch.tensor([0]))
    assert float(new[0]) == 1.0 and record["floored"] == 1


def test_the_frozen_state_leaves_the_table_untouched():
    table = torch.tensor([1.0, 1.0])
    lk.KAPPA_FROZEN = True
    new, record = lk.update(table, torch.tensor([9.0]), torch.tensor([0]))
    lk.KAPPA_FROZEN = False
    assert torch.equal(new, table) and record["frozen"] is True


def test_the_update_carries_no_gradient():
    table = lk.initial(2)
    item_kappa = torch.tensor([2.0], requires_grad=True)
    new, _ = lk.update(table, item_kappa, torch.tensor([0]))
    assert not new.requires_grad
    assert item_kappa.grad is None


def test_the_cost_uses_the_codes_curvature_and_the_frozen_state_is_euclidean():
    torch.manual_seed(0)
    latent = torch.randn(4, 6, dtype=torch.float64) * 0.1
    codes = torch.randn(8, 6, dtype=torch.float64) * 0.1
    kappa = torch.linspace(0.5, 2.0, 8, dtype=torch.float64)
    curved = lk.level1_cost(latent, codes, kappa)
    euclid = latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codes.t() + codes.pow(2).sum(1, keepdim=True).t()
    assert not torch.allclose(curved, euclid)
    lk.KAPPA_FROZEN = True
    frozen = lk.level1_cost(latent, codes, kappa)
    lk.KAPPA_FROZEN = False
    assert torch.equal(frozen, euclid)


def test_to_ball_stays_strictly_inside_and_the_distance_is_finite_at_coincidence():
    u = torch.randn(8, 5, dtype=torch.float64) * 3.0
    for k in (0.4, 1.0, 3.0):
        kappa = torch.full((1, 1), k, dtype=torch.float64)
        p = lk.to_ball(u, kappa)
        assert float((k * p.pow(2).sum(-1)).max()) < 1.0
    p = torch.rand(1, 4, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.1
    for gap in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
        d = lk.distance(p, p + gap, torch.full((1, 1), 1.0, dtype=torch.float64))
        assert bool(torch.isfinite(d).all()), gap
