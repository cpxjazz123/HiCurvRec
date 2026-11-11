"""Issue #185: the relative-curvature cost, its two degenerate states, the clip and the refresh."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import cmr_geometry as g  # noqa: E402


def test_the_ratio_is_the_item_over_the_code():
    items = torch.tensor([2.0, 0.5])
    codes = torch.tensor([1.0, 0.25])
    rho = g.rho_matrix(items, codes)
    assert abs(float(rho[0, 0]) - 2.0) < 1e-12
    assert abs(float(rho[1, 1]) - 2.0) < 1e-12
    assert abs(float(rho[0, 1]) - 8.0) < 1e-12


def test_the_clip_holds_the_extremes():
    items = torch.tensor([1e6, 1e-6])
    codes = torch.tensor([1e-6, 1e6])
    rho = g.rho_matrix(items, codes)
    assert float(rho.max()) <= g.CLIP[1] and float(rho.min()) >= g.CLIP[0]


def test_ratio_off_is_exactly_the_euclidean_cost():
    latent = torch.randn(4, 6, dtype=torch.float64) * 0.1
    codes = torch.randn(8, 6, dtype=torch.float64) * 0.1
    items = torch.linspace(0.5, 2.0, 4, dtype=torch.float64)
    code_kappa = torch.linspace(0.7, 1.5, 8, dtype=torch.float64)
    g.RATIO_OFF = True
    got = g.level1_cost(latent, codes, items, code_kappa)
    g.RATIO_OFF = False
    want = latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codes.t() + codes.pow(2).sum(1, keepdim=True).t()
    assert torch.equal(got, want)


def test_numerator_only_repeats_the_item_curvature_across_codes():
    items = torch.tensor([0.5, 2.0])
    codes = torch.tensor([1.0, 4.0, 9.0])
    g.NUMERATOR_ONLY = True
    rho = g.rho_matrix(items, codes)
    g.NUMERATOR_ONLY = False
    assert torch.equal(rho[0], torch.full((3,), 0.5, dtype=torch.float64))
    assert torch.equal(rho[1], torch.full((3,), 2.0, dtype=torch.float64))


def test_denominator_only_uses_the_inverse_code_curvature():
    items = torch.tensor([0.5, 2.0])
    codes = torch.tensor([1.0, 4.0])
    g.DENOMINATOR_ONLY = True
    rho = g.rho_matrix(items, codes)
    g.DENOMINATOR_ONLY = False
    assert torch.equal(rho[0], rho[1])                 # independent of the item
    assert abs(float(rho[0, 1]) - 0.25) < 1e-12


def test_the_refresh_takes_the_mean_and_keeps_empty_codes():
    table = torch.ones(4, dtype=torch.float64)
    new, record = g.refresh_codes(table, torch.tensor([2.0, 4.0]), torch.tensor([0, 0]))
    assert abs(float(new[0]) - 3.0) < 1e-12
    assert abs(float(new[1]) - 1.0) < 1e-12 and record["empty"] == 3


def test_the_refresh_and_the_ratio_carry_no_gradient():
    table = torch.ones(2, dtype=torch.float64)
    items = torch.tensor([2.0], requires_grad=True)
    new, _ = g.refresh_codes(table, items, torch.tensor([0]))
    assert not new.requires_grad and items.grad is None
    rho = g.rho_matrix(items.detach(), table)
    assert not rho.requires_grad


def test_to_ball_and_the_distance_survive_coincident_points():
    u = torch.randn(6, 4, dtype=torch.float64) * 3.0
    for k in (0.4, 1.0, 3.0):
        kappa = torch.full((1, 1), k, dtype=torch.float64)
        p = g.to_ball(u, kappa)
        assert float((k * p.pow(2).sum(-1)).max()) < 1.0
    p = torch.rand(1, 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 0.1
    for gap in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
        d = g.distance(p, p + gap, torch.full((1, 1), 1.0, dtype=torch.float64))
        assert bool(torch.isfinite(d).all()), gap
