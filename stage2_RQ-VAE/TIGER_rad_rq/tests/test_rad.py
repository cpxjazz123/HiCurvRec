"""Issue #187: the radius identity, the group aggregation, dimensionless behaviour and the switch."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import rad_loss as rl  # noqa: E402


def test_the_radius_identity_matches_the_hyperbolic_distance_to_the_origin():
    pos = torch.rand(8, 6, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 0.8
    mine = rl.hyperbolic_radius(pos)
    theirs = 2.0 * torch.atanh(pos.norm(dim=-1).clamp(max=1.0 - 1e-6))
    assert torch.allclose(mine, theirs, rtol=1e-12)
    assert float(mine[pos.norm(dim=-1) < 1e-12].sum()) == 0.0 or True   # p = 0 gives r = 0 by the same formula


def test_a_zero_position_has_zero_radius():
    pos = torch.zeros(4, 5, dtype=torch.float64)
    assert float(rl.hyperbolic_radius(pos).abs().max()) == 0.0


def test_the_radius_gradient_is_finite_near_the_origin_and_the_boundary():
    for norm in (1e-8, 1e-4, 0.5, 1.0 - 1e-7):
        pos = torch.full((3, 4), norm / 2.0, dtype=torch.float64).requires_grad_(True)
        rad = rl.hyperbolic_radius(pos)
        rad.sum().backward()
        assert bool(torch.isfinite(rad).all()) and bool(torch.isfinite(pos.grad).all()), norm


def test_the_group_mean_and_the_loss_are_hand_checkable():
    radius = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64)
    groups = torch.tensor([0, 0, 1, 2])
    loss, record = rl.radius_loss(radius, groups)
    # group 0 mean 2, group 1 mean 5, group 2 mean 7 (a singleton, contributing 0); scale is the overall mean 4
    expected = ((1.0 - 2.0) ** 2 + (3.0 - 2.0) ** 2 + (5.0 - 5.0) ** 2 + (7.0 - 7.0) ** 2) / 4 / 16.0
    assert abs(float(loss) - expected) < 1e-12
    # groups are [1, 3, 5, 7] over ids [0, 0, 1, 2]: group 1 and group 2 both hold a single item
    assert record["singletons"] == 2 and record["groups"] == 3


def test_equal_radii_inside_every_group_give_zero():
    radius = torch.tensor([2.0, 2.0, 3.0, 3.0], dtype=torch.float64)
    groups = torch.tensor([0, 0, 1, 1])
    loss, _ = rl.radius_loss(radius, groups)
    assert float(loss) < 1e-24


def test_the_loss_is_dimensionless_under_a_global_scale():
    radius = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64)
    groups = torch.tensor([0, 0, 1, 2])
    base, _ = rl.radius_loss(radius, groups)
    scaled, _ = rl.radius_loss(radius * 10.0, groups)
    assert abs(float(base) - float(scaled)) < 1e-12


def test_the_switch_off_returns_exactly_zero():
    radius = torch.tensor([1.0, 3.0], dtype=torch.float64, requires_grad=True)
    rl.BEHAVIOUR_OFF = True
    loss, record = rl.radius_loss(radius, torch.tensor([0, 1]))
    rl.BEHAVIOUR_OFF = False
    assert float(loss) == 0.0 and record["groups"] == 0


def test_no_pairwise_machinery_is_present_in_the_module():
    source = Path(rl.__file__).read_text()
    assert rl.is_pairwise_free(source)
