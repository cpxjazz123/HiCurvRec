"""The scale-invariant AF3 rank objective of issue #157 (red first)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import rank_objective as ro  # noqa: E402


def _rho(num=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(num, dtype=torch.float64, generator=generator) + 0.1


def test_normalisation_is_scale_invariant():
    rho = _rho()
    high = torch.tensor([0, 1, 2, 3])
    low = torch.tensor([4, 5, 6, 7])
    base = ro.rank_loss(ro.normalise(_rho()), ro.normalise(_rho()), high, low) if False else None
    distances = _rho()
    for factor in (0.25, 4.0):
        scaled = ro.normalise(distances * factor)
        assert torch.allclose(scaled, ro.normalise(distances), atol=1e-12)


def test_rank_loss_drops_when_high_edges_are_closer():
    rho_good = torch.tensor([0.2, 0.2, 0.2, 0.2, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    high = torch.tensor([0, 1, 2, 3])
    low = torch.tensor([4, 5, 6, 7])
    aligned = ro.rank_loss(rho_good, high, low)
    wrong = ro.rank_loss(rho_good.flip(0), high, low)
    assert 0.0 <= float(aligned) <= 1.0
    assert float(aligned) < float(wrong)


def test_spread_floor_fires_only_below_the_measured_floor():
    collapsed = torch.full((8,), 1.0, dtype=torch.float64)
    sigma_min = torch.tensor(0.1, dtype=torch.float64)
    assert float(ro.spread_loss(collapsed, sigma_min)) > 0.0
    wide = torch.linspace(0.0, 5.0, 8, dtype=torch.float64)
    assert float(ro.spread_loss(wide, sigma_min)) == 0.0


def test_losses_carry_gradient_to_the_distance_terms():
    rho = torch.rand(8, dtype=torch.float64, requires_grad=True)
    high = torch.tensor([0, 1])
    low = torch.tensor([6, 7])
    loss = ro.rank_loss(rho, high, low)
    loss.backward()
    assert rho.grad is not None and float(rho.grad.abs().sum()) > 0.0


def test_structure_off_zeroes_every_term(monkeypatch):
    monkeypatch.setattr(ro, "STRUCTURE_OFF", True)
    rho = torch.rand(8, dtype=torch.float64)
    out = ro.structure_losses([rho], torch.tensor([0, 1]), torch.tensor([6, 7]),
                              torch.zeros(0, 3, dtype=torch.long), torch.tensor(0.1))
    assert float(out["rank"]) == 0.0 and float(out["spread"]) == 0.0 and float(out["tri"]) == 0.0


def test_triangle_term_uses_rho_units():
    rho = torch.tensor([[0.1, 0.1, 0.5]], dtype=torch.float64)
    assert float(ro.triangle_term(rho)) == pytest.approx(0.5, rel=1e-9)
