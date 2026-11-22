"""The coupled behaviour-curvature metric, which is what CLAUDE.md 19h asks for.

Two things have to hold before any epoch is spent: the two ends of the mixture must be exactly
the metrics they claim (flat when the behaviour weight is zero, curvature -1 when it is one), and
the behaviour signal has to actually change the assignment -- otherwise 19h's "removing either
side changes the result" is not satisfied and the arm measures nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
ARM = ROOT / "stage2_RQ-VAE" / "TIGER_af3_graphgeo_rq"
sys.path.insert(0, str(ARM))

from distances import graphlet  # noqa: E402


@pytest.fixture(autouse=True)
def reset_state():
    graphlet.set_kappa_table(None)
    graphlet.set_item_ids(None)
    yield
    graphlet.set_kappa_table(None)
    graphlet.set_item_ids(None)


def _points(n, dim=6, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, dim) * 1.5


def test_zero_behaviour_weight_is_exactly_euclidean():
    latent, codes = _points(9), _points(5, seed=1)
    graphlet.set_kappa_table(torch.zeros(9))
    graphlet.set_item_ids(torch.arange(9))
    assert torch.allclose(graphlet.distances(latent, codes),
                          graphlet.euclidean(latent, codes))


def test_full_behaviour_weight_is_exactly_the_hyperbolic_cost():
    latent, codes = _points(7), _points(4, seed=2)
    graphlet.set_kappa_table(torch.ones(7))
    graphlet.set_item_ids(torch.arange(7))
    assert torch.allclose(graphlet.distances(latent, codes),
                          graphlet.hyperbolic(latent, codes))


def test_the_behaviour_signal_changes_the_cost():
    """19h: dropping the signal must change the result, or the coupling is decorative."""
    latent, codes = _points(8), _points(4, seed=3)
    weights = torch.linspace(0.0, 1.0, 8)
    graphlet.set_kappa_table(weights)
    graphlet.set_item_ids(torch.arange(8))
    coupled = graphlet.distances(latent, codes)
    graphlet.set_kappa_table(torch.zeros(8))
    flat = graphlet.distances(latent, codes)
    assert not torch.allclose(coupled, flat, rtol=1e-3, atol=1e-3)
    # the first item has weight 0 and must be unchanged
    assert torch.allclose(coupled[0], flat[0], rtol=1e-5, atol=1e-5)


def test_the_same_latent_gets_different_geometry_under_a_different_item_id():
    """The id decides the geometry, so one vector must cost differently as item 0 and item 1."""
    graphlet.set_kappa_table(torch.tensor([0.0, 1.0]))
    latent, codes = _points(1), _points(3, seed=4)
    graphlet.set_item_ids(torch.tensor([0]))
    flat = graphlet.distances(latent, codes)
    graphlet.set_item_ids(torch.tensor([1]))
    hyperbolic = graphlet.distances(latent, codes)
    assert not torch.allclose(flat, hyperbolic, rtol=1e-3, atol=1e-3)
    assert torch.allclose(flat, graphlet.euclidean(latent, codes))
    assert torch.allclose(hyperbolic, graphlet.hyperbolic(latent, codes))


def test_a_batch_permuted_with_its_ids_gives_identical_rows():
    """Shuffling rows and ids together must not change any row's cost."""
    graphlet.set_kappa_table(torch.tensor([0.0, 1.0, 0.25, 0.75]))
    latent, codes = _points(4), _points(3, seed=4)
    graphlet.set_item_ids(torch.arange(4))
    straight = graphlet.distances(latent, codes)
    order = torch.tensor([2, 0, 3, 1])
    graphlet.set_item_ids(order)
    shuffled = graphlet.distances(latent[order], codes)
    assert torch.allclose(shuffled, straight[order], rtol=1e-5, atol=1e-5)


def test_a_mismatched_id_batch_falls_back_to_flat_instead_of_indexing_wrong():
    latent, codes = _points(5), _points(3, seed=5)
    graphlet.set_kappa_table(torch.ones(50))
    graphlet.set_item_ids(torch.tensor([0, 1]))          # wrong length
    cost = graphlet.distances(latent, codes)
    assert torch.allclose(cost, graphlet.euclidean(latent, codes))


def test_cost_is_finite_and_gradients_flow():
    latent = _points(6).requires_grad_(True)
    codes = _points(3, seed=6).requires_grad_(True)
    graphlet.set_kappa_table(torch.linspace(0.1, 0.9, 6))
    graphlet.set_item_ids(torch.arange(6))
    cost = graphlet.distances(latent, codes)
    assert torch.isfinite(cost).all() and torch.all(cost >= 0)
    cost.sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
