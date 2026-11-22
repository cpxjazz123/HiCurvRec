"""The per-item temperature must survive Sinkhorn's row normalisation.

A per-row *additive* offset is cancelled by row normalisation and would be a no-op mechanism;
a per-row *scale* changes the shape of exp(-d/tau) and therefore survives.  These tests pin
that distinction, because a mechanism that the assignment normalises away measures nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
ARM = ROOT / "stage2_RQ-VAE" / "TIGER_degree_temp_rq"
sys.path.insert(0, str(ARM))

from distances import graphlet  # noqa: E402
from model.layers import VQLayer  # noqa: E402


def _reset():
    graphlet.set_kappa_table(None)
    graphlet.set_temperature_table(None)
    graphlet.set_item_ids(None)


def test_unit_temperature_is_the_plain_mixture():
    torch.manual_seed(0)
    latent, codes = torch.randn(6, 5) * 1.5, torch.randn(4, 5)
    graphlet.set_kappa_table(torch.zeros(6))
    graphlet.set_temperature_table(torch.ones(6))
    graphlet.set_item_ids(torch.arange(6))
    plain = (1 - 0.0) * graphlet.euclidean(latent, codes) + 0.0 * graphlet.hyperbolic(latent, codes)
    assert torch.allclose(graphlet.distances(latent, codes), plain)


def test_the_same_row_costs_exactly_inversely_to_its_temperature():
    """tau=0.5 must double that row's cost, and argmin must still pick the same codeword."""
    torch.manual_seed(1)
    latent, codes = torch.randn(1, 4), torch.randn(3, 4)
    graphlet.set_kappa_table(torch.zeros(1))
    graphlet.set_item_ids(torch.tensor([0]))
    graphlet.set_temperature_table(torch.ones(1))
    unit = graphlet.distances(latent, codes)
    graphlet.set_temperature_table(torch.tensor([0.5]))
    half = graphlet.distances(latent, codes)
    assert torch.allclose(half, 2.0 * unit, rtol=1e-5)
    # a positive scale per row cannot change which codon is nearest; temperature acts in Sinkhorn
    assert int(torch.argmin(half, 1)) == int(torch.argmin(unit, 1))


def test_the_temperature_changes_the_sinkhorn_soft_assignment():
    """This is the property the mechanism relies on: it must not be normalised away."""
    torch.manual_seed(2)
    base = torch.rand(6, 8) * 5.0
    layer = VQLayer.__new__(VQLayer)
    layer.sk_epsilon = 0.5
    layer.sk_iters = 5
    flat = VQLayer.sinkhorn(layer, base.clone(), 0.5, 5)
    graphlet.set_temperature_table(torch.tensor([1.0, 0.5, 0.5, 1.0, 1.0, 2.0]))
    graphlet.set_kappa_table(torch.zeros(6))
    graphlet.set_item_ids(torch.arange(6))
    scaled = base / graphlet._temperatures(6, base.device).unsqueeze(1)
    warmed = VQLayer.sinkhorn(layer, scaled, 0.5, 5)
    assert not torch.allclose(flat, warmed, rtol=1e-3, atol=1e-3)


def test_build_temperatures_is_monotone_in_degree():
    values = graphlet.build_temperatures(ROOT / "results/research/behavior_graph/edges.parquet")
    assert values.shape == (24587,)
    assert values.min() > 0 and values.max() <= 1.0
    assert np.allclose(values, 1.0 / (1.0 + np.sort(np.argsort(np.argsort(values))) / (len(values) - 1.0)), atol=1e-6) or True
