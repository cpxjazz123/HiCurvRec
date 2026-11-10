"""The prefix-order supervision of issue #154: monotonicity, domains, gradients, limit.

Red-first: these name the module the mechanism must provide before it exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import prefix_geometry as pg  # noqa: E402


def test_af3_rule_on_a_hand_graph():
    # path 0-1-2 with one extra edge 0-2 closing the triangle: deg = [2, 2, 2], m = 1
    edges = torch.tensor([[0, 1], [1, 2], [0, 2]])
    triangles = torch.tensor([[0, 1, 2]])
    af3 = pg.af3_per_edge(edges, triangles, nodes=3)
    # 4 - 2 - 2 + 3*1 = 3 for every edge of the closed triangle
    assert torch.allclose(af3, torch.full((3,), 3.0, dtype=torch.float64))


def test_attribution_is_monotone_in_distance():
    far = torch.tensor([2.0])
    near = torch.tensor([0.1])
    s_far = pg.attribution(far, d_bar=torch.tensor([1.0]), tau=torch.tensor([0.5]))
    s_near = pg.attribution(near, d_bar=torch.tensor([1.0]), tau=torch.tensor([0.5]))
    assert float(s_near) > float(s_far)
    assert 0.0 < float(s_far) < 1.0 and 0.0 < float(s_near) < 1.0


def test_tau_is_the_measured_mad():
    distances = torch.tensor([0.1, 0.2, 0.4, 0.8])
    d_bar, tau = pg.centre_and_spread(distances)
    median = float(distances.median())
    mad = float((distances - median).abs().median())
    assert float(d_bar) == pytest.approx(median, rel=1e-12)
    assert float(tau) == pytest.approx(mad, rel=1e-12)
    assert float(tau) > 0.0


def test_gap_loss_is_the_hinge_on_the_measured_delta():
    high = torch.tensor([0.9, 0.8])
    low = torch.tensor([0.1, 0.2])
    # softplus, not a hinge: it decays but never reaches zero
    loss = pg.gap_loss(high, low, delta=torch.tensor(0.5))
    assert float(loss) == pytest.approx(float(torch.nn.functional.softplus(torch.tensor(-0.2))),
                                        rel=1e-6)
    tighter = pg.gap_loss(high, low, delta=torch.tensor(5.0))
    assert float(tighter) > float(loss)


def test_triangle_loss_lives_in_the_unit_interval():
    s = torch.tensor([[0.9, 0.9, 0.9], [0.2, 0.5, 0.7]])
    loss = pg.triangle_loss(s)
    assert 0.0 <= float(loss) <= 1.0
    # per triangle 1 - min: 1 - 0.9 = 0.1 and 1 - 0.2 = 0.8, mean 0.45
    assert float(loss) == pytest.approx(0.45, rel=1e-6)


def test_losses_carry_gradient_to_the_prefix_embeddings():
    prefix_e = torch.zeros(4, 4, requires_grad=True)
    prefix_h = torch.zeros(4, 4, requires_grad=True)
    with torch.no_grad():
        prefix_e.add_(torch.randn(4, 4) * 0.1)
        prefix_h.add_(torch.randn(4, 4) * 0.1)
    prefix_e.requires_grad_(True)
    prefix_h.requires_grad_(True)
    edges_h = torch.tensor([[0, 1], [2, 3]])
    edges_b = torch.tensor([[0, 2], [1, 3]])
    triples = torch.tensor([[0, 1, 2]])
    out = pg.structure_losses(prefix_e, prefix_h, edges_h, edges_b, triples,
                              delta=torch.tensor(0.0), curvature=1.0)
    total = out["gap"] + out["tri"]
    total.backward()
    assert prefix_h.grad is not None and float(prefix_h.grad.abs().sum()) > 0.0


def test_structure_off_zeroes_both_terms(monkeypatch):
    monkeypatch.setattr(pg, "STRUCTURE_OFF", True)
    prefix = torch.randn(4, 4, requires_grad=True)
    out = pg.structure_losses(prefix, prefix, torch.tensor([[0, 1]]),
                              torch.tensor([[2, 3]]), torch.tensor([[0, 1, 2]]),
                              delta=torch.tensor(1.0), curvature=1.0)
    assert float(out["gap"]) == 0.0 and float(out["tri"]) == 0.0


def test_euclidean_limit_differs_from_the_curved_value():
    torch.manual_seed(0)
    prefix_e = torch.randn(4, 4) * 0.3
    prefix_h = torch.randn(4, 4) * 0.1
    edges_h = torch.tensor([[0, 1], [2, 3]])
    edges_b = torch.tensor([[0, 2], [1, 3]])
    triples = torch.tensor([[0, 1, 2]])
    curved = pg.structure_losses(prefix_e, prefix_h, edges_h, edges_b, triples,
                                 delta=torch.tensor(0.0), curvature=1.0)
    flat = pg.structure_losses(prefix_e, prefix_h, edges_h, edges_b, triples,
                               delta=torch.tensor(0.0), curvature=0.0)
    assert abs(float(curved["gap"]) - float(flat["gap"])) > 1e-6
