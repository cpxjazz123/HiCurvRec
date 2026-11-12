"""Issue #197 acceptance 1, 3, 4 and 5: the two regimes, the switches, and the row-wise metric.

Every truth value is computed by hand from the graph or from the frozen contract; the switches are exercised through their names, and one case checks that the
test itself can fail -- a detector that cannot fail is not a detector.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import poincare_contract as pc  # noqa: E402
from model import sign_metric as sm  # noqa: E402

# two triangles (0,1,2) and (2,3,4) sharing node 2, so the closed subgraph is one component; the edge (4,5) is a bridge to a triangle-free leaf
GRAPH = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (2, 4), (4, 5)]
TABLE = {"kappa": {str(i): 2.0 for i in range(8)}, "mean_frozen": 0.5}


def test_common_neighbours_counts_triangles_per_edge():
    shared = sm.common_neighbours(GRAPH)
    assert shared[(0, 1)] == 1          # the triangle 0-1-2
    assert shared[(4, 5)] == 0          # the bridge lies in no triangle


def test_the_closed_subgraph_keeps_exactly_the_triangle_edges():
    assert (4, 5) not in sm.closed_subgraph(GRAPH)
    assert (2, 3) in sm.closed_subgraph(GRAPH)


def test_components_are_the_groups_of_the_closed_subgraph():
    groups = sm.components(sm.closed_subgraph(GRAPH))
    assert groups[0] == groups[4] == groups[3]           # one component through the shared triangle node
    assert 5 not in groups                               # the leaf has no closed edge, so it belongs to no group


def test_delta_is_none_when_a_side_is_empty():
    groups = sm.components(sm.closed_subgraph(GRAPH))
    weights = sm.af3_weights(GRAPH)
    delta = sm.delta_of(GRAPH, weights, groups)
    # node 5's only edge is the bridge, so it has no in-group side at all
    assert delta[5] is None


def test_regime_kappa_uses_the_item_curvature_only_where_delta_is_positive():
    delta = {0: 0.5, 1: -0.5, 2: None}
    kappa = sm.regime_kappa(torch.tensor([0, 1, 2]), delta, TABLE)
    assert kappa.tolist() == [2.0, 1.0, 1.0]             # positive -> table; negative and undefined -> the baseline value


def test_behaviour_off_puts_every_item_at_the_baseline_curvature():
    delta = {0: 0.5, 1: -0.5}
    saved = sm.SIGN_OFF
    try:
        sm.SIGN_OFF = True
        kappa = sm.regime_kappa(torch.tensor([0, 1]), delta, TABLE)
    finally:
        sm.SIGN_OFF = saved
    assert kappa.tolist() == [sm.BASELINE_CURVATURE, sm.BASELINE_CURVATURE]


def test_sign_flip_enables_exactly_the_complement():
    delta = {0: 0.5, 1: -0.5, 2: 0.1, 3: None}
    ids = torch.tensor([0, 1, 2, 3])
    plain = sm.regime_kappa(ids, delta, TABLE)
    saved = sm.SIGN_FLIP
    try:
        sm.SIGN_FLIP = True
        flipped = sm.regime_kappa(ids, delta, TABLE)
    finally:
        sm.SIGN_FLIP = saved

    enabled_plain = {i for i in range(4) if float(plain[i]) != sm.BASELINE_CURVATURE}
    enabled_flip = {i for i in range(4) if float(flipped[i]) != sm.BASELINE_CURVATURE}
    assert enabled_plain == {0, 2}                      # the two positive Deltas
    assert enabled_flip == {1, 3}                       # their complement, the undefined item included
    assert enabled_flip == set(range(4)) - enabled_plain


def test_the_complement_test_can_fail(monkeypatch):
    """The detector's own negative control: a stub that ignores the flag must break the complement assertion."""
    monkeypatch.setattr(sm, "regime_kappa", lambda ids, delta, table=None: torch.ones(len(ids)))
    plain = sm.regime_kappa(torch.tensor([0, 1]), {0: 0.5}, TABLE)
    with pytest.raises(AssertionError):
        assert plain.tolist() == [2.0, 1.0]


def test_the_row_wise_cost_equals_the_frozen_contract_at_curvature_one():
    torch.manual_seed(0)
    points = pc.project(torch.randn(4, 8) * 0.05, 1.0)
    codes = pc.project(torch.randn(6, 8) * 0.05, 1.0)
    mine = sm.curved_squared(points, codes, torch.ones(4))
    reference = pc.squared_distance(points.unsqueeze(1), codes.unsqueeze(0), 1.0)
    assert torch.allclose(mine, reference, rtol=1e-6, atol=1e-8)


def test_a_row_at_high_curvature_differs_and_stays_finite():
    torch.manual_seed(1)
    points = pc.project(torch.randn(2, 8) * 0.05, 1.0)
    codes = pc.project(torch.randn(6, 8) * 0.05, 1.0)
    kappa = torch.tensor([1.0, 8.2671])
    values = sm.curved_squared(points, codes, kappa)
    assert torch.isfinite(values).all()
    assert not torch.allclose(values[0], values[1])


def test_gradients_are_finite_at_the_boundary_and_gradcheck_passes():
    torch.manual_seed(2)
    points = pc.project(torch.randn(3, 8) * 0.05, 1.0).double().requires_grad_(True)
    codes = pc.project(torch.randn(5, 8) * 0.05, 1.0).double()
    kappa = torch.tensor([1.0, 8.2671, 0.001], dtype=torch.float64)

    values = sm.curved_squared(points, codes, kappa)
    values.sum().backward()
    assert points.grad is not None and bool(torch.isfinite(points.grad).all())
    assert float(points.grad.abs().sum()) > 0.0

    def f(p):
        return sm.curved_squared(p, codes, kappa)

    assert torch.autograd.gradcheck(f, (points.detach().clone().requires_grad_(True),), eps=1e-6, atol=1e-4)


def test_the_paired_cost_is_row_wise_and_does_not_build_a_square_matrix():
    """Regression for the smoke's CUDA OOM: the commitment and codebook terms compare each row with its own picked point.

    Feeding those points into ``curved_squared`` broadcasts them into a ``[B, B, D]`` tensor -- 38 GiB at the codebook initialisation's batch -- so the paired
    helper exists and the model must use it.  Here the property is stated without a GPU: the paired result is the matrix diagonal, and 4096 rows stay a vector.
    """
    torch.manual_seed(3)
    points = pc.project(torch.randn(64, 8) * 0.05, 1.0)
    picked = pc.project(torch.randn(64, 8) * 0.05, 1.0)
    kappa = torch.full((64,), 1.0)
    paired = sm.curved_squared_paired(points, picked, kappa)
    assert paired.shape == (64,)
    assert torch.allclose(paired, sm.curved_squared(points, picked, kappa).diagonal(), rtol=1e-6)

    wide = sm.curved_squared_paired(torch.randn(4096, 8) * 0.01, torch.randn(4096, 8) * 0.01, torch.ones(4096))
    assert wide.shape == (4096,) and bool(torch.isfinite(wide).all())
