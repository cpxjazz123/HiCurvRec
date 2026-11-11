"""The curvature metric of issue #177: the origin identity, symmetry, positivity and the switches."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import kappa_metric as km  # noqa: E402


def _ball(n=8, dim=4, seed=0, scale=0.1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(n, dim, generator=generator, dtype=torch.float64) * 2 - 1) * scale


def test_origin_identity_matches_the_closed_form():
    v = _ball(seed=1)
    kappa = torch.full((v.shape[0], 1), 0.7, dtype=torch.float64)
    origin = torch.zeros_like(v)
    mine = km.distance(origin, v, kappa)
    theirs = km.at_origin(v, kappa)
    assert torch.allclose(mine, theirs, rtol=1e-6, atol=1e-9)


def test_symmetry_and_self_distance():
    u, v = _ball(seed=2, n=4), _ball(seed=3, n=4)
    kappa = torch.full((4, 1), 1.0, dtype=torch.float64)
    assert torch.allclose(km.distance(u, v, kappa), km.distance(v, u, kappa), rtol=1e-10)
    assert float(km.distance(u, u, kappa).abs().max()) < 1e-9


def test_triangle_inequality_holds_on_sampled_triples():
    a, b, c = _ball(seed=4, n=16), _ball(seed=5, n=16), _ball(seed=6, n=16)
    kappa = torch.full((16, 1), 1.0, dtype=torch.float64)   # k = 1: the frozen reference's own regime
    left = km.distance(a, c, kappa)
    right = km.distance(a, b, kappa) + km.distance(b, c, kappa)
    assert bool((left <= right + 1e-9).all())


def test_metric_agrees_with_the_frozen_distance_at_several_curvatures():
    # the frozen poincare_contract.distance is the known-good reference; agreement at k != 1 is what validates the formula
    from model.poincare_contract import distance as frozen
    u, v = _ball(seed=7, n=4, dim=3), _ball(seed=8, n=4, dim=3)
    for k in (0.5, 1.3, 4.0):
        mine = km.distance(u, v, torch.full((4, 1), k, dtype=torch.float64))
        ref = frozen(u, v, float(k))
        assert torch.allclose(mine, ref, rtol=1e-6, atol=1e-9), k


def test_geometry_off_is_the_euclidean_path(monkeypatch):
    u, v = _ball(seed=9, n=4), _ball(seed=10, n=4)
    monkeypatch.setattr(km, "GEOMETRY_OFF", True)
    assert torch.allclose(km.distance(u, v, 1.0), 2.0 * (u - v).norm(dim=-1), rtol=1e-12)


def test_prefix_kappa_level_one_is_one_and_prefixes_lookup():
    table = {"level_1": {"kappa": 1.0}, "mean_frozen": 1.0,
             "level_2": {"rows": {"(3,)": 0.25}}, "level_3": {"rows": {"(3, 7)": 2.5}}}
    assert km.prefix_kappa(table, 1, None) == 1.0
    assert km.prefix_kappa(table, 2, (3,)) == 0.25
    assert km.prefix_kappa(table, 3, (3, 7)) == 2.5
    # an unseen prefix falls back to the frozen mean rather than to a silent zero
    assert km.prefix_kappa(table, 2, (99,)) == table["mean_frozen"]
