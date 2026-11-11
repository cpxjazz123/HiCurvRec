"""Issue #178's pair-curvature distances: the metric properties, kappa=1 agreement and the switch."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import r48k_geometry as g  # noqa: E402


def _pts(n=8, dim=4, seed=0, scale=0.1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(n, dim, generator=generator, dtype=torch.float64) * 2 - 1) * scale


def test_origin_identity_and_symmetry():
    q = _pts(seed=1)
    kappa = torch.full((1, 1), 0.7, dtype=torch.float64)
    p = torch.zeros_like(q)
    mine = g.distance(g.project_to_ball(p, kappa), g.project_to_ball(q, kappa), kappa)
    root = torch.sqrt(kappa)
    closed = (2.0 / root) * torch.atanh((root * g.project_to_ball(q, kappa).norm(dim=-1, keepdim=True)).clamp(max=1 - 1e-7))
    assert torch.allclose(mine, closed.squeeze(-1), rtol=1e-6, atol=1e-9)
    a, b = _pts(seed=2, n=4), _pts(seed=3, n=4)
    k2 = torch.full((4, 4), 1.0, dtype=torch.float64)
    pa = g.project_to_ball(a, k2).unsqueeze(1).expand(4, 4, 4)
    pb = g.project_to_ball(b, k2).unsqueeze(0).expand(4, 4, 4)
    assert torch.allclose(g.distance(pa, pb, k2.unsqueeze(-1)), g.distance(pb, pa, k2.unsqueeze(-1)), rtol=1e-9)


def test_self_distance_is_zero():
    u = _pts(seed=4)
    kappa = torch.full((1, 1), 2.0, dtype=torch.float64)
    pu = g.project_to_ball(u, kappa)
    # the issue's eps_pq floor makes a coincident pair's distance ~2 sqrt(eps) rather than exactly zero; it stays negligible
    assert float(g.distance(pu, pu, kappa).abs().max()) < 1e-6


def test_projection_stays_strictly_inside_the_ball():
    u = torch.randn(16, 5, dtype=torch.float64) * 3.0
    for k in (0.25, 1.0, 4.5):
        kappa = torch.full((1, 1), k, dtype=torch.float64)
        p = g.project_to_ball(u, kappa)
        assert float((k * p.pow(2).sum(-1)).max()) < 1.0
    zero = torch.zeros(1, 5, dtype=torch.float64)
    assert torch.allclose(g.project_to_ball(zero, torch.full((1, 1), 1.0, dtype=torch.float64)), zero)


def test_kappa_one_agrees_with_the_frozen_contract():
    # this tree has no poincare_contract module, so the frozen reference is loaded from the candidate tree that does
    import importlib.util
    ref_path = TREE.parent / "TIGER_mixed_rq" / "model" / "poincare_contract.py"
    spec = importlib.util.spec_from_file_location("frozen_poincare_contract", ref_path)
    frozen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen_module)
    frozen = frozen_module.distance
    a, b = _pts(seed=5, n=6), _pts(seed=6, n=6)
    kappa = torch.ones(6, 1, dtype=torch.float64)
    mine = g.distance(g.project_to_ball(a, kappa), g.project_to_ball(b, kappa), kappa)
    ref = frozen(g.project_to_ball(a, kappa), g.project_to_ball(b, kappa), 1.0)
    assert torch.allclose(mine, ref, rtol=1e-6, atol=1e-9)


def test_geometry_off_is_the_euclidean_cdist(monkeypatch):
    a, b = _pts(seed=7, n=5), _pts(seed=8, n=5)
    ks, kc = torch.ones(5), torch.ones(5)
    monkeypatch.setattr(g, "GEOMETRY_OFF", True)
    assert torch.equal(g.behaviour_distances(a, b, ks, kc), torch.cdist(a, b, p=2))


def test_pair_kappa_is_the_geometric_mean():
    sk = torch.tensor([0.25, 4.0])
    ck = torch.tensor([1.0, 1.0])
    assert [round(float(x), 6) for x in g.pair_kappa(sk, ck).flatten()] == [0.5, 0.5, 2.0, 2.0]


def test_the_table_loads_through_the_markers_and_lookup_works():
    table = g.load_table()
    # the artifact carries two means: mean_frozen (0.121, the value the issue prescribes for the 31 blind-spot items) and
    # mean_kappa (1.0, the mean of the measured values). Only the former is a fill value.
    assert 0.10 < table["mean_frozen"] < 0.15
    assert abs(table["mean_kappa"] - 1.0) < 1e-9
    ids = torch.tensor([0, 1, 2])
    values = g.kappa_of(ids)
    assert values.shape == (3,) and bool((values >= g.KAPPA_FLOOR).all())
