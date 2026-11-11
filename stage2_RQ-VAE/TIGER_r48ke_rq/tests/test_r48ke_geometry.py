"""Issue #183: the representation map, its inverse, the distance and the three states."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import r48ke_geometry as g  # noqa: E402


def test_the_map_stays_strictly_inside_the_ball_and_is_monotone():
    z = torch.randn(16, 5, dtype=torch.float64) * 3.0
    for k in (0.25, 1.0, 4.5):
        kappa = torch.full((16, 1), k, dtype=torch.float64)
        p = g.expmap0(z, kappa)
        assert float((k * p.pow(2).sum(-1)).max()) < 1.0
        # monotone: larger input norm gives a larger mapped norm, for a fixed kappa
        norms_z = z.norm(dim=-1).argsort()
        norms_p = p.norm(dim=-1)
        assert bool((norms_p[norms_z][1:] >= norms_p[norms_z][:-1] - 1e-12).all())


def test_zero_maps_to_zero_by_the_limit():
    z = torch.zeros(2, 4, dtype=torch.float64)
    assert torch.equal(g.expmap0(z, torch.ones(2, 1, dtype=torch.float64)), z)


def test_the_map_and_its_inverse_round_trip():
    z = torch.randn(8, 6, dtype=torch.float64) * 0.4
    kappa = torch.full((8, 1), 1.3, dtype=torch.float64)
    back = g.logmap0(g.expmap0(z, kappa), kappa)
    # the map shrinks by (1 - 1e-6) so that kappa|p|^2 stays strictly below one; the inverse therefore returns z to within that
    # factor, which is why the tolerance here is 3e-6 rather than 1e-6
    assert torch.allclose(back, z, rtol=1e-9, atol=1e-10)   # exact inverse, not merely close


def test_geometry_off_leaves_the_representation_untouched():
    z = torch.randn(4, 3, dtype=torch.float64)
    g.GEOMETRY_OFF = True
    same = g.expmap0(z, torch.ones(4, 1, dtype=torch.float64))
    g.GEOMETRY_OFF = False
    assert torch.equal(same, z)


def test_the_distance_is_finite_at_coincident_and_near_coincident_points():
    for gap in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
        p = (torch.rand(1, 4, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 2 - 1) * 0.1
        q = p + gap
        kappa = torch.full((1, 1), 1.0, dtype=torch.float64)
        d = g.distance(p, q, kappa)
        assert bool(torch.isfinite(d).all()), gap


def test_embed_only_returns_the_euclidean_matrix(monkeypatch):
    a = torch.randn(4, 5, dtype=torch.float64) * 0.1
    b = torch.randn(4, 5, dtype=torch.float64) * 0.1
    ks = torch.full((4,), 1.0)
    monkeypatch.setattr(g, "EMBED_ONLY", True)
    assert torch.equal(g.behaviour_distances(a, b, ks, ks), torch.cdist(a, b, p=2))


def test_the_real_table_loads_and_the_pair_kappa_is_the_geometric_mean():
    values = g.kappa_of(torch.tensor([0, 1]))
    assert values.shape == (2,) and bool((values >= 1e-3).all())
    pair = g.pair_kappa(torch.tensor([0.25, 4.0]), torch.tensor([1.0, 1.0]))
    assert [round(float(x), 6) for x in pair.flatten()] == [0.5, 0.5, 2.0, 2.0]
