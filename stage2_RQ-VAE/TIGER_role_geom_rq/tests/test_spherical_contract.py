"""The spherical branch's numerical contract.

    python3 -m pytest stage2_RQ-VAE/TIGER_role_geom_rq/tests/test_spherical_contract.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.spherical_contract import (SPHERE_POLE, sphere_distance, sphere_expmap0,
                                      sphere_logmap0, sphere_pole_count, sphere_report,
                                      sphere_squared_distance)


def test_expmap_lands_on_the_unit_sphere() -> None:
    v = torch.randn(64, 16, dtype=torch.float64) * 0.3
    p = sphere_expmap0(v)
    assert p.shape == (64, 17)
    assert torch.isfinite(p).all()
    assert torch.allclose(p.norm(dim=-1), torch.ones(64, dtype=torch.float64), atol=1e-12)


def test_logmap_inverts_expmap_away_from_the_pole() -> None:
    v = torch.randn(64, 16, dtype=torch.float64) * 0.3
    assert torch.allclose(sphere_logmap0(sphere_expmap0(v)), v, atol=1e-9)


def test_the_geodesic_distance_from_the_origin_is_the_tangent_norm() -> None:
    v = torch.randn(32, 16, dtype=torch.float64) * 0.3
    origin = torch.zeros(1, 17, dtype=torch.float64)
    origin[..., -1] = 1.0
    distance = sphere_distance(sphere_expmap0(v), origin.expand(32, 17))
    assert torch.allclose(distance, v.norm(dim=-1), atol=1e-9)


def test_self_distance_is_zero_and_the_distance_is_symmetric() -> None:
    p = sphere_expmap0(torch.randn(32, 16, dtype=torch.float64) * 0.4)
    q = sphere_expmap0(torch.randn(32, 16, dtype=torch.float64) * 0.4)
    self_distance = sphere_distance(p, p)
    assert float(self_distance.abs().max()) < 1e-6
    assert torch.allclose(sphere_distance(p, q), sphere_distance(q, p), atol=1e-12)
    assert torch.allclose(sphere_squared_distance(p, q), sphere_distance(p, q).pow(2), atol=1e-12)


def test_antipodal_points_are_pi_apart_and_are_counted() -> None:
    p = torch.zeros(1, 17, dtype=torch.float64)
    p[0, 0] = 1.0
    q = -p
    assert abs(float(sphere_distance(p, q)) - math.pi) < 1e-9
    report = sphere_report(p, q)
    assert report["antipodal_pairs"] == 1


def test_the_map_is_not_injective_past_the_pole_and_that_is_reported() -> None:
    v = torch.zeros(1, 16, dtype=torch.float64)
    v[0, 0] = SPHERE_POLE + 0.2
    past = sphere_expmap0(v)
    wrapped = sphere_expmap0(v - torch.tensor([[2.0 * math.pi] + [0.0] * 15], dtype=torch.float64))
    assert torch.allclose(past, wrapped, atol=1e-9)          # the same point, two tangent vectors
    assert sphere_pole_count(v) == 1


def test_no_nan_for_extreme_inputs() -> None:
    v = torch.tensor([[1e3] + [0.0] * 15, [0.0] * 16], dtype=torch.float64)
    p = sphere_expmap0(v)
    assert torch.isfinite(p).all()
    assert torch.isfinite(sphere_distance(p, p)).all()
    assert torch.isfinite(sphere_logmap0(p)).all()
