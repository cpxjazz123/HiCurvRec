"""Known answers, boundary cases and an independent reference for the distance.

The reference is a different mathematical path from production: it uses the
arccosh identity for the ball and the ambient dot-product/arccos formula for the
sphere, never the asinh/asin branch the candidate evaluates.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import stereographic


def reference_squared(x, y, k):
    """Independent arccosh/acos evaluation of the same family."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xy = ((x - y) ** 2).sum(-1)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    if k == 0:
        return 4.0 * xy
    if k < 0:
        a = -k
        argument = 1.0 + 2.0 * a * xy / ((1.0 - a * x2) * (1.0 - a * y2))
        return (np.arccosh(argument) ** 2) / a
    radius2 = 1.0 / k
    dot = (x * y).sum(-1)
    numerator = 4.0 * radius2 * dot + (x2 - radius2) * (y2 - radius2)
    denominator = (radius2 + x2) * (radius2 + y2)
    angle = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
    return radius2 * angle ** 2


def test_zero_curvature_is_exactly_the_euclidean_limit():
    torch.manual_seed(268)
    x = torch.randn(64, 32, dtype=torch.float64)
    y = torch.randn(64, 32, dtype=torch.float64)
    got = stereographic.squared(x, y, 0.0)
    expected = 4.0 * (x - y).square().sum(-1)
    assert torch.equal(got, expected)


def test_reference_agreement_on_both_signs():
    rng = np.random.default_rng(268)
    x = rng.normal(scale=0.2, size=(128, 32))
    y = rng.normal(scale=0.2, size=(128, 32))
    for k in (-0.125, -0.05, 0.0, 0.05, 0.5, 1.0):
        got = stereographic.squared(torch.from_numpy(x), torch.from_numpy(y), k)
        expected = reference_squared(x, y, k)
        assert np.allclose(got.numpy(), expected, rtol=1e-9, atol=1e-9), k


def test_near_zero_continuity():
    rng = np.random.default_rng(7)
    x = torch.from_numpy(rng.normal(scale=0.2, size=(32, 32)))
    y = torch.from_numpy(rng.normal(scale=0.2, size=(32, 32)))
    base = stereographic.squared(x, y, 0.0)
    for k in (-1e-4, 1e-4):
        near = stereographic.squared(x, y, k)
        assert torch.allclose(near, base, rtol=1e-3, atol=1e-9), k


def test_coincident_points_are_zero():
    x = torch.randn(16, 32, dtype=torch.float64)
    for k in (-0.125, 0.0, 0.5):
        assert float(stereographic.squared(x, x, k).abs().max()) < 1e-18


def test_frozen_box_corner_is_inside_the_ball():
    # max||w|| = 2.8118 at the frozen scale; the legal lower curvature is -0.125.
    corner = torch.tensor([[2.8118] + [0.0] * 31], dtype=torch.float64)
    origin = torch.zeros(1, 32, dtype=torch.float64)
    value = stereographic.squared(corner, origin, -0.125)
    assert torch.isfinite(value).all() and float(value) > 0
    with pytest.raises(ValueError):
        stereographic.squared(corner, origin, -0.2)


def test_sphere_antipode_is_finite_and_monotone_in_curvature():
    x = torch.zeros(1, 32, dtype=torch.float64)
    y = torch.zeros(1, 32, dtype=torch.float64)
    y[0, 0] = 3.0
    weak = float(stereographic.squared(x, y, 0.01))
    strong = float(stereographic.squared(x, y, 1.0))
    # In the stereographic chart a fixed chord is compressed as kappa grows:
    # the same coordinate distance subtends a smaller angle on the smaller
    # sphere, so the geodesic distance must decrease with kappa.
    assert math.isfinite(weak) and math.isfinite(strong) and strong < weak


def test_projection_keeps_training_points_inside_the_ball():
    k = -0.125
    radius = (1.0 - 1e-3) / math.sqrt(-k)
    far = torch.zeros(1, 32, dtype=torch.float64)
    far[0, 0] = 4.0
    projected = stereographic.project(far, k)
    assert float(projected.norm()) <= radius + 1e-12
    inside = torch.zeros(1, 32, dtype=torch.float64)
    inside[0, 0] = 0.5
    assert torch.equal(stereographic.project(inside, k), inside)
    # The projected outside point has a finite, positive distance.
    origin = torch.zeros(1, 32, dtype=torch.float64)
    value = stereographic.squared(stereographic.project(far, k), origin, k)
    assert torch.isfinite(value).all() and float(value) > 0


def test_projection_has_finite_gradients_for_non_negative_curvature():
    torch.manual_seed(0)
    for k in (0.0, 0.5, 1.0):
        x = (torch.randn(4, 32, dtype=torch.float64) * 10.0).requires_grad_(True)
        projected = stereographic.project(x, torch.full((4,), k), eps=1e-2)
        projected.sum().backward()
        assert torch.isfinite(x.grad).all(), k
        x.grad = None
