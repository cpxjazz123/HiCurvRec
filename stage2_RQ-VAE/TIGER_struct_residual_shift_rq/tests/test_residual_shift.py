"""The residual shift of issue #153: domain, limits, alpha rule, non-inertness.

Red-first: these tests name the API the mechanism must have before it exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import mixed_rq as mq  # noqa: E402
from model.mixed_rq import AlphaTable, MixedRQ  # noqa: E402
from model.poincare_contract import expmap0, mobius_add  # noqa: E402


def _payload():
    # a synthetic table sized so the mechanics are observable at the test's z scale;
    # whether the frozen table moves >=10% of the rows is gate 1 on the real probe
    centroid = torch.tensor([0.05, -0.03] + [0.0] * 14, dtype=torch.float64)
    return {
        "alpha_rule": "||c|| / (||c|| + median||r||)",
        "centroids": {"0_1": centroid.tolist(), "1_1": centroid.tolist(),
                      "2_1": centroid.tolist(), "3_1": centroid.tolist(),
                      "0_2": centroid.tolist(), "1_2": centroid.tolist(),
                      "2_2": centroid.tolist(), "3_2": centroid.tolist()},
        "median_residual_norm": {f"{g}_{l}": 0.02 for g in range(4) for l in (1, 2)},
        "alpha": {f"{g}_{l}": float(centroid.norm() / (float(centroid.norm()) + 0.02))
                  for g in range(4) for l in (1, 2)},
    }


def test_alpha_is_a_measured_ratio_inside_the_unit_interval():
    payload = _payload()
    for value in payload["alpha"].values():
        assert 0.0 < value < 1.0
    norm = torch.tensor(payload["centroids"]["0_1"], dtype=torch.float64).norm()
    expected = float(norm) / (float(norm) + payload["median_residual_norm"]["0_1"])
    assert payload["alpha"]["0_1"] == pytest.approx(expected, rel=1e-12)


def test_shift_is_the_identity_when_switched_off(monkeypatch):
    monkeypatch.setattr(mq, "SHIFT_OFF", True)
    monkeypatch.setattr(mq, "GEOMETRY_OFF", False)
    point = torch.tensor([[0.03, -0.02] + [0.0] * 14], dtype=torch.float64)
    groups = torch.zeros(1, dtype=torch.long)
    shifted = mq.apply_residual_shift(point, groups, _payload(), level=1, curvature=1.0)
    assert torch.equal(shifted, point)


def test_euclidean_limit_adds_the_plain_vector(monkeypatch):
    monkeypatch.setattr(mq, "SHIFT_OFF", False)
    monkeypatch.setattr(mq, "GEOMETRY_OFF", True)
    payload = _payload()
    point = torch.tensor([[0.03, -0.02] + [0.0] * 14], dtype=torch.float64)
    groups = torch.zeros(1, dtype=torch.long)
    shifted = mq.apply_residual_shift(point, groups, payload, level=1, curvature=1.0)
    centroid = torch.tensor(payload["centroids"]["0_1"], dtype=torch.float64)
    alpha = payload["alpha"]["0_1"]
    assert torch.allclose(shifted, point - alpha * centroid, atol=1e-12)


def test_hyperbolic_shift_matches_the_mobius_form(monkeypatch):
    monkeypatch.setattr(mq, "SHIFT_OFF", False)
    monkeypatch.setattr(mq, "GEOMETRY_OFF", False)
    payload = _payload()
    point = torch.tensor([[0.03, -0.02] + [0.0] * 14], dtype=torch.float64)
    groups = torch.zeros(1, dtype=torch.long)
    shifted = mq.apply_residual_shift(point, groups, payload, level=1, curvature=1.0)
    centroid = torch.tensor(payload["centroids"]["0_1"], dtype=torch.float64)
    expected = mobius_add(point, -payload["alpha"]["0_1"] * centroid, 1.0)
    assert torch.allclose(shifted, expected, atol=1e-12)
    assert float(shifted.norm()) < 1.0


def test_shift_accepts_every_level_of_the_table():
    payload = _payload()
    point = torch.zeros(4, 16, dtype=torch.float64)
    groups = torch.arange(4)
    for level in (1, 2):
        shifted = mq.apply_residual_shift(point, groups, payload, level=level,
                                          curvature=1.0)
        assert shifted.shape == point.shape and torch.isfinite(shifted).all()


def test_shift_moves_codes_on_a_real_level(monkeypatch):
    torch.manual_seed(0)
    rq = MixedRQ([64, 64, 64], AlphaTable(), curvature=1.0)
    rq.init_codebooks(torch.randn(512, 32), torch.device("cpu"))
    z = torch.randn(256, 32) * 0.2
    rq.train()
    with torch.no_grad():
        rq(z)
    plain = rq.last_codes[-1].clone()
    monkeypatch.setattr(mq, "SHIFT_OFF", False)
    payload = _payload()
    groups = torch.arange(256) % 4
    rq.attach_centroids(payload)
    with torch.no_grad():
        rq(z, groups=groups)
    shifted = rq.last_codes[-1].clone()
    moved = float((plain != shifted).float().mean())
    # the unit test only establishes that the shift reaches the quantizer at all; how
    # many rows it moves on the frozen table is gate 1, measured on the real probe
    assert moved > 0.0, "the shift never reached the quantizer"


def test_alpha_zero_reproduces_the_baseline_exactly(monkeypatch):
    torch.manual_seed(1)
    rq = MixedRQ([64, 64, 64], AlphaTable(), curvature=1.0)
    rq.init_codebooks(torch.randn(512, 32), torch.device("cpu"))
    z = torch.randn(128, 32) * 0.2
    rq.train()
    with torch.no_grad():
        rq(z)
    baseline = rq.last_codes[-1].clone()
    payload = _payload()
    payload["alpha"] = {key: 0.0 for key in payload["alpha"]}
    rq.attach_centroids(payload)
    with torch.no_grad():
        rq(z, groups=torch.arange(128) % 4)
    assert torch.equal(rq.last_codes[-1], baseline)
