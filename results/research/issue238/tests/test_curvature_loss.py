"""C11/C13/C22 — the curvature fidelity loss and its two switches.

The loss is the entire mechanism, so its degenerate cases matter: identical values must give
zero, the frozen scale must actually enter, and switching the behaviour side off must change the
objective (otherwise the real graph target is decorative).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _armloader import load_arm  # noqa: E402

load_arm("TIGER_curvfid_rq")
from losses import curvature_fidelity as cf  # noqa: E402


def test_a_perfect_prediction_gives_zero():
    k_g = torch.tensor([0.0, 1.0, -2.0, 3.5])
    value = cf.curvature_fidelity_loss(k_g / 2.0, k_g, scale=2.0, delta=1.0)
    assert float(value) == pytest.approx(0.0, abs=1e-8)


def test_the_frozen_scale_enters_the_residual():
    """If the scale were ignored the prediction would be compared in the wrong units."""
    k_g = torch.tensor([1.0, 2.0])
    k_q = torch.tensor([0.5, 1.0])
    scaled = cf.curvature_fidelity_loss(k_q, k_g, scale=2.0, delta=10.0)
    unscaled = cf.curvature_fidelity_loss(k_q, k_g, scale=1.0, delta=10.0)
    assert float(scaled) == pytest.approx(0.0, abs=1e-10)     # 2*0.5 == 1 and 2*1.0 == 2
    assert float(unscaled) > 0.0


def test_switching_the_behaviour_side_off_changes_the_objective():
    k_g = torch.tensor([1.0, -1.0, 2.0])
    k_q = torch.tensor([0.1, 0.2, 0.3])
    on = cf.curvature_fidelity_loss(k_q, k_g, scale=1.0, delta=1.0, access_behavior=True)
    off = cf.curvature_fidelity_loss(k_q, k_g, scale=1.0, delta=1.0, access_behavior=False)
    assert not torch.allclose(on, off, rtol=1e-4, atol=1e-4)


def test_it_is_finite_and_differentiable_and_huber_bounds_the_gradient():
    k_q = torch.tensor([0.0, 100.0], requires_grad=True)
    k_g = torch.tensor([0.0, 0.0])
    value = cf.curvature_fidelity_loss(k_q, k_g, scale=1.0, delta=1.0)
    assert torch.isfinite(value)
    value.backward()
    # Huber's gradient saturates at delta for the outlier and the mean divides by the two elements,
    # so the outlier's gradient is delta/2 = 0.5 rather than growing with the residual.
    assert float(k_q.grad[1]) == pytest.approx(0.5, abs=1e-6)
    assert float(k_q.grad[0]) == pytest.approx(0.0, abs=1e-6)


def test_an_empty_triplet_set_is_refused_instead_of_returning_nan():
    with pytest.raises(ValueError):
        cf.curvature_fidelity_loss(torch.zeros(0), torch.zeros(0), scale=1.0, delta=1.0)


def test_stratum_reporting_is_separate_from_the_loss():
    residual = torch.tensor([1.0, -2.0, 0.5])
    masks = {"positive": torch.tensor([True, False, False]),
             "negative": torch.tensor([False, True, False]),
             "near_zero": torch.tensor([False, False, True])}
    out = cf.report_by_stratum(residual, masks)
    assert out["positive"] == pytest.approx(1.0)
    assert out["negative"] == pytest.approx(2.0)
