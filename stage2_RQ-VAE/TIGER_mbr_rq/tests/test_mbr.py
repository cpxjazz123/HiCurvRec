"""Issue #186: the Moebius residual step, its kappa = 0 limit and the boundary behaviour."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import mbr_step as m  # noqa: E402


def test_kappa_zero_is_the_flat_step_bitwise():
    x = torch.randn(8, 5, dtype=torch.float64)
    y = torch.randn(8, 5, dtype=torch.float64)
    kappa = torch.zeros(8, 1, dtype=torch.float64)
    assert torch.allclose(m.mobius_sub(x, y, kappa), y - x, rtol=1e-12, atol=1e-15)


def test_step_off_reproduces_the_flat_step_bitwise():
    x = torch.randn(4, 3, dtype=torch.float64)
    y = torch.randn(4, 3, dtype=torch.float64)
    kappa = torch.full((4, 1), 2.0, dtype=torch.float64)
    m.STEP_OFF = True
    got = m.mobius_sub(x, y, kappa)
    m.STEP_OFF = False
    assert torch.equal(got, y - x)


def test_kappa_one_matches_the_frozen_mobius_subtraction():
    import importlib.util
    ref_path = Path("/fs04/ar57/wenyu/GeneRec") / "stage2_RQ-VAE" / "TIGER_mixed_rq" / "model" / "poincare_contract.py"
    spec = importlib.util.spec_from_file_location("frozen_contract", ref_path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    x = torch.randn(6, 4, dtype=torch.float64) * 0.2
    y = torch.randn(6, 4, dtype=torch.float64) * 0.2
    kappa = torch.ones(6, 1, dtype=torch.float64)
    mine = m.mobius_sub(x, y, kappa)
    theirs = frozen.mobius_add(-x, y, torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(mine, theirs, rtol=1e-6, atol=1e-9)


def test_a_point_minus_itself_is_zero():
    x = torch.randn(5, 4, dtype=torch.float64) * 0.3
    for k in (0.0, 1.0, 3.0):
        kappa = torch.full((5, 1), k, dtype=torch.float64)
        out = m.mobius_sub(x, x, kappa)
        assert float(out.abs().max()) < 1e-9, k


def test_the_result_stays_inside_the_ball_at_the_curvatures_the_table_holds():
    x = torch.randn(16, 6, dtype=torch.float64) * 0.9
    y = torch.randn(16, 6, dtype=torch.float64) * 0.9
    for k in (1e-6, 1e-3, 0.1, 1.0, 8.2671):
        kappa = torch.full((16, 1), k, dtype=torch.float64)
        out = m.mobius_sub(x, y, kappa)
        assert float((k * out.pow(2).sum(-1)).max()) <= 1.0 + 1e-9, k


def test_gradients_are_finite_at_small_and_large_curvature():
    # .requires_grad_(True) after the scaling: multiplying first would make x and y non-leaf tensors, whose .grad is never populated
    x = (torch.randn(8, 4, dtype=torch.float64) * 0.5).requires_grad_(True)
    y = (torch.randn(8, 4, dtype=torch.float64) * 0.5).requires_grad_(True)
    for k in (1e-6, 1e-3, 0.1, 1.0, 8.2671):
        kappa = torch.full((8, 1), k, dtype=torch.float64)
        x.grad = None
        y.grad = None
        out = m.mobius_sub(x, y, kappa)
        out.sum().backward(retain_graph=True)
        assert bool(torch.isfinite(out).all()), k
        assert x.grad is not None and bool(torch.isfinite(x.grad).all()), k
        assert y.grad is not None and bool(torch.isfinite(y.grad).all()), k


def test_the_flat_limit_is_residual_minus_code_not_the_reverse():
    # the arm calls mobius_sub(code, residual): at kappa = 0 that is residual - code, the baseline's step. The reversed order would give
    # code - residual, a sign flip on every level.
    residual = torch.randn(5, 4, dtype=torch.float64)
    code = torch.randn(5, 4, dtype=torch.float64)
    m.STEP_OFF = True
    got = m.mobius_sub(code, residual, torch.ones(5, 1, dtype=torch.float64))
    m.STEP_OFF = False
    assert torch.equal(got, residual - code)
    assert not torch.equal(got, code - residual)
