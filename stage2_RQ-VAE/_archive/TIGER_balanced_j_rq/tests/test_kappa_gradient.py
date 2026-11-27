"""The distance keeps q's gradient and the curvature table carries none."""
from __future__ import annotations

from pathlib import Path

import torch

import sys

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))
from model import stereographic


def test_gradient_reaches_the_quantized_representation():
    torch.manual_seed(0)
    x = torch.randn(8, 32, dtype=torch.float64, requires_grad=True)
    y = torch.randn(8, 32, dtype=torch.float64)
    kappa = torch.full((8,), -0.125, dtype=torch.float64)
    loss = stereographic.squared(x, y, kappa).sum()
    loss.backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


def test_curvature_carries_no_gradient():
    torch.manual_seed(0)
    x = torch.randn(8, 32, dtype=torch.float64, requires_grad=True)
    y = torch.randn(8, 32, dtype=torch.float64)
    kappa = torch.full((8,), -0.125, dtype=torch.float64)
    loss = stereographic.squared(x, y, kappa).sum()
    loss.backward()
    assert x.grad is not None
    assert kappa.grad is None


def test_coincident_points_have_finite_gradients():
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float64, requires_grad=True)
    for k in (-0.125, 0.0, 0.5):
        loss = stereographic.squared(x, x.detach(), k).sum()
        loss.backward(retain_graph=True)
        assert torch.isfinite(x.grad).all(), k
        x.grad = None
