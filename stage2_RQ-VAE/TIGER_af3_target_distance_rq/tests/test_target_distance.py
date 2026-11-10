"""The per-edge AF3 target-distance regression of issue #155 (red first)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import target_distance as td  # noqa: E402


def test_target_is_monotone_decreasing_in_af3():
    cuts = {"q05": -900.0, "q95": -20.0}
    low_af3 = td.normalise(torch.tensor([-900.0]), cuts)
    high_af3 = td.normalise(torch.tensor([-20.0]), cuts)
    assert float(low_af3) == pytest.approx(0.0, abs=1e-9)
    assert float(high_af3) == pytest.approx(1.0, abs=1e-9)
    below, above = td.normalise(torch.tensor([-2000.0]), cuts), td.normalise(torch.tensor([0.0]), cuts)
    assert float(below) == 0.0 and float(above) == 1.0
    target = td.target_distance(low_af3, d_ref=0.3)
    target_high = td.target_distance(high_af3, d_ref=0.3)
    assert float(target_high) < float(target)


def test_struct_loss_is_bounded_and_signed_the_right_way():
    d_ref = torch.tensor(0.5)
    distance = torch.tensor([0.4, 0.1])
    target = torch.tensor([0.5, 0.05])          # low AF3 wants far, high AF3 wants near
    aligned = td.struct_loss(distance, target, d_ref)
    wrong = td.struct_loss(torch.tensor([0.1, 0.4]), target, d_ref)
    assert 0.0 <= float(aligned) <= 1.0
    assert float(aligned) < float(wrong)


def test_struct_loss_carries_gradient_to_the_prefixes():
    # scale first, then make the tensors leaves: multiplying a leaf detaches it from .grad
    prefix_e = (torch.randn(4, 4) * 0.1).requires_grad_(True)
    prefix_h = (torch.randn(4, 4) * 0.1).requires_grad_(True)
    edges = torch.tensor([[0, 1], [2, 3]])
    target = torch.tensor([0.2, 0.1])
    loss = td.edge_regression(prefix_e, prefix_h, edges, target, torch.tensor(0.3), 1.0)
    loss.backward()
    assert prefix_h.grad is not None and float(prefix_h.grad.abs().sum()) > 0.0


def test_structure_off_zeroes_everything(monkeypatch):
    monkeypatch.setattr(td, "STRUCTURE_OFF", True)
    prefix = torch.zeros(4, 4, requires_grad=True)
    out = td.structure_losses([(prefix, prefix)], torch.tensor([[0, 1]]),
                              torch.tensor([0.2]), torch.zeros(0, 3, dtype=torch.long),
                              torch.tensor(0.3), 1.0)
    assert float(out["struct"]) == 0.0 and float(out["tri"]) == 0.0


def test_euclidean_limit_differs():
    torch.manual_seed(0)
    prefix_e = torch.randn(4, 4) * 0.1
    prefix_h = torch.randn(4, 4) * 0.1
    edges = torch.tensor([[0, 1], [2, 3]])
    target = torch.tensor([0.2, 0.1])
    curved = td.edge_regression(prefix_e, prefix_h, edges, target, torch.tensor(0.3), 1.0)
    flat = td.edge_regression(prefix_e, prefix_h, edges, target, torch.tensor(0.3), 0.0)
    assert abs(float(curved) - float(flat)) > 1e-6


def test_triangle_term_uses_the_same_normalised_distance():
    d_ref = torch.tensor(0.5)
    distance = torch.tensor([[0.05, 0.05, 0.5]])
    loss = td.triangle_term(distance, d_ref)
    assert 0.0 <= float(loss) <= 1.0
    # 1 - min(1 - d/d_ref) is max(d/d_ref) = 1.0 for the [0.1, 0.1, 1.0] triangle; the
    # earlier 0.9 expectation mis-read the formula
    assert float(loss) == pytest.approx(1.0, rel=1e-6)
