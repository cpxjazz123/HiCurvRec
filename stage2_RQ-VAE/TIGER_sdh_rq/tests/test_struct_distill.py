"""Issue #181 acceptance 2-4 on the student side: the STE readout, the candidate sets, the relational KL and its gradient."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import struct_distill as sd  # noqa: E402


def test_the_ste_readout_forwards_the_quantised_value_and_backpropagates_to_the_encoder():
    z = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
    q = torch.randn(8, 6, dtype=torch.float64)
    z_hat = sd.ste_readout(z, q)
    # forward value is the quantised readout, up to the float rounding of z + (q - z)
    assert torch.allclose(z_hat.detach(), q, rtol=1e-12, atol=1e-12)
    z_hat.sum().backward()
    assert z.grad is not None and torch.allclose(z.grad, torch.ones_like(z))


def test_candidate_sets_are_deterministic_bounded_and_exclude_the_anchor():
    neighbours = {0: [1, 2, 3, 0], 1: [0]}
    two_hop = {0: [4, 5, 1]}
    first = sd.candidate_sets([0, 1], neighbours, two_hop, n_items=10, seed=7)
    second = sd.candidate_sets([0, 1], neighbours, two_hop, n_items=10, seed=7)
    assert first == second
    for anchor, chosen in zip([0, 1], first):
        assert anchor not in chosen
        assert len(chosen) == len(set(chosen))
        assert len(chosen) <= sd.ONE_HOP + sd.TWO_HOP + sd.GLOBAL
    assert 1 in first[0] and 4 in first[0]


def test_the_relational_loss_is_zero_for_a_student_that_matches_the_teacher():
    teacher = torch.randn(4, 8, dtype=torch.float64)
    assert float(sd.relational_loss(teacher.clone(), teacher)) < 1e-8


def test_the_relational_loss_grows_when_the_student_is_permuted():
    teacher = torch.randn(4, 8, dtype=torch.float64)
    student = teacher[:, torch.arange(8).flip(0)]
    assert float(sd.relational_loss(student, teacher)) > 1e-3


def test_the_teacher_side_carries_no_gradient():
    teacher = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    student = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    sd.relational_loss(student, teacher).backward()
    assert teacher.grad is None
    assert student.grad is not None and float(student.grad.abs().sum()) > 0.0


def test_zscore_is_row_wise_with_the_variance_floor():
    x = torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    z = sd.zscore(x)
    assert torch.allclose(z[0], torch.zeros(3, dtype=torch.float64), atol=1e-6)   # constant row -> floor, not NaN
    assert abs(float(z[1].mean())) < 1e-12 and abs(float(z[1].std(unbiased=False)) - 1.0) < 1e-6


def test_the_student_cosines_read_the_ste_readout_not_the_pre_quantisation_value():
    z = torch.tensor([[1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    cos = sd.student_cosines(sd.ste_readout(z, q), candidates)
    # forward uses q, so the second candidate is the aligned one
    assert float(cos[0, 1]) > float(cos[0, 0])


def test_the_lambda_schedule_is_flat_then_linear_then_capped():
    assert sd.lambda_schedule(1) == 0.0
    assert sd.lambda_schedule(sd.WARMUP_EPOCH) == 0.0
    assert abs(sd.lambda_schedule(2250) - 0.5) < 1e-12
    assert sd.lambda_schedule(sd.FULL_EPOCH) == 1.0
    assert sd.lambda_schedule(3001) == 1.0


def test_the_lambda_star_is_the_issue_formula_and_survives_a_zero_denominator():
    assert abs(sd.calibrate(2.0, 4.0) - 0.05) < 1e-12
    # a zero denominator is guarded, not capped: the value stays finite and enormous rather than raising
    import math
    assert math.isfinite(sd.calibrate(1.0, 0.0))


def test_two_hop_stops_early_on_a_hub_anchor():
    # a hub with 2000 neighbours each holding 2 far nodes: unbounded expansion is 4000 scans of an O(n) list, the bound keeps it small
    neighbours = {0: list(range(1, 2001))}
    for node in range(1, 2001):
        neighbours[node] = [5000 + node, 9000 + node]
    pool = sd.two_hop_of(0, neighbours)
    assert len(pool) <= sd.TWO_HOP * 4
    assert 0 not in pool and len(pool) == len(set(pool))
