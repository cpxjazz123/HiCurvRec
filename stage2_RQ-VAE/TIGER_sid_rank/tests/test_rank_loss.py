"""Value, gradient, stratification, sampler and switch checks for rank_loss.

Run with the Stage 2 interpreter: `python3 tests/test_rank_loss.py`.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rank_loss as R


def test_softplus_value():
    pos = torch.tensor([1.0], requires_grad=True)
    neg = torch.tensor([0.0], requires_grad=True)
    loss = R.rank_loss(pos, neg, margin=0.5)
    assert math.isclose(loss.item(), math.log(1 + math.exp(-0.5)), rel_tol=1e-6)


def test_gradient_direction():
    pos = torch.tensor([0.3], requires_grad=True)
    neg = torch.tensor([0.3], requires_grad=True)
    R.rank_loss(pos, neg, margin=0.5).backward()
    assert pos.grad.item() < 0
    assert neg.grad.item() > 0


def test_margin_monotonic():
    pos, neg = torch.tensor([1.0]), torch.tensor([0.0])
    assert R.rank_loss(pos, neg, 0.1) < R.rank_loss(pos, neg, 1.0)


def test_similarity_is_negative_squared_distance_euclidean():
    a, b = torch.tensor([[0.0, 0.0]]), torch.tensor([[3.0, 4.0]])
    saved = R.GEOMETRY_ON
    try:
        R.GEOMETRY_ON = False
        assert math.isclose(R.similarity(a, b).item(), -25.0)
    finally:
        R.GEOMETRY_ON = saved


def test_stratified_mean_weights_strata_equally():
    big, small = torch.full((100,), 1.0), torch.full((2,), 3.0)
    assert math.isclose(R.stratified_mean([big, small]).item(), 2.0)


def test_negative_sampler_never_returns_a_neighbour():
    adjacency = {1, 2, 3, 4, 5}
    candidates = torch.arange(10, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        assert R.sample_negatives(adjacency, candidates, generator) not in adjacency


def test_switches_disable_the_terms():
    base, term = torch.tensor(1.0), torch.tensor(2.0)
    saved = (R.RANK_ON_FINAL, R.RANK_ON_COARSE)
    try:
        R.RANK_ON_FINAL, R.RANK_ON_COARSE = False, False
        assert math.isclose(R.total_loss(base, term, term).item(), 1.0, rel_tol=1e-6)
        R.RANK_ON_FINAL = True
        assert math.isclose(R.total_loss(base, term, None).item(),
                            1.0 + R.LAMBDA_FINAL * 2.0, rel_tol=1e-6)
    finally:
        R.RANK_ON_FINAL, R.RANK_ON_COARSE = saved


def test_batch_term_is_finite_and_zero_without_positives():
    q = torch.randn(4, 3, requires_grad=True)
    items = torch.arange(4)
    generator = torch.Generator().manual_seed(0)
    empty = R.batch_rank_term(q, items, {}, generator)
    assert float(empty) == 0.0
    neighbours = {0: {1}, 1: {0}}
    term = R.batch_rank_term(q, items, neighbours, generator)
    assert torch.isfinite(term)
    term.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_poincare_distance_matches_closed_form():
    zero = torch.zeros(1, 2, dtype=torch.float64)
    point = torch.tensor([[0.25, 0.0]], dtype=torch.float64)
    # tanh scaling maps 0.25 -> tanh(0.25); the closed form for (0, r) is 2*artanh(r)
    r = float(torch.tanh(torch.tensor(0.25, dtype=torch.float64)))
    expected = (2.0 * torch.atanh(torch.tensor(r)).item()) ** 2
    assert math.isclose(R.poincare_sq(zero, point).item(), expected, rel_tol=1e-6)


def test_geometry_switch_changes_the_similarity():
    a = torch.tensor([[0.2, 0.1]])
    b = torch.tensor([[0.9, 0.3]])
    saved = R.GEOMETRY_ON
    try:
        R.GEOMETRY_ON = True
        hyp = R.similarity(a, b).item()
        R.GEOMETRY_ON = False
        euc = R.similarity(a, b).item()
        assert hyp != euc
        assert math.isclose(euc, -((a - b) ** 2).sum().item(), rel_tol=1e-6)
    finally:
        R.GEOMETRY_ON = saved


def test_poincare_is_finite_at_touching_points():
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[0.0, 0.0]])
    value = R.poincare_sq(a, b)
    assert torch.isfinite(value).all()
    assert math.isclose(value.item(), 0.0, abs_tol=1e-6)


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("ok   ", name)
            except AssertionError:
                failures += 1
                print("FAIL ", name)
    raise SystemExit(1 if failures else 0)


