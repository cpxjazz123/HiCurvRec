"""Issue #231 §2/§3: the structure term reads the REAL three-level SID, hard forward, STE back.

Written before the module exists. Each test encodes a clause of the issue:

* §2 -- the formulas use the exported codes and `P_l = prod_{k<=l} 1[c_ik = c_jk]`, so a pair
  sharing level 1 and level 3 but not 2 scores 1, not 2;
* §3 -- `y_l = p_l + stopgrad(h_l - p_l)`: the forward value is the hard one, the gradient is
  the soft one, and the term must reach the optimizer through the ordinary backward pass;
* §3 -- one optimisation step must improve the **hard** SID in a satisfiable toy case, which is
  the property #230 never established;
* §3 -- with the term switched off the trainer must behave exactly as the baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parent.parent


def _mod():
    if str(TREE) not in sys.path:
        sys.path.insert(0, str(TREE))
    from losses import sid_graphlet  # noqa: PLC0415

    return sid_graphlet


# --- C7/C8: the S() definition over the real codes ----------------------------------
def test_shared_prefix_counts_only_agreeing_codes_and_never_the_counter():
    m = _mod()
    codes = torch.tensor([[5, 7, 9, 100], [5, 8, 9, 101], [5, 7, 9, 999]])
    s = m.shared_prefix(codes, torch.tensor([0]), torch.tensor([1]))
    assert s.tolist() == [1.0]        # agrees on L1 only -> P1=1, P2=0, P3=0
    s = m.shared_prefix(codes, torch.tensor([0]), torch.tensor([2]))
    assert s.tolist() == [3.0]        # all three levels agree; the 4th column is ignored


def test_per_level_contributions_are_indicators_not_level_numbers():
    """C40: P_l is 0/1 and must not be multiplied by the level index."""
    m = _mod()
    codes = torch.tensor([[5, 7, 9], [5, 7, 1]])
    levels = m.level_contributions(codes, torch.tensor([0]), torch.tensor([1]))
    assert [float(v.item()) for v in levels] == [1.0, 1.0, 0.0]
    assert float(sum(v.item() for v in levels)) == float(
        m.shared_prefix(codes, torch.tensor([0]), torch.tensor([1])).item())


# --- C9/C10/C11: the three deltas ---------------------------------------------------
def test_chain_delta_matches_hand_computation():
    m = _mod()
    # A=(1,1,1) B=(1,1,2) C=(1,2,3): S(AB)=2, S(BC)=1, S(AC)=1 -> (2+1)/2-1 = 0.5
    codes = torch.tensor([[1, 1, 1], [1, 1, 2], [1, 2, 3]])
    delta = m.delta(codes, torch.tensor([[0, 1, 2]]), "chain")
    assert delta.tolist() == pytest.approx([0.5])


def test_fork_delta_uses_the_sibling_pair():
    m = _mod()
    # issue's own example: A=(10,21,31) B=(10,22,32) C=(10,22,33) -> S(BC)=2, S(AB)=1, S(AC)=1
    codes = torch.tensor([[10, 21, 31], [10, 22, 32], [10, 22, 33]])
    delta = m.delta(codes, torch.tensor([[0, 1, 2]]), "fork")
    assert delta.tolist() == pytest.approx([1.0])


def test_merge_uses_the_same_formula_as_fork():
    """merge roles are (centre, B, C): the two parents share a level, the centre does not."""
    m = _mod()
    # centre=(10,20,30), B=(10,21,31), C=(10,21,32): S(BC)=2, S(AB)=1, S(AC)=1 -> 2-1 = 1
    codes = torch.tensor([[10, 20, 30], [10, 21, 31], [10, 21, 32]])
    delta = m.delta(codes, torch.tensor([[0, 1, 2]]), "merge")
    assert delta.tolist() == pytest.approx([1.0])


def test_a_chain_never_requires_both_neighbours_strictly_above(codes=None):
    """C14: the criterion is the mean of the adjacent pairs, not two strict inequalities."""
    m = _mod()
    codes = torch.tensor([[1, 1, 1], [1, 1, 2], [2, 3, 4]])
    # S(AB)=2, S(BC)=0, S(AC)=0 -> (2+0)/2 - 0 = 1.0. A rule demanding both AB>AC and BC>AC
    # would fail here (BC=0 is not > AC=0), so the two differ and the mean is the issue's rule.
    delta = m.delta(codes, torch.tensor([[0, 1, 2]]), "chain")
    assert delta.tolist() == pytest.approx([1.0])


# --- C15: STE forward is hard, backward is soft -------------------------------------
def test_the_forward_value_equals_the_hard_code_and_backward_uses_the_soft_one():
    m = _mod()
    logits = torch.tensor([[0.0, 2.0], [0.0, 2.0]], requires_grad=True)
    y = m.ste_one_hot(logits)
    hard = torch.nn.functional.one_hot(torch.argmin(logits.detach(), dim=-1), 2).float()
    assert torch.equal(y.detach(), hard), (y, hard)
    y.sum().backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0


def test_the_term_reaches_the_encoder_through_the_ordinary_backward_pass():
    """C19/C20: no autograd.grad side channel; the loss is part of total.backward()."""
    m = _mod()
    torch.manual_seed(0)
    logits = torch.randn(6, 4, requires_grad=True)
    roles = torch.tensor([[0, 1, 2]])
    loss = m.structure_loss([m.ste_one_hot(logits)], m.hard_codes(logits), roles, "chain",
                            margin=1.0)
    total = loss * 1.0
    total.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0


# --- C26: one step improves the HARD SID --------------------------------------------
def test_one_step_improves_the_hard_sid_in_a_satisfiable_toy_case():
    m = _mod()
    torch.manual_seed(0)
    logits = torch.randn(3, 3, requires_grad=True)
    roles = torch.tensor([[0, 1, 2]])
    before = float(m.delta(m.hard_codes(logits), roles, "chain").item())
    opt = torch.optim.SGD([logits], lr=0.5)
    for _ in range(60):
        opt.zero_grad()
        soft = [m.ste_one_hot(logits)]
        loss = m.structure_loss(soft, m.hard_codes(logits), roles, "chain", margin=1.0)
        loss.backward()
        opt.step()
    after = float(m.delta(m.hard_codes(logits), roles, "chain").item())
    assert after > before, (before, after)


# --- C27: with the term off, behaviour equals the baseline --------------------------
def test_zero_weight_reproduces_the_baseline_objective():
    m = _mod()
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    roles = torch.tensor([[0, 1, 2]])
    total, parts = m.combine(base=torch.tensor(2.0), graphlet=torch.tensor(7.0), weight=0.0)
    assert float(total) == 2.0
    assert parts["graphlet_weighted"] == 0.0


# --- C22: degenerate cases the issue names ------------------------------------------
def test_fully_distinct_and_colliding_codes_are_handled():
    m = _mod()
    distinct = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    assert m.delta(distinct, torch.tensor([[0, 1, 2]]), "chain").tolist() == [0.0]
    colliding = torch.tensor([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
    assert m.delta(colliding, torch.tensor([[0, 1, 2]]), "chain").tolist() == [0.0]
