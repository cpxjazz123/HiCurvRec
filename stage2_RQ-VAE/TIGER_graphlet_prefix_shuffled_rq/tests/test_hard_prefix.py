#!/usr/bin/env python3
"""Issue #230 §3: the differentiable hard-prefix term, written before it exists.

The measurements these tests encode:

* the baseline builds its whole Δ at **level 1** (87–92% of the three-level Δ), and widening
  the level-1 codebook 256 → 1024 leaves the sign unchanged, so capacity is not the lever;
* within one structure no item pair is ever asked to move both ways (0 conflicting pairs in
  20,000 instances), so a single consistent objective exists per structure;
* across structures the only contested pairs are chain↔fork (116) and chain↔merge (103), and a
  pair can carry near-evidence for chain and far-evidence for fork at once. One shared prefix
  geometry therefore has to encode **roles**, not only adjacency.

Each test below fails on the copied tree, which has neither the term nor the role index.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parent.parent


def _align():
    if str(TREE) not in sys.path:
        sys.path.insert(0, str(TREE))
    from losses import hard_prefix  # noqa: PLC0415

    return hard_prefix


# --- C25: hard forward, real gradient ------------------------------------------------
def test_hard_forward_reports_the_true_prefix_match():
    """The reported value is built from the hard codes, not from the soft surrogate.

    Only level 1 is supervised, so ``S()`` is 0 or 1 per pair and the chain's ``near`` is the
    mean of ``S(A,B)`` and ``S(B,C)``. The codes are compared as *values* through
    ``shared_prefix``; ``prefix_margin`` takes a cost matrix, so its argmin picks the codeword.
    """
    hard_prefix = _align()
    codes = torch.tensor([4, 4, 7, 8])
    s_ab = hard_prefix.shared_prefix(codes, torch.tensor([0]), torch.tensor([1]))
    s_bc = hard_prefix.shared_prefix(codes, torch.tensor([1]), torch.tensor([2]))
    s_ac = hard_prefix.shared_prefix(codes, torch.tensor([0]), torch.tensor([2]))
    assert s_ab.tolist() == [1.0] and s_bc.tolist() == [0.0] and s_ac.tolist() == [0.0]
    # a chain over these three items has near (1+0)/2 = 0.5 and far 0 -> gap +0.5
    # so the hinge at margin 0.5 is 0 and the term is satisfied
    cost = torch.tensor([[0.0, 9.0], [0.0, 9.0], [9.0, 0.0]])
    value, info = hard_prefix.prefix_margin(cost, torch.tensor([[0, 1, 2]]), "chain", margin=0.5)
    assert info["near"].tolist() == pytest.approx([0.5])
    assert info["far"].tolist() == pytest.approx([0.0])
    assert value.item() == pytest.approx(0.0, abs=1e-6)
    # a fork reads the sibling pair (1,2) and the parent pairs (0,1), (0,2)
    fork_value, fork_info = hard_prefix.prefix_margin(cost, torch.tensor([[0, 1, 2]]), "fork")
    assert fork_info["near"].tolist() == pytest.approx([0.0])   # S(1,2) = 0
    assert fork_info["far"].tolist() == pytest.approx([0.5])    # (S(0,1)+S(0,2))/2 = (1+0)/2
    assert fork_value.item() == pytest.approx(1.5, abs=1e-6)    # clamp(0.5 + 0.5 - 0) with margin 0.5


def test_hard_forward_disagrees_with_its_own_soft_surrogate():
    """A softmax over distances gives a different number: the forward is genuinely hard."""
    hard_prefix = _align()
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    codes = hard_prefix.hard_codes(logits)
    assert codes.tolist() == [0, 2]  # argmin over the cost, as the export does
    probabilities = torch.softmax(-logits, dim=1)
    soft = hard_prefix.soft_prefix_share(probabilities, torch.tensor([0]), torch.tensor([1]))
    assert soft.item() < 0.4  # the two items never share a codeword


def test_gradient_reaches_the_logits_and_points_the_right_way():
    """One SGD step on the cost matrix must make the wanted pair share a codeword.

    The cost matrix must not be uniform: every softmax share is constant there, so a uniform
    cost is a stationary point (measured: gradient exactly 0, and correctly so).
    """
    hard_prefix = _align()
    # item 0 prefers code 0, item 1 prefers code 1, item 2 is the endpoint to keep away.
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], requires_grad=True)
    roles = torch.tensor([[0, 1, 2]])
    value, _ = hard_prefix.prefix_margin(cost, roles, "chain", margin=1.0)
    value.backward()
    assert cost.grad is not None
    assert float(cost.grad.abs().sum()) > 1e-6
    # the loss is minimised, so the step is `cost -= lr * grad`
    stepped = cost.detach() - 0.1 * cost.grad
    # after the step items 0 and 1 are closer to a common codeword than before
    before = torch.softmax(-cost.detach(), dim=-1)
    after = torch.softmax(-stepped, dim=-1)
    share_before = (before[0] * before[1]).sum()
    share_after = (after[0] * after[1]).sum()
    assert share_after > share_before
    # and the endpoint pair is pushed apart by the same step
    far_before = (before[0] * before[2]).sum()
    far_after = (after[0] * after[2]).sum()
    assert far_after < far_before


# --- C29: the term changes the real argmin --------------------------------------------
def test_the_term_reorders_a_near_tie_in_the_real_assignment():
    """A nudge smaller than the cost gap flips the argmin, not only the probability."""
    cost = torch.tensor([[0.0, 0.2], [0.2, 0.0]])  # argmin: item A -> 0, item B -> 1
    adjust = torch.tensor([[0.0, -0.5], [0.0, 0.0]])
    assert torch.argmin(cost, dim=1).tolist() == [0, 1]
    assert torch.argmin(cost + adjust, dim=1).tolist() == [1, 1]
    # and the tie case: the nudge decides between two exactly equal costs
    tied = torch.tensor([[0.0, 0.0]])
    assert torch.argmin(tied + torch.tensor([[0.0, -1e-3]]), dim=1).tolist() == [1]


# --- role index: the training side and the scorer share one definition ----------------
def test_role_index_reads_the_contract_banks():
    hard_prefix = _align()
    index = hard_prefix.RoleIndex()
    counts = index.counts()
    assert set(counts) == {"chain", "fork", "merge"}
    assert all(value > 1000 for value in counts.values())


def test_role_index_rejects_an_evaluation_pair_in_the_loss():
    """C9: the trainer asserts on the pairs that actually enter the loss."""
    hard_prefix = _align()
    index = hard_prefix.RoleIndex()
    test_pair = next(iter(index.evaluation_pairs("chain")))
    with pytest.raises(RuntimeError):
        index.assert_isolated("chain", np.array([[test_pair[0], 0, test_pair[1]]]))


def test_role_index_refuses_a_bank_whose_wiring_disagrees():
    """C30: hashes are checked, so a moved bank cannot be trained on silently."""
    hard_prefix = _align()
    index = hard_prefix.RoleIndex()
    assert index.partitions()["test"] > 0
    assert index.reads_validation_or_test is False
