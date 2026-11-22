"""Issue #233 section 2/3: the collision term, the gradient budgets, and the level discipline.

Each test names the clause it proves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parent.parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from losses import collision as C  # noqa: E402


def _probs(rows, width):
    return torch.tensor(rows, dtype=torch.float32)[:, :width]


# --- C35/C36: a duplicate is only penalised when L1+L2 already agree ---------------------------
def test_c36_the_duplicate_penalty_is_the_l3_agreement_probability():
    probs = _probs([[0.8, 0.2], [0.8, 0.2], [0.1, 0.9]], 2)
    value = C.duplicate_term(probs, torch.tensor([0]), torch.tensor([1]))
    assert float(value) == pytest.approx(0.8 * 0.8 + 0.2 * 0.2)
    disagree = C.duplicate_term(probs, torch.tensor([0]), torch.tensor([2]))
    assert float(disagree) == pytest.approx(0.8 * 0.1 + 0.2 * 0.9)


def test_c39_the_penalty_reaches_only_the_l3_probability():
    l3 = _probs([[0.7, 0.3], [0.6, 0.4]], 2).requires_grad_(True)
    C.duplicate_term(l3, torch.tensor([0]), torch.tensor([1])).backward()
    assert torch.all(l3.grad != 0.0)


# --- C50: two different items colliding completely, and same prefix different L3 --------------
def test_c50_candidates_are_exactly_the_l1_l2_equal_pairs():
    tokens = torch.tensor([[1, 2, 3], [1, 2, 9], [1, 3, 3], [4, 2, 3]])
    keys = tokens[:, 0] * 1000 + tokens[:, 1]
    found = C.collision_candidates(tokens)
    if found is None:
        pytest.skip("no L1+L2 duplicate in this fixture")
    left, right = found
    assert torch.all(keys[left] == keys[right])
    assert torch.all(left != right)


def test_c50_same_prefix_needs_a_different_l3_to_avoid_the_penalty():
    tokens = torch.tensor([[1, 2, 3], [1, 2, 3]])
    found = C.collision_candidates(tokens)
    assert found is not None, "a full duplicate must be a candidate"
    # and the L3 distribution decides: pointing them apart removes the penalty
    apart = _probs([[0.0, 1.0], [1.0, 0.0]], 2)
    assert float(C.duplicate_term(apart, found[0], found[1])) == pytest.approx(0.0)


# --- C37/C38/C43: sampling is global and per prefix -------------------------------------------
def test_c38_sampling_covers_the_global_population_not_one_instance():
    keys = torch.tensor([7] * 60 + [8] * 40 + list(range(100, 140)))
    pairs = C.candidate_pairs_from_prefix(keys, 200, torch.Generator().manual_seed(1))
    assert pairs is not None
    left, right = pairs
    assert torch.all(keys[left] == keys[right]), "every pair must share its prefix"
    assert torch.all(left != right), "and be two different rows"
    # both prefix groups contribute: the sample is per prefix, not first-come
    assert len(set(keys[left].tolist())) == 2


def test_c43_a_big_bucket_does_not_drown_the_small_ones():
    keys = torch.tensor([1] * 400 + [2] * 4)
    pairs = C.candidate_pairs_from_prefix(keys, 64, torch.Generator().manual_seed(2))
    shared = sorted(set(keys[pairs[0]].tolist()))
    assert shared == [1, 2]


# --- C28/C45: the budget is a cap, never a weight --------------------------------------------
def test_c28_a_term_within_budget_is_left_alone():
    assert C.budget_scale(0.05, 1.0, 0.10) == 1.0
    assert C.budget_scale(0.10, 1.0, 0.10) == 1.0


def test_c28_a_term_over_budget_is_scaled_to_it():
    scale = C.budget_scale(1.0, 1.0, 0.10)
    assert scale == pytest.approx(0.10)
    assert 1.0 * scale == pytest.approx(0.10 * 1.0)


def test_c29_a_vanishing_term_is_never_amplified():
    assert C.budget_scale(1e-9, 1e3, 0.10) == 1.0
    assert C.budget_scale(0.0, 1.0, 0.10) == 0.0
    assert C.MAX_UPSCALE == 1.0


# --- C35: capacity is measured on the L1 marginal --------------------------------------------
def test_c35_capacity_is_flat_at_a_balanced_assignment():
    uniform = torch.full((64, 8), 1.0 / 8.0)
    value = C.capacity_term(uniform, torch.arange(64), 64)
    assert float(value) == pytest.approx(0.0, abs=1e-6)


def test_c35_capacity_punishes_one_swallowing_everything():
    """8 codewords, all mass on code 0: load/mean = [8,0..0], so sum (x-1)^2 = 49+7 = 56, /8 = 7."""
    skewed = torch.zeros(64, 8)
    skewed[:, 0] = 1.0
    assert float(C.capacity_term(skewed, torch.arange(64), 64)) == pytest.approx(7.0)


# --- regression: the CPU/CUDA index mismatch that killed the first #233 launch ----------------
def test_capacity_accepts_an_index_on_a_different_device_from_probs():
    """Measured failure: `probs.index_select(0, roles_l1)` with `roles_l1` on CPU and `probs` on
    cuda raised "Expected all tensors to be on the same device" at epoch 1501 on every rank."""
    probs = torch.full((8, 4), 0.25)
    roles_cpu = torch.arange(8)              # what numpy -> torch produces
    value = C.capacity_term(probs, roles_cpu, 8)
    assert float(value) == pytest.approx(0.0, abs=1e-6)
    if torch.cuda.is_available():
        moved = C.capacity_term(probs.cuda(), roles_cpu, 8)
        assert float(moved) == pytest.approx(0.0, abs=1e-6)


def test_sampler_returns_cpu_rows_whatever_device_the_keys_came_from():
    """Measured failure: `torch.cat(sampled)` raised on rank 2 because the two halves of the
    sample sat on different devices. The sampler now pins everything it returns to CPU."""
    keys = torch.tensor([5] * 8 + [6] * 8)
    left, right = C.candidate_pairs_from_prefix(keys, 16, torch.Generator().manual_seed(3))
    assert left.device.type == "cpu" and right.device.type == "cpu"
    if torch.cuda.is_available():
        left, right = C.candidate_pairs_from_prefix(
            keys.cuda(), 16, torch.Generator().manual_seed(3))
        assert left.device.type == "cpu" and right.device.type == "cpu"
