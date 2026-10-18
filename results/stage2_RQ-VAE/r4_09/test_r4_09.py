"""R4-09 tests: target-branch stop-gradient in the original L1 InfoNCE (issue #59)."""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r4_common")
import r4_testkit as K  # noqa: E402

ARM = "r4_09"
FLAG = "R4_09_TARGET_DETACH"
T = K.load_arm(ARM)

PAIRS = 4
DIM = 4


def _set(value: bool) -> None:
    if hasattr(T, FLAG):
        setattr(T, FLAG, value)


def _spy():
    seen: list[torch.Tensor] = []
    real = T._contrastive_ce

    def wrapper(logits, source_ids, target_ids, successor_indicator=None):
        seen.append(logits.detach().clone())
        return real(logits, source_ids, target_ids, successor_indicator)

    T._contrastive_ce = wrapper
    return seen, real


def _logits(residuals, source_ids, target_ids):
    seen, real = _spy()
    try:
        T.behavior_contrastive_loss(residuals, source_ids, target_ids, None)
    finally:
        T._contrastive_ce = real
    assert len(seen) == 3, len(seen)
    return seen


def _split_leaves(seed: int = 23):
    """Anchor rows and target rows as separate leaves so the two halves are visible."""
    torch.manual_seed(seed)
    anchor = torch.randn(PAIRS, DIM, requires_grad=True)
    candidate = torch.randn(PAIRS, DIM, requires_grad=True)
    block = torch.cat([anchor, candidate], dim=0)
    residuals = torch.stack([block, block, block], dim=0)
    return anchor, candidate, residuals


# ---------------------------------------------------------------- RED tests


def test_red_target_branch_receives_no_l1_gradient():
    """The L1 behaviour CE must not push gradient into the target branch.

    L2/L3 are unchanged by design and still reach both branches, so the assertion is
    made on an L1-only loss by narrowing the arm's ACTIVE_LEVELS for this test.
    """
    _set(True)
    original_levels = T.ACTIVE_LEVELS
    T.ACTIVE_LEVELS = (0,)
    try:
        anchor, candidate, residuals = _split_leaves()
        loss = T.behavior_contrastive_loss(
            residuals, torch.arange(PAIRS), torch.arange(PAIRS, 2 * PAIRS), None
        )
        loss.backward()
        anchor_grad = float(anchor.grad.abs().sum())
        candidate_grad = float(candidate.grad.abs().sum())
    finally:
        T.ACTIVE_LEVELS = original_levels
    assert anchor_grad > 0.0, "the anchor branch must still receive gradient"
    assert candidate_grad == 0.0, (
        f"target branch still receives L1 gradient (|grad| = {candidate_grad:.6e})"
    )


def test_red_l1_forward_logits_are_identical_to_baseline():
    """Detaching a branch changes no value, only the gradient path."""
    _set(False)
    residuals = torch.randn(3, 2 * PAIRS, DIM)
    source_ids = torch.arange(PAIRS, dtype=torch.long)
    target_ids = torch.arange(PAIRS, 2 * PAIRS, dtype=torch.long)
    baseline = _logits(residuals, source_ids, target_ids)[0]
    _set(True)
    enabled = _logits(residuals, source_ids, target_ids)[0]
    assert torch.equal(baseline, enabled), "L1 forward logits changed under detach"


def test_red_l2_l3_logits_and_gradients_are_identical():
    _set(False)
    residuals = torch.randn(3, 2 * PAIRS, DIM, requires_grad=True)
    source_ids = torch.arange(PAIRS, dtype=torch.long)
    target_ids = torch.arange(PAIRS, 2 * PAIRS, dtype=torch.long)
    off = T.behavior_contrastive_loss(residuals, source_ids, target_ids, None)
    off.backward()
    off_grad = residuals.grad.detach().clone()
    _set(True)
    residuals2 = residuals.detach().clone().requires_grad_(True)
    on = T.behavior_contrastive_loss(residuals2, source_ids, target_ids, None)
    on.backward()
    assert abs(float(off) - float(on)) < 1e-6, "forward loss changed under detach"
    l2 = slice(PAIRS, None)
    assert torch.allclose(off_grad[l2], residuals2.grad[l2], atol=1e-6), "L2/L3 gradient changed"


# ------------------------------------------------------------ contract tests


def test_reconstruction_path_still_updates_the_target_item():
    """The item batch keeps updating every item's representation."""
    _set(True)
    step = K.mini_step(T)
    assert step["finite"] and step["encoder_grad_norm"] > 0.0
    assert any(float(p.abs().sum()) > 0 for p in step["params"]), "encoder did not move"


def test_off_flag_matches_exact_resample():
    _set(False)
    anchor, candidate, residuals = _split_leaves()
    loss = T.behavior_contrastive_loss(
        residuals, torch.arange(PAIRS), torch.arange(PAIRS, 2 * PAIRS), None
    )
    loss.backward()
    assert float(candidate.grad.abs().sum()) > 0.0, (
        "disabled mechanism still detached the target branch"
    )
    _set(True)


def test_finite_forward_and_backward():
    step = K.mini_step(T)
    assert step["finite"] and step["encoder_grad_norm"] > 0.0


def test_train_only_data_audit():
    K.check_train_only(T)


def test_sid_three_token_contract():
    K.check_sid_contract(T)


if __name__ == "__main__":
    K.run(
        __name__,
        [
            test_red_target_branch_receives_no_l1_gradient,
            test_red_l1_forward_logits_are_identical_to_baseline,
            test_red_l2_l3_logits_and_gradients_are_identical,
            test_reconstruction_path_still_updates_the_target_item,
            test_off_flag_matches_exact_resample,
            test_finite_forward_and_backward,
            test_train_only_data_audit,
            test_sid_three_token_contract,
        ],
    )
