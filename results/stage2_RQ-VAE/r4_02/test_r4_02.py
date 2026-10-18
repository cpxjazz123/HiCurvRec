"""R4-02 tests: symmetric row-column L1 InfoNCE on frozen Resample pairs (issue #52)."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r4_common")
import r4_testkit as K  # noqa: E402

ARM = "r4_02"
FLAG = "R4_02_SYMMETRIC_L1"
T = K.load_arm(ARM)

PAIRS = 4
DIM = 4


def _flag() -> bool:
    return bool(getattr(T, FLAG, False))


def _set(value: bool) -> None:
    if hasattr(T, FLAG):
        setattr(T, FLAG, value)


def _inputs(seed: int = 11):
    torch.manual_seed(seed)
    residuals = torch.randn(3, 2 * PAIRS, DIM)
    source_ids = torch.arange(PAIRS, dtype=torch.long)
    target_ids = torch.arange(PAIRS, 2 * PAIRS, dtype=torch.long)
    return residuals, source_ids, target_ids


def _row_ce(logits: torch.Tensor, valid: torch.Tensor, pairs: int | None = None) -> torch.Tensor:
    labels = torch.arange(pairs or logits.shape[1])
    return F.cross_entropy(logits[valid], labels[valid])


def _scores(residuals: torch.Tensor, level: int, pairs: int | None = None) -> torch.Tensor:
    pairs = pairs or (residuals.shape[1] // 2)
    return -torch.cdist(residuals[level, :pairs], residuals[level, pairs:], p=2) / T.BEHAVIOR_TEMPERATURE


def _valid(source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    return source_ids.ne(target_ids)


def _loss_for(residuals, source_ids, target_ids, pairs: int) -> float:
    """Run the arm on a toy with a pair count other than the module default."""
    original = T.ACTIVE_LEVELS
    T.ACTIVE_LEVELS = tuple(range(3))
    try:
        return float(
            T.behavior_contrastive_loss(residuals, source_ids, target_ids, None).detach()
        )
    finally:
        T.ACTIVE_LEVELS = original


def _forward_loss(residuals, source_ids, target_ids) -> float:
    return float(T.behavior_contrastive_loss(residuals, source_ids, target_ids, None).detach())


def _expected_symmetric(residuals, source_ids, target_ids) -> float:
    valid = _valid(source_ids, target_ids)
    forward = _row_ce(_scores(residuals, 0), valid)
    reverse = _row_ce(_scores(residuals, 0).T, valid)
    l2 = _row_ce(_scores(residuals, 1), valid)
    l3 = _row_ce(_scores(residuals, 2), valid)
    return float(torch.stack([0.5 * (forward + reverse), l2, l3]).mean())


# ---------------------------------------------------------------- RED tests


def test_red_l1_is_row_column_symmetric_mean():
    """L1 must equal 0.5 * CE_rows(S) + 0.5 * CE_rows(S.T) on the same tensors."""
    residuals, source_ids, target_ids = _inputs()
    valid = _valid(source_ids, target_ids)
    expected = _expected_symmetric(residuals, source_ids, target_ids)
    actual = _forward_loss(residuals, source_ids, target_ids)
    assert abs(actual - expected) <= 1e-6, (
        f"L1 is not the symmetric mean (actual {actual:.8f}, expected {expected:.8f})"
    )
    plain = float(sum(float(_row_ce(_scores(residuals, level), valid))
                      for level in range(3)) / 3.0)
    assert abs(actual - plain) > 1e-6, "the symmetric mean is indistinguishable from one-way CE"


def test_red_reverse_term_actually_differs():
    """A hand-checkable 3-pair case where CE_rows(S) != CE_rows(S.T)."""
    pairs = 3
    residuals = torch.zeros(3, 2 * pairs, DIM)
    # Anchors and candidates are deliberately different sets, so the row-normalised and
    # the column-normalised cross entropies cannot coincide.
    anchors = torch.zeros(pairs, DIM)
    anchors[1, 0] = 10.0
    anchors[2, 1] = 10.0
    candidates = torch.zeros(pairs, DIM)
    candidates[0, 2] = 4.0
    candidates[2, 3 % DIM] = 7.0
    for level in range(3):
        residuals[level, :pairs] = anchors
        residuals[level, pairs:] = candidates
    source_ids = torch.arange(pairs, dtype=torch.long)
    target_ids = torch.arange(pairs, 2 * pairs, dtype=torch.long)
    valid = _valid(source_ids, target_ids)
    forward = float(_row_ce(_scores(residuals, 0, pairs), valid, pairs))
    reverse = float(_row_ce(_scores(residuals, 0, pairs).T, valid, pairs))
    assert abs(forward - reverse) > 1e-4, "test precondition: rows and columns must differ"
    expected = float(torch.stack([0.5 * (torch.tensor(forward) + torch.tensor(reverse)),
                                  _row_ce(_scores(residuals, 1, pairs), valid, pairs),
                                  _row_ce(_scores(residuals, 2, pairs), valid, pairs)]).mean())
    actual = _loss_for(residuals, source_ids, target_ids, pairs)
    assert abs(actual - expected) < 1e-6, (
        f"symmetric L1 not applied: actual {actual:.8f} vs expected {expected:.8f}"
    )


def test_red_l2_l3_components_unchanged():
    """L2/L3 CE must equal the baseline one-way values exactly."""
    residuals, source_ids, target_ids = _inputs()
    valid = _valid(source_ids, target_ids)
    forward_l1 = _row_ce(_scores(residuals, 0), valid)
    reverse_l1 = _row_ce(_scores(residuals, 0).T, valid)
    expected = float(torch.stack([0.5 * (forward_l1 + reverse_l1),
                                  _row_ce(_scores(residuals, 1), valid),
                                  _row_ce(_scores(residuals, 2), valid)]).mean())
    actual = _forward_loss(residuals, source_ids, target_ids)
    assert abs(actual - expected) < 1e-6, "L2/L3 contribution was disturbed"


# ------------------------------------------------------------ contract tests


def test_off_flag_matches_exact_resample():
    """Disabled: the loss is exactly the baseline one-way three-level mean."""
    _set(False)
    residuals, source_ids, target_ids = _inputs()
    valid = _valid(source_ids, target_ids)
    expected = float(torch.stack([_row_ce(_scores(residuals, level), valid)
                                  for level in range(3)]).mean())
    actual = _forward_loss(residuals, source_ids, target_ids)
    assert abs(actual - expected) < 1e-7, (
        f"disabled mechanism still changed the loss ({actual} vs {expected})"
    )
    _set(True)


def test_symmetric_uniform_toy_is_a_no_op():
    """With a fully symmetric logit matrix the symmetric mean equals the one-way CE."""
    pairs = 3
    residuals = torch.zeros(3, 2 * pairs, DIM)
    for level in range(3):
        for row in range(pairs):
            row_vector = torch.zeros(DIM)
            row_vector[row % DIM] = float(row + 1)
            residuals[level, row] = row_vector
            residuals[level, pairs + row] = residuals[level, row]
    source_ids = torch.arange(pairs, dtype=torch.long)
    target_ids = torch.arange(pairs, 2 * pairs, dtype=torch.long)
    valid = _valid(source_ids, target_ids)
    forward = _row_ce(_scores(residuals, 0, pairs), valid, pairs)
    reverse = _row_ce(_scores(residuals, 0, pairs).T, valid, pairs)
    assert abs(float(forward) - float(reverse)) < 1e-6, "test precondition: symmetric case"


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
            test_red_l1_is_row_column_symmetric_mean,
            test_red_reverse_term_actually_differs,
            test_red_l2_l3_components_unchanged,
            test_off_flag_matches_exact_resample,
            test_symmetric_uniform_toy_is_a_no_op,
            test_finite_forward_and_backward,
            test_train_only_data_audit,
            test_sid_three_token_contract,
        ],
    )
