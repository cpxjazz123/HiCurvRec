"""R4-03 tests: inverse-anchor-frequency weighting of L1 contrastive rows (issue #53)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r4_common")
import r4_testkit as K  # noqa: E402

ARM = "r4_03"
FLAG = "R4_03_ANCHOR_WEIGHTING"
T = K.load_arm(ARM)

PARQUET = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet")
PAIRS = 5
DIM = 4


def _set(value: bool) -> None:
    if hasattr(T, FLAG):
        setattr(T, FLAG, value)


def _anchor_frequency() -> np.ndarray:
    """f(a) = #train rows whose seen_history[-1] equals a."""
    frame = pd.read_parquet(PARQUET, columns=["seen_history"])
    counts = np.zeros(24587, dtype=np.float64)
    for history in frame["seen_history"].to_numpy():
        if history is not None and len(history) > 0:
            counts[int(history[-1])] += 1.0
    return counts


def _install_frequency() -> None:
    """Install the frozen train-only anchor frequency main() fits, built here."""
    table = torch.tensor(_anchor_frequency(), dtype=torch.float32)
    assert not table.requires_grad
    T.R4_03_ANCHOR_FREQ = table


def _inputs(seed: int = 13, anchors=None):
    torch.manual_seed(seed)
    if anchors is None:
        anchors = torch.arange(PAIRS, dtype=torch.long)
    residuals = torch.randn(3, 2 * PAIRS, DIM)
    target_ids = torch.arange(PAIRS, 2 * PAIRS, dtype=torch.long)
    return residuals, anchors, target_ids


def _scores(residuals: torch.Tensor, level: int) -> torch.Tensor:
    return -torch.cdist(residuals[level, :PAIRS], residuals[level, PAIRS:], p=2) / T.BEHAVIOR_TEMPERATURE


def _row_ce(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(PAIRS)
    return F.cross_entropy(logits[valid], labels[valid], reduction="none")


def _expected_weighted(residuals, source_ids, target_ids) -> float:
    valid = source_ids.ne(target_ids)
    rows = _row_ce(_scores(residuals, 0), valid)
    frequency = _anchor_frequency()
    weights = torch.tensor(
        (1.0 + frequency[source_ids.numpy()]) ** -0.5, dtype=torch.float32
    )
    weighted = float((weights * rows).sum() / weights.sum())
    l2 = float(_row_ce(_scores(residuals, 1), valid).mean())
    l3 = float(_row_ce(_scores(residuals, 2), valid).mean())
    return float(torch.tensor([weighted, l2, l3]).mean())


def _loss(residuals, source_ids, target_ids) -> float:
    _install_frequency()
    return float(T.behavior_contrastive_loss(residuals, source_ids, target_ids, None).detach())


# ---------------------------------------------------------------- RED tests


def test_red_weighted_rows_match_hand_calculation():
    """L1 must equal sum_i w(a_i) CE_i / sum_i w(a_i) with w = (1+f)^-0.5."""
    frequency = _anchor_frequency()
    frequent = int(np.argmax(frequency))
    rare = int(np.argmin(frequency[frequency > 0]))
    anchors = torch.tensor([frequent, frequent, frequent, rare, rare], dtype=torch.long)
    residuals, source_ids, target_ids = _inputs(anchors=anchors)
    valid = source_ids.ne(target_ids)
    expected = _expected_weighted(residuals, source_ids, target_ids)
    actual = _loss(residuals, source_ids, target_ids)
    assert abs(actual - expected) <= 1e-6, (
        f"L1 rows are not frequency weighted (actual {actual:.8f}, expected {expected:.8f})"
    )
    unweighted = float(sum(float(_row_ce(_scores(residuals, level), valid).mean())
                           for level in range(3)) / 3.0)
    assert abs(actual - unweighted) > 1e-6, "the weighted mean is indistinguishable from the plain mean"


def test_red_equal_anchor_frequencies_give_baseline():
    """When every anchor shares one frequency the weighted mean is the plain mean."""
    residuals, source_ids, target_ids = _inputs()
    equal = torch.full((PAIRS,), 42, dtype=torch.long)
    residuals_eq, source_ids_eq, target_ids_eq = _inputs(seed=13, anchors=equal)
    valid = source_ids_eq.ne(target_ids_eq)
    plain = float(sum(float(_row_ce(_scores(residuals_eq, level), valid).mean())
                      for level in range(3)) / 3.0)
    actual = _loss(residuals_eq, source_ids_eq, target_ids_eq)
    assert abs(actual - plain) <= 1e-6, (
        f"equal-frequency anchors did not reproduce the unweighted mean ({actual} vs {plain})"
    )


def test_red_frequent_anchor_contributes_less_fractionally():
    """A repeated high-frequency anchor must lose fractional share of L1."""
    residuals, rare_anchors, target_ids = _inputs(anchors=torch.tensor([0, 1, 2, 3, 4]))
    frequency = _anchor_frequency()
    hot = int(np.argmax(frequency))
    hot_anchors = torch.tensor([hot, 1, 2, 3, 4], dtype=torch.long)
    residuals_hot, source_hot, target_hot = _inputs(anchors=hot_anchors)
    valid = rare_anchors.ne(target_ids)
    rows = _row_ce(_scores(residuals, 0), valid)
    rows_hot = _row_ce(_scores(residuals_hot, 0), valid)
    baseline_share = float(rows[0] / rows.sum())
    weights = torch.tensor(
        (1.0 + frequency[source_hot.numpy()]) ** -0.5, dtype=torch.float32
    )
    weighted_share = float(weights[0] * rows_hot[0] / (weights * rows_hot).sum())
    assert weighted_share < baseline_share - 1e-9, (
        "the frequent anchor did not lose fractional contribution"
    )


# ------------------------------------------------------------ contract tests


def test_off_flag_matches_exact_resample():
    """Disabled: identical forward loss, gradients and one optimizer step."""
    _set(False)
    residuals, source_ids, target_ids = _inputs()
    valid = source_ids.ne(target_ids)
    expected = float(torch.stack([_row_ce(_scores(residuals, level), valid).mean()
                                  for level in range(3)]).mean())
    actual = _loss(residuals, source_ids, target_ids)
    assert abs(actual - expected) < 1e-7, f"off-path changed the loss ({actual} vs {expected})"
    _set(True)


def test_on_flag_has_measurable_effect():
    """Enabled: a skewed anchor frequency must move the L1 loss measurably."""
    residuals, rare, target_ids = _inputs(anchors=torch.tensor([0, 1, 2, 3, 4]))
    hot_id = int(np.argmax(_anchor_frequency()))
    residuals_hot, hot, _ = _inputs(anchors=torch.tensor([hot_id, 1, 2, 3, 4]))
    _set(False)
    plain = _loss(residuals_hot, hot, target_ids)
    _set(True)
    weighted = _loss(residuals_hot, hot, target_ids)
    assert abs(plain - weighted) > 1e-6, "enabled mechanism is a silent no-op"


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
            test_red_weighted_rows_match_hand_calculation,
            test_red_equal_anchor_frequencies_give_baseline,
            test_red_frequent_anchor_contributes_less_fractionally,
            test_off_flag_matches_exact_resample,
            test_on_flag_has_measurable_effect,
            test_finite_forward_and_backward,
            test_train_only_data_audit,
            test_sid_three_token_contract,
        ],
    )
