"""R4-05 tests: one-skip-history anchor in the contrastive pair stream (issue #55)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r4_common")
import r4_testkit as K  # noqa: E402

ARM = "r4_05"
FLAG = "R4_05_LAG2_SOURCE"
T = K.load_arm(ARM)


def _set(value: bool) -> None:
    if hasattr(T, FLAG):
        setattr(T, FLAG, value)


def _frame(histories, targets) -> pd.DataFrame:
    histories = [np.asarray(h, dtype=np.int64) for h in histories]
    return pd.DataFrame(
        {
            "user": np.arange(len(histories)),
            "history": histories,
            "seen_history": histories,
            "target": np.asarray(targets, dtype=np.int64),
        }
    )


def _toy() -> T.TransitionPairs:
    return T.TransitionPairs(
        _frame(
            histories=[[7], [7, 8], [7, 8, 9], [10, 11, 12, 13]],
            targets=[8, 9, 10, 14],
        )
    )


# ---------------------------------------------------------------- RED tests


def test_red_sources_use_the_one_skip_item_when_available():
    """source = history[-2] for sequences of length >= 2, history[-1] for length 1."""
    _set(True)
    pairs = _toy().pairs.tolist()
    expected_sources = [7, 7, 8, 12]
    actual_sources = [row[0] for row in pairs]
    assert actual_sources == expected_sources, (
        f"one-skip anchor not applied: sources {actual_sources} != {expected_sources}"
    )


def test_red_length_one_sequence_falls_back_to_last_item():
    """A single-item sequence has no [-2]; the original [-1] must be used."""
    _set(True)
    pairs = _toy().pairs.tolist()
    assert pairs[0][0] == 7, "length-1 sequence did not fall back to history[-1]"


def test_red_targets_are_untouched():
    """The positive target of every row must not move."""
    _set(True)
    pairs = _toy().pairs.tolist()
    assert [row[1] for row in pairs] == [8, 9, 10, 14]


# ------------------------------------------------------------ contract tests


def test_off_flag_matches_exact_resample():
    """Disabled: every source is the original history[-1]."""
    _set(False)
    pairs = _toy().pairs.tolist()
    assert [row[0] for row in pairs] == [7, 8, 9, 13], pairs
    _set(True)


def test_pair_count_and_row_order_preserved():
    """One positive pair per train row, same order, same length as the baseline."""
    _set(True)
    dataset = _toy()
    assert len(dataset) == 4
    off = _baseline_sources()
    on = [row[0] for row in dataset.pairs.tolist()]
    assert len(on) == len(off)
    assert on != off, "enabled mechanism is a silent no-op"


def _baseline_sources():
    _set(False)
    sources = [row[0] for row in _toy().pairs.tolist()]
    _set(True)
    return sources


def test_source_is_always_in_the_rows_past_and_never_the_future_target():
    """No future item may act as a source, and no source may equal its own target."""
    frame = _frame(histories=[[1, 2, 3], [4, 5]], targets=[6, 7])
    past = frame["seen_history"].tolist()
    targets = frame["target"].tolist()
    for row_index, row in enumerate(T.TransitionPairs(frame).pairs.tolist()):
        assert row[0] in list(past[row_index]), f"row {row_index} source is not in its own past"
        assert row[0] != targets[row_index], f"row {row_index} leaked its future target"


def test_finite_contrastive_gradients_on_lag2_pairs():
    """The contrastive path must produce finite, nonzero encoder gradients."""
    step = K.mini_step(T)
    assert step["finite"] and step["encoder_grad_norm"] > 0.0
    residuals = torch.randn(3, 8, 4, requires_grad=True)
    source_ids = torch.arange(4, dtype=torch.long)
    target_ids = torch.arange(4, 8, dtype=torch.long)
    loss = T.behavior_contrastive_loss(residuals, source_ids, target_ids, None)
    loss.backward()
    assert torch.isfinite(residuals.grad).all()
    assert float(residuals.grad.abs().sum()) > 0.0


def test_train_only_data_audit():
    K.check_train_only(T)


def test_sid_three_token_contract():
    K.check_sid_contract(T)


if __name__ == "__main__":
    K.run(
        __name__,
        [
            test_red_sources_use_the_one_skip_item_when_available,
            test_red_length_one_sequence_falls_back_to_last_item,
            test_red_targets_are_untouched,
            test_off_flag_matches_exact_resample,
            test_pair_count_and_row_order_preserved,
            test_source_is_always_in_the_rows_past_and_never_the_future_target,
            test_finite_contrastive_gradients_on_lag2_pairs,
            test_train_only_data_audit,
            test_sid_three_token_contract,
        ],
    )
