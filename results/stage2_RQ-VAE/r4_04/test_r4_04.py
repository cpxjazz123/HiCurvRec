"""R4-04 tests: unique-transition-uniform sampling of behavior pairs (issue #54).

The RED test computes the sampling distribution itself. When the arm does not
expose the uniform-over-unique-edges draw, the test measures the BASELINE
row-frequency distribution and fails on the measured number, so the failure is
behavioural rather than an import or attribute error.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r4_common")
import r4_testkit as K  # noqa: E402

ARM = "r4_04"
FLAG = "R4_04_UNIQUE_UNIFORM"
T = K.load_arm(ARM)

DRAWS = 20000


def _set(value: bool) -> None:
    if hasattr(T, FLAG):
        setattr(T, FLAG, value)


def _frame(edges: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user": np.arange(len(edges)),
            "history": [np.asarray([source], dtype=np.int64) for source, _ in edges],
            "seen_history": [np.asarray([source], dtype=np.int64) for source, _ in edges],
            "target": np.asarray([target for _, target in edges], dtype=np.int64),
        }
    )


def _toy_pairs(repeats: int = 100):
    """One edge repeated ``repeats`` times plus one singleton edge."""
    edges = [(3, 4)] * repeats + [(5, 6)]
    return T.TransitionPairs(_frame(edges))


def _empirical_ratio(dataset) -> float:
    """Draw from the dataset the way the trainer does, and measure the edge ratio."""
    generator = torch.Generator().manual_seed(42)
    picked = torch.randint(len(dataset), (DRAWS,), generator=generator)
    pairs = dataset[picked]
    hot = int(((pairs[:, 0] == 3) & (pairs[:, 1] == 4)).sum())
    cold = int(((pairs[:, 0] == 5) & (pairs[:, 1] == 6)).sum())
    assert cold > 0, "the singleton edge was never drawn; test precondition failed"
    return hot / cold


def _unique_support(dataset) -> set[tuple[int, int]]:
    return {tuple(row) for row in dataset.pairs.tolist()}


# ---------------------------------------------------------------- RED tests


def test_red_edge_probability_ratio_is_uniform_over_unique_edges():
    """100x-repeated vs 1x edges must draw ~1:1 instead of ~100:1."""
    _set(True)
    dataset = _toy_pairs()
    if hasattr(T, "r4_sample_uniform_pairs"):
        generator = torch.Generator().manual_seed(42)
        drawn = T.r4_sample_uniform_pairs(T._r4_04_unique_edges(dataset), DRAWS, generator)
        hot = int(((drawn[:, 0] == 3) & (drawn[:, 1] == 4)).sum())
        cold = int(((drawn[:, 0] == 5) & (drawn[:, 1] == 6)).sum())
        ratio = hot / cold
    else:
        ratio = _empirical_ratio(dataset)
    assert ratio < 2.0, (
        f"edge sampling is still row-frequency proportional (hot/cold = {ratio:.2f}, expected ~1)"
    )


def test_red_unique_support_is_unchanged():
    """The sampled edges must be a subset of the original train edge set."""
    _set(True)
    dataset = _toy_pairs()
    assert _unique_support(dataset) == {(3, 4), (5, 6)}


def test_red_repeated_row_keeps_the_pair_count_of_the_baseline():
    """Uniform draws with replacement must not shrink or grow the epoch length."""
    _set(True)
    dataset = _toy_pairs()
    assert len(dataset) == 101, "the full row stream is still the training relation"
    if hasattr(T, "r4_sample_uniform_pairs"):
        generator = torch.Generator().manual_seed(42)
        drawn = T.r4_sample_uniform_pairs(T._r4_04_unique_edges(dataset), 512, generator)
        assert drawn.shape == (512, 2), drawn.shape


# ------------------------------------------------------------ contract tests


def test_off_flag_matches_exact_resample():
    """Disabled: the baseline row-frequency draw with the baseline pair count."""
    _set(False)
    dataset = _toy_pairs()
    ratio = _empirical_ratio(dataset)
    assert 80.0 < ratio < 120.0, (
        f"disabled mechanism did not restore the baseline 100:1 draw (ratio {ratio:.1f})"
    )
    _set(True)


def test_determinism_of_the_uniform_draw():
    """Same seed, same drawn edges."""
    _set(True)
    dataset = _toy_pairs()
    if not hasattr(T, "r4_sample_uniform_pairs"):
        raise AssertionError("uniform-over-unique-edges draw is not implemented")
    edges = T._r4_04_unique_edges(dataset)
    first = T.r4_sample_uniform_pairs(edges, 256, torch.Generator().manual_seed(42))
    second = T.r4_sample_uniform_pairs(edges, 256, torch.Generator().manual_seed(42))
    assert torch.equal(first, second), "the uniform draw is not deterministic under a fixed seed"


def test_every_drawn_edge_is_a_real_train_edge():
    """No synthetic pair may enter the behaviour batch."""
    _set(True)
    dataset = _toy_pairs()
    if not hasattr(T, "r4_sample_uniform_pairs"):
        raise AssertionError("uniform-over-unique-edges draw is not implemented")
    drawn = T.r4_sample_uniform_pairs(T._r4_04_unique_edges(dataset), 1024, torch.Generator().manual_seed(7))
    support = _unique_support(dataset)
    assert all(tuple(row) in support for row in drawn.tolist())


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
            test_red_edge_probability_ratio_is_uniform_over_unique_edges,
            test_red_unique_support_is_unchanged,
            test_red_repeated_row_keeps_the_pair_count_of_the_baseline,
            test_off_flag_matches_exact_resample,
            test_determinism_of_the_uniform_draw,
            test_every_drawn_edge_is_a_real_train_edge,
            test_finite_forward_and_backward,
            test_train_only_data_audit,
            test_sid_three_token_contract,
        ],
    )
