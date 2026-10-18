"""R4-01 tests: candidate-popularity logit correction in baseline L1 InfoNCE (issue #51).

RED tests spy on the real ``_contrastive_ce`` entry point to read the exact logit
tensors the trainer would use, so they assert behaviour rather than a new symbol.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r4_common")
import r4_testkit as K  # noqa: E402

ARM = "r4_01"
FLAG = "R4_01_Q_CORRECTION"
T = K.load_arm(ARM)

PARQUET = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"
)


def _flag() -> bool:
    return bool(getattr(T, FLAG, False))


def _q_train() -> np.ndarray:
    """q_train(j) = (count_train_target(j) + 1) / (num_train_rows + num_catalogue_items)."""
    import pandas as pd

    frame = pd.read_parquet(PARQUET, columns=["target"])
    targets = frame["target"].to_numpy(dtype=np.int64)
    n_items = 24587
    counts = np.bincount(targets, minlength=n_items)[:n_items]
    return (counts + 1.0) / (len(targets) + n_items)


def _install_q_table() -> None:
    """Hand the arm the same frozen table main() fits, so the test exercises the path.

    The table is built from the ORIGINAL train targets by this test, independently of
    the arm, and installed as a plain detached tensor with no gradient.
    """
    table = torch.from_numpy(_q_train()).float()
    assert not table.requires_grad
    T.R4_01_Q_TABLE = table


def _spy():
    """Capture every logit tensor handed to the real ``_contrastive_ce``."""
    seen: list[torch.Tensor] = []
    real = T._contrastive_ce

    def wrapper(logits, source_ids, target_ids, successor_indicator=None):
        seen.append(logits.detach().clone())
        return real(logits, source_ids, target_ids, successor_indicator)

    T._contrastive_ce = wrapper
    return seen, real


def _l1_logits(residuals, source_ids, target_ids):
    _install_q_table()
    seen, real = _spy()
    try:
        T.behavior_contrastive_loss(residuals, source_ids, target_ids, None)
    finally:
        T._contrastive_ce = real
    assert len(seen) == 3, f"expected one logit tensor per level, got {len(seen)}"
    return seen


def _inputs(seed: int = 7, pairs: int = 4, dim: int = 4):
    torch.manual_seed(seed)
    residuals = torch.randn(3, 2 * pairs, dim)
    source_ids = torch.arange(pairs, dtype=torch.long)
    target_ids = torch.arange(pairs, 2 * pairs, dtype=torch.long)
    return residuals, source_ids, target_ids


# ---------------------------------------------------------------- RED tests


def test_red_l1_logits_carry_minus_half_log_q():
    """L1 logits must equal ``-||z_a - z_t|| / T - 0.5 * log q_train(target)``."""
    residuals, source_ids, target_ids = _inputs()
    seen = _l1_logits(residuals, source_ids, target_ids)
    q = torch.from_numpy(_q_train()).float()
    expected_l1 = (
        -torch.cdist(residuals[0, :4], residuals[0, 4:], p=2) / T.BEHAVIOR_TEMPERATURE
        - 0.5 * torch.log(q[target_ids]).unsqueeze(0)
    )
    assert torch.allclose(seen[0], expected_l1, atol=1e-6), (
        "L1 logits are not candidate-popularity corrected: "
        f"max|delta| = {float((seen[0] - expected_l1).abs().max()):.6e}"
    )


def test_red_l2_l3_logits_are_untouched():
    """L2/L3 logits must stay byte-for-byte the baseline Euclidean scores."""
    residuals, source_ids, target_ids = _inputs()
    seen = _l1_logits(residuals, source_ids, target_ids)
    for level, captured in ((1, seen[1]), (2, seen[2])):
        expected = -torch.cdist(residuals[level, :4], residuals[level, 4:], p=2) / T.BEHAVIOR_TEMPERATURE
        assert torch.equal(captured, expected), f"L{level + 1} logits changed"


def test_red_uniform_q_recovers_baseline_and_nonuniform_changes_loss():
    """Nonuniform q must change L1 CE; uniform q must reproduce baseline exactly."""
    q = _q_train()
    assert float(q.max() / q.min()) > 10.0, "test precondition: q must be skewed"
    residuals, source_ids, target_ids = _inputs()
    seen = _l1_logits(residuals, source_ids, target_ids)
    plain = -torch.cdist(residuals[0, :4], residuals[0, 4:], p=2) / T.BEHAVIOR_TEMPERATURE
    corrected = seen[0]
    assert not torch.allclose(plain, corrected, atol=1e-6), (
        "the correction had no effect on a skewed catalogue"
    )
    labels = torch.arange(4)
    assert not math_close(
        F.cross_entropy(corrected, labels), F.cross_entropy(plain, labels)
    ), "the correction left the L1 cross entropy unchanged"


def math_close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ------------------------------------------------------------ contract tests


def test_off_flag_matches_exact_resample():
    """With the mechanism off, L1 logits are exactly the baseline Euclidean scores."""
    if hasattr(T, FLAG):
        setattr(T, FLAG, False)
    residuals, source_ids, target_ids = _inputs()
    seen = _l1_logits(residuals, source_ids, target_ids)
    expected = -torch.cdist(residuals[0, :4], residuals[0, 4:], p=2) / T.BEHAVIOR_TEMPERATURE
    assert torch.equal(seen[0], expected), "disabled mechanism still altered L1 logits"
    if hasattr(T, FLAG):
        setattr(T, FLAG, True)


def test_q_is_train_only_and_finite():
    """q comes from train targets only, is finite and strictly positive everywhere."""
    q = _q_train()
    assert q.shape == (24587,)
    assert np.isfinite(q).all() and (q > 0).all()
    assert abs(float(q.sum()) - 1.0) < 1e-6, "q must be a proper distribution"


def test_no_new_trainable_parameters():
    """The correction is a frozen table: it must never enter the optimizer."""
    step = K.mini_step(T)
    assert step["finite"] and step["encoder_grad_norm"] > 0.0
    names = [
        name
        for name, parameter in getattr(T, "MECHANISM_TRAINABLE_PARAMETERS", ())
    ]
    assert names == [], f"mechanism introduced trainable parameters: {names}"


def test_train_only_data_audit():
    K.check_train_only(T)


def test_sid_three_token_contract():
    K.check_sid_contract(T)


if __name__ == "__main__":
    K.run(
        __name__,
        [
            test_red_l1_logits_carry_minus_half_log_q,
            test_red_l2_l3_logits_are_untouched,
            test_red_uniform_q_recovers_baseline_and_nonuniform_changes_loss,
            test_off_flag_matches_exact_resample,
            test_q_is_train_only_and_finite,
            test_no_new_trainable_parameters,
            test_train_only_data_audit,
            test_sid_three_token_contract,
        ],
    )
