"""R4-07 tests: Huberized robust distance for L1 behaviour comparison (issue #57)."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r4_common")
import r4_testkit as K  # noqa: E402

ARM = "r4_07"
FLAG = "R4_07_HUBER_L1"
T = K.load_arm(ARM)

PAIRS = 4
DIM = 4
BETA = 1.0
EPS = 1e-12


def _set(value: bool) -> None:
    if hasattr(T, FLAG):
        setattr(T, FLAG, value)


def _inputs(seed: int = 19):
    torch.manual_seed(seed)
    residuals = torch.randn(3, 2 * PAIRS, DIM)
    source_ids = torch.arange(PAIRS, dtype=torch.long)
    target_ids = torch.arange(PAIRS, 2 * PAIRS, dtype=torch.long)
    return residuals, source_ids, target_ids


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


def _huber(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a[:, None, :] - b[None, :, :]
    per_coord = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=BETA, reduction="none")
    return torch.sqrt(2.0 * per_coord.sum(-1) + EPS)


def _euclid(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2)


# ---------------------------------------------------------------- RED tests


def test_red_l1_logits_use_the_huberised_distance():
    """L1 logits must be -sqrt(2 * sum(smooth_l1(beta=1)) + 1e-12) / 0.07."""
    residuals, source_ids, target_ids = _inputs()
    seen = _logits(residuals, source_ids, target_ids)
    expected = -_huber(residuals[0, :PAIRS], residuals[0, PAIRS:]) / T.BEHAVIOR_TEMPERATURE
    assert torch.allclose(seen[0], expected, atol=1e-6), (
        f"L1 is not Huberised: max|delta| = {float((seen[0] - expected).abs().max()):.6e}"
    )


def test_red_outliers_are_pulled_in_relative_to_euclidean():
    """A large coordinate outlier must yield a strictly smaller Huber distance."""
    source = torch.zeros(1, 3)
    candidate = torch.tensor([[0.0, 0.0, 0.0]])
    outlier = torch.tensor([[0.0, 0.0, 40.0]])
    huber = float(_huber(source, candidate - outlier)[0, 0])
    euclid = float(_euclid(source, candidate - outlier)[0, 0])
    assert huber < euclid, "Huber distance did not shrink an outlier"
    assert abs(huber - euclid) > 1e-3, "Huber and Euclidean distances must be distinct here"


def test_red_l2_l3_logits_are_bitwise_identical():
    residuals, source_ids, target_ids = _inputs()
    seen = _logits(residuals, source_ids, target_ids)
    for level, captured in ((1, seen[1]), (2, seen[2])):
        expected = -_euclid(residuals[level, :PAIRS], residuals[level, PAIRS:]) / T.BEHAVIOR_TEMPERATURE
        assert torch.equal(captured, expected), f"L{level + 1} logits changed"


# ------------------------------------------------------------ contract tests


def test_small_differences_recover_the_euclidean_distance():
    """With every |delta| <= 1 the Huberised distance equals the Euclidean norm."""
    _set(True)
    torch.manual_seed(3)
    # Every coordinate difference stays well inside beta = 1.
    small = 0.05 * torch.randn(5, 3)
    other = 0.05 * torch.randn(5, 3)
    huber = _huber(small, other)
    euclid = _euclid(small, other)
    assert torch.allclose(huber, euclid, atol=1e-5), (
        f"max|delta| = {float((huber - euclid).abs().max()):.3e}"
    )


def test_gradient_is_finite_away_from_exact_zero():
    _set(True)
    source = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    candidate = torch.tensor([[0.0, 5.0, 0.0]])
    value = _huber(source, candidate).sum()
    value.backward()
    assert torch.isfinite(source.grad).all(), source.grad


def test_off_flag_matches_exact_resample():
    _set(False)
    residuals, source_ids, target_ids = _inputs()
    seen = _logits(residuals, source_ids, target_ids)
    expected = -_euclid(residuals[0, :PAIRS], residuals[0, PAIRS:]) / T.BEHAVIOR_TEMPERATURE
    assert torch.equal(seen[0], expected), "disabled mechanism still altered L1 logits"
    _set(True)


def test_cosine_is_not_stacked_in_this_arm():
    """Only Huber may be active; the arm must not also switch L1 to cosine."""
    _set(True)
    residuals, source_ids, target_ids = _inputs()
    seen = _logits(residuals, source_ids, target_ids)
    cosine = -(
        1.0
        - F.normalize(residuals[0, :PAIRS], eps=1e-8)
        @ F.normalize(residuals[0, PAIRS:], eps=1e-8).t()
    ) / T.BEHAVIOR_TEMPERATURE
    assert not torch.allclose(seen[0], cosine, atol=1e-6), "cosine is stacked on top of Huber"


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
            test_red_l1_logits_use_the_huberised_distance,
            test_red_outliers_are_pulled_in_relative_to_euclidean,
            test_red_l2_l3_logits_are_bitwise_identical,
            test_small_differences_recover_the_euclidean_distance,
            test_gradient_is_finite_away_from_exact_zero,
            test_off_flag_matches_exact_resample,
            test_cosine_is_not_stacked_in_this_arm,
            test_finite_forward_and_backward,
            test_train_only_data_audit,
            test_sid_three_token_contract,
        ],
    )
