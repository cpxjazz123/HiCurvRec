"""R4-06 tests: unit-normalized cosine distance for L1 contrastive logits (issue #56)."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r4_common")
import r4_testkit as K  # noqa: E402

ARM = "r4_06"
FLAG = "R4_06_COSINE_L1"
T = K.load_arm(ARM)

PAIRS = 4
DIM = 4


def _set(value: bool) -> None:
    if hasattr(T, FLAG):
        setattr(T, FLAG, value)


def _inputs(seed: int = 17):
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


def _euclid(residuals, level: int) -> torch.Tensor:
    return -torch.cdist(residuals[level, :PAIRS], residuals[level, PAIRS:], p=2) / T.BEHAVIOR_TEMPERATURE


def _cosine(residuals, level: int) -> torch.Tensor:
    source = F.normalize(residuals[level, :PAIRS], eps=1e-8)
    candidates = F.normalize(residuals[level, PAIRS:], eps=1e-8)
    return -(1.0 - source @ candidates.t()) / T.BEHAVIOR_TEMPERATURE


# ---------------------------------------------------------------- RED tests


def test_red_l1_logits_are_cosine_scores():
    """L1 logits must be -(1 - cos) / 0.07, not the Euclidean scores."""
    residuals, source_ids, target_ids = _inputs()
    seen = _logits(residuals, source_ids, target_ids)
    expected = _cosine(residuals, 0)
    assert torch.allclose(seen[0], expected, atol=1e-6), (
        f"L1 is not cosine: max|delta| = {float((seen[0] - expected).abs().max()):.6e}"
    )


def test_red_positive_rescaling_changes_euclid_but_not_cosine():
    """Scaling one vector must leave the cosine logits alone and move the Euclidean ones."""
    residuals, source_ids, target_ids = _inputs()
    baseline_cos = _logits(residuals, source_ids, target_ids)[0]
    baseline_euclid = _euclid(residuals, 0)
    residuals_scaled = residuals.clone()
    residuals_scaled[0, :1] = residuals_scaled[0, :1] * 3.5
    scaled_cos = _logits(residuals_scaled, source_ids, target_ids)[0]
    scaled_euclid = _euclid(residuals_scaled, 0)
    assert torch.allclose(baseline_cos, scaled_cos, atol=1e-6), "cosine logits changed under rescaling"
    assert not torch.allclose(baseline_euclid, scaled_euclid, atol=1e-6), (
        "test precondition: Euclidean logits must change under rescaling"
    )


def test_red_l2_l3_logits_are_bitwise_identical():
    residuals, source_ids, target_ids = _inputs()
    seen = _logits(residuals, source_ids, target_ids)
    for level, captured in ((1, seen[1]), (2, seen[2])):
        assert torch.equal(captured, _euclid(residuals, level)), f"L{level + 1} logits changed"


# ------------------------------------------------------------ contract tests


def test_zero_and_near_zero_norms_are_finite():
    _set(True)
    residuals, source_ids, target_ids = _inputs()
    residuals = residuals.clone()
    residuals[0, 0] = 0.0
    residuals[0, 1] = 1e-20
    residuals.requires_grad_(True)
    loss = T.behavior_contrastive_loss(residuals, source_ids, target_ids, None)
    loss.backward()
    assert torch.isfinite(loss).all(), "non-finite L1 loss on a zero-norm vector"
    assert torch.isfinite(residuals.grad).all(), "non-finite gradient on a zero-norm vector"


def test_matching_directions_score_equally_for_any_positive_magnitude():
    _set(True)
    residuals = torch.zeros(3, 2 * PAIRS, DIM)
    direction = torch.tensor([1.0, 2.0, -1.0, 0.5])
    residuals[0, 0] = direction
    residuals[0, PAIRS] = direction * 17.0
    source_ids = torch.arange(PAIRS, dtype=torch.long)
    target_ids = torch.arange(PAIRS, 2 * PAIRS, dtype=torch.long)
    seen = _logits(residuals, source_ids, target_ids)
    assert abs(float(seen[0][0, 0])) < 1e-4, f"aligned directions did not score ~0: {seen[0][0, 0]}"


def test_off_flag_matches_exact_resample():
    _set(False)
    residuals, source_ids, target_ids = _inputs()
    seen = _logits(residuals, source_ids, target_ids)
    assert torch.equal(seen[0], _euclid(residuals, 0)), "disabled mechanism still altered L1"
    _set(True)


def test_no_projection_head_parameters_added():
    model, _ = K.tiny_model(T)
    names = {name for name, _ in model.named_parameters()}
    assert not any("project" in name or "proj" in name for name in names), sorted(names)


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
            test_red_l1_logits_are_cosine_scores,
            test_red_positive_rescaling_changes_euclid_but_not_cosine,
            test_red_l2_l3_logits_are_bitwise_identical,
            test_zero_and_near_zero_norms_are_finite,
            test_matching_directions_score_equally_for_any_positive_magnitude,
            test_off_flag_matches_exact_resample,
            test_no_projection_head_parameters_added,
            test_finite_forward_and_backward,
            test_train_only_data_audit,
            test_sid_three_token_contract,
        ],
    )
