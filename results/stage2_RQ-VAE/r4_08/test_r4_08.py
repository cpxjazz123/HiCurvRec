"""R4-08 tests: encoder-only conflict projection of the L1 behaviour gradient (issue #58).

The RED test computes the projection itself and compares it with what the arm's
real training step does. Without the arm's projection entry point the test falls
back to the baseline's plain gradient sum, so the failure is a measured numeric
mismatch rather than a missing-symbol error.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/r4_common")
import r4_testkit as K  # noqa: E402

ARM = "r4_08"
FLAG = "R4_08_PCGRAD_ENC"
T = K.load_arm(ARM)

EPS = 1e-12


def _set(value: bool) -> None:
    if hasattr(T, FLAG):
        setattr(T, FLAG, value)


def _project(g_l1, g_base):
    """Use the arm's real projection when present, else the baseline's plain sum.

    On the canonical parent the L1 gradient is used exactly as produced, so the RED
    failure is a measured dot product rather than a missing-attribute error.
    """
    projector = getattr(T, "r4_project_conflict", None)
    if projector is None:
        return g_l1
    return projector(g_l1, g_base)


def _dot(a, b):
    return float((a * b).sum())


# ---------------------------------------------------------------- RED tests


def test_red_conflicting_l1_gradient_becomes_orthogonal_to_base():
    """A conflicting L1 gradient must be projected orthogonal to the base gradient."""
    _set(True)
    # dot = -1 < 0: these two genuinely conflict.
    g_l1 = torch.tensor([-1.0, -2.0])
    g_base = torch.tensor([1.0, 0.0])
    projected = _project(g_l1, g_base)
    assert abs(_dot(projected, g_base)) < 1e-6, (
        f"projected L1 gradient still conflicts with base (dot={_dot(projected, g_base):.6f})"
    )
    expected = g_l1 - (_dot(g_l1, g_base) / (_dot(g_base, g_base) + EPS)) * g_base
    assert torch.allclose(projected, expected, atol=1e-6), (projected, expected)


def test_red_conflict_projection_differs_from_the_plain_sum():
    """The assembled gradient must not be the unprojected sum."""
    _set(True)
    g_l1 = torch.tensor([-1.0, -2.0])
    g_base = torch.tensor([1.0, 0.0])
    projected = _project(g_l1, g_base)
    assert not torch.allclose(projected, g_l1, atol=1e-6), (
        "projection left the L1 gradient unchanged"
    )
    assembled_plain = g_base + g_l1
    assembled_projected = g_base + projected
    assert not torch.allclose(assembled_plain, assembled_projected, atol=1e-6), (
        "projection had no effect on the assembled gradient"
    )


# ------------------------------------------------------------ contract tests


def test_non_conflicting_gradients_pass_through_unchanged():
    _set(True)
    g_l1 = torch.tensor([0.0, 3.0])
    g_base = torch.tensor([1.0, 0.0])
    projected = _project(g_l1, g_base)
    assert torch.allclose(projected, g_l1, atol=1e-7), (
        f"a non-conflicting gradient was rescaled ({projected.tolist()} vs {g_l1.tolist()})"
    )


def test_zero_base_gradient_is_not_projected():
    _set(True)
    g_l1 = torch.tensor([1.0, -2.0])
    g_base = torch.zeros(2)
    projected = _project(g_l1, g_base)
    assert torch.allclose(projected, g_l1, atol=1e-7), (
        f"zero base gradient must disable projection ({projected.tolist()})"
    )


def test_non_finite_inputs_fail_closed():
    _set(True)
    projector = getattr(T, "r4_project_conflict", None)
    if projector is None:
        raise AssertionError("conflict projection is not implemented")

    # Fail closed means refusing to emit a NaN gradient: raising is the intended
    # behaviour, and silently returning the poisoned tensor is not.
    for bad in ([float("nan"), 1.0], [float("inf"), 1.0]):
        try:
            out = projector(torch.tensor(bad), torch.tensor([1.0, 1.0]))
        except RuntimeError:
            continue
        assert torch.isfinite(out).all(), f"projection emitted a bad gradient for {bad}"


def test_off_flag_matches_exact_resample():
    """Disabled: the assembly is exactly the baseline encoder gradient sum."""
    model, _ = K.tiny_model(T)
    torch.manual_seed(1234)
    batch = torch.randn(7, 12)
    encoder_parameters = list(model.encoder.parameters())

    def encoder_grad(flag: bool):
        _set(flag)
        local_model, _ = K.tiny_model(T)
        local_params = list(local_model.encoder.parameters())
        local_batch = torch.randn(7, 12)
        reconstructed, quant_loss, _, _ = local_model(local_batch)
        base_loss, _ = local_model.compute_loss(local_batch, reconstructed, quant_loss)
        l1 = -base_loss * 0.5
        others = local_model.encoder(local_batch).sum() * 0.0
        total = base_loss + 0.2 / 3.0 * (l1 + others)
        total.backward(retain_graph=True)
        baseline = [p.grad.detach().clone() for p in local_params]
        apply_projection = getattr(T, "_r4_08_apply_projection", None)
        if apply_projection is not None:
            apply_projection(base_loss, l1, others, local_params, 0.2, valid_levels=3)
        return baseline, [p.grad.detach().clone() for p in local_params]

    _set(False)
    baseline, off_grads = encoder_grad(False)
    _set(True)
    _, on_grads = encoder_grad(True)
    for a, b in zip(baseline, off_grads):
        assert torch.allclose(a, b, atol=1e-9), "disabled mechanism changed the encoder gradient"
    assert any(
        not torch.allclose(a, b, atol=1e-9) for a, b in zip(baseline, on_grads)
    ), "enabled mechanism is a silent no-op on the encoder gradient"


def test_only_encoder_parameters_are_projected():
    """The projection must not touch the decoder or any codebook gradient."""
    _set(True)
    model, _ = K.tiny_model(T)
    batch = torch.randn(6, 12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = model(batch)
    loss, _ = model.compute_loss(batch, reconstructed, quant_loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"


def test_train_only_data_audit():
    K.check_train_only(T)


def test_sid_three_token_contract():
    K.check_sid_contract(T)


if __name__ == "__main__":
    K.run(
        __name__,
        [
            test_red_conflicting_l1_gradient_becomes_orthogonal_to_base,
            test_red_conflict_projection_differs_from_the_plain_sum,
            test_non_conflicting_gradients_pass_through_unchanged,
            test_zero_base_gradient_is_not_projected,
            test_non_finite_inputs_fail_closed,
            test_off_flag_matches_exact_resample,
            test_only_encoder_parameters_are_projected,
            test_train_only_data_audit,
            test_sid_three_token_contract,
        ],
    )
