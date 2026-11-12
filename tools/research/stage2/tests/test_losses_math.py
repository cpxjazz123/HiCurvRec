"""One table of mathematical cases, applied to every loss the baseline owns.

Each loss is a different formula, but they are all mean-squared errors over one pair
of tensors, so the same properties must hold for all of them: zero at equality, never
negative, unchanged when both sides shift together, quadratic in a common scale, equal
to an explicit elementwise mean, exact in float64, and pure. The differences that do
matter -- which side is detached, how the terms compose -- are tested once, for the
one loss they belong to.

Run: <stage2 python> tools/research/stage2/tests/test_losses_math.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BASELINE = REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
sys.path.insert(0, str(BASELINE))

import torch  # noqa: E402

import losses  # noqa: E402
from losses import codebook_loss, commitment_loss, quantization_loss, \
    quantization_total, reconstruction_loss  # noqa: E402

TOL = 1e-6

#: name -> the pairwise loss, and how to call it on a `(left, right)` pair.
PAIRWISE = {
    "reconstruction": lambda left, right: reconstruction_loss(left, right),
    "commitment": lambda left, right: commitment_loss(left, right),   # (latent, quantized)
    "codebook": lambda left, right: codebook_loss(left, right),       # (latent, quantized)
}


def _pair(seed: int = 0, shape=(6, 4), dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(shape, generator=generator, dtype=dtype)
    right = torch.randn(shape, generator=generator, dtype=dtype)
    return left, right


def test_every_loss_is_zero_at_equality() -> str:
    for name, fn in PAIRWISE.items():
        x, _ = _pair(1)
        value = float(fn(x, x))
        if value != 0.0:
            raise AssertionError("%s(x, x) = %r, not 0" % (name, value))
    return "all three pairwise losses are exactly zero on identical inputs"


def test_every_loss_is_non_negative() -> str:
    for name, fn in PAIRWISE.items():
        for seed in range(5):
            value = float(fn(*_pair(seed)))
            if value < 0:
                raise AssertionError("%s returned a negative value: %r" % (name, value))
    return "no pairwise loss is negative on random inputs"


def test_shifting_both_sides_leaves_the_loss_alone() -> str:
    for name, fn in PAIRWISE.items():
        left, right = _pair(2)
        base = float(fn(left, right))
        shifted = float(fn(left + 3.0, right + 3.0))
        if abs(shifted - base) > 1e-3:
            raise AssertionError("%s is not shift invariant: %r vs %r"
                                 % (name, base, shifted))
    return "the losses depend on the difference, not on the absolute position"


def test_a_common_scale_enters_squared() -> str:
    for name, fn in PAIRWISE.items():
        left, right = _pair(3)
        base = float(fn(left, right))
        for scale in (0.5, 2.0, 4.0):
            scaled = float(fn(left * scale, right * scale))
            expected = base * scale * scale
            if abs(scaled - expected) > TOL * max(1.0, expected):
                raise AssertionError("%s does not scale as s^2 at s=%r: %r vs %r"
                                     % (name, scale, scaled, expected))
    return "a common scale multiplies the loss by its square"


def test_every_loss_matches_an_explicit_elementwise_mean() -> str:
    for name, fn in PAIRWISE.items():
        left, right = _pair(4, dtype=torch.float64)
        explicit = float(((left - right) ** 2).mean())
        value = float(fn(left, right))
        if abs(value - explicit) > 1e-12:
            raise AssertionError("%s = %r, the elementwise mean is %r"
                                 % (name, value, explicit))
    return "each loss equals the explicit mean of the squared differences"


def test_float64_values_are_exact() -> str:
    for name, fn in PAIRWISE.items():
        left = torch.tensor([[1.0, -2.0], [0.5, 4.0]], dtype=torch.float64)
        right = torch.tensor([[0.0, -1.0], [1.5, 4.0]], dtype=torch.float64)
        expected = ((left - right) ** 2).mean().item()
        value = float(fn(left, right))
        if abs(value - expected) > 1e-15:
            raise AssertionError("%s drifted in float64: %r vs %r" % (name, value, expected))
    return "in float64 the losses agree with the analytic value to 1e-15"


def test_every_loss_leaves_its_inputs_untouched() -> str:
    for name, fn in PAIRWISE.items():
        left, right = _pair(5)
        left_copy, right_copy = left.clone(), right.clone()
        fn(left, right)
        if not (torch.equal(left, left_copy) and torch.equal(right, right_copy)):
            raise AssertionError("%s modified its inputs in place" % name)
    return "no loss mutates the tensors it is handed"


def test_every_loss_refuses_a_shape_mismatch() -> str:
    for name, fn in PAIRWISE.items():
        for bad in ((torch.randn(6, 1), torch.randn(6, 4)),
                    (torch.randn(6, 4), torch.randn(5, 4))):
            try:
                fn(*bad)
            except ValueError:
                continue
            raise AssertionError("%s accepted mismatched shapes %r; mse_loss would "
                                 "broadcast them" % (name, tuple(bad[0].shape)))
    return "a shape mismatch raises instead of broadcasting into a plausible number"


def test_every_loss_refuses_an_empty_batch() -> str:
    for name, fn in PAIRWISE.items():
        try:
            fn(torch.empty(0, 4), torch.empty(0, 4))
        except ValueError:
            continue
        raise AssertionError("%s returned a value for an empty batch (NaN upstream)" % name)
    return "an empty batch raises instead of returning NaN"


def test_every_loss_refuses_an_integer_pair() -> str:
    for name, fn in PAIRWISE.items():
        try:
            fn(torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.long))
        except TypeError:
            continue
        raise AssertionError("%s accepted an integer pair" % name)
    return "an integer pair raises instead of being silently cast"


def test_the_two_conjugate_terms_update_opposite_sides() -> str:
    latent, quantized = _pair(6)
    latent.requires_grad_(True)
    quantized.requires_grad_(True)
    commitment_loss(latent, quantized).backward()
    if latent.grad is not None:
        raise AssertionError("commitment moved the encoder's input; it must be detached")
    if quantized.grad is None or float(quantized.grad.abs().sum()) == 0:
        raise AssertionError("commitment did not move the code it compares against")

    latent2, quantized2 = _pair(6)
    latent2.requires_grad_(True)
    quantized2.requires_grad_(True)
    codebook_loss(latent2, quantized2).backward()
    if quantized2.grad is not None:
        raise AssertionError("codebook moved the code's value; it must be detached")
    if latent2.grad is None or float(latent2.grad.abs().sum()) == 0:
        raise AssertionError("codebook did not move the encoder output")
    return "commitment moves only the codebook side, codebook only the encoder side"


def test_reconstruction_moves_the_reconstruction_only_by_default() -> str:
    target = torch.randn(4, 3)
    reconstructed = torch.randn(4, 3, requires_grad=True)
    reconstruction_loss(reconstructed, target).backward()
    if reconstructed.grad is None:
        raise AssertionError("the reconstruction loss does not reach the decoder output")
    expected = 2 * (reconstructed.detach() - target) / target.numel()
    if not torch.allclose(reconstructed.grad, expected, atol=1e-6):
        raise AssertionError("the gradient is not 2(recon - target)/N")
    return "the reconstruction gradient is 2(reconstruction - target)/N"


def test_quantization_is_commitment_plus_beta_codebook() -> str:
    latent, quantized = _pair(7)
    for beta in (0.0, 0.25, 1.0, 4.0):
        got = quantization_loss(latent, quantized, beta)
        want = commitment_loss(latent, quantized) + beta * codebook_loss(latent, quantized)
        if not torch.equal(got, want):
            raise AssertionError("quantization(beta=%r) is not commitment + beta*codebook"
                                 % beta)
        ema = quantization_loss(latent, quantized, beta, commitment=False)
        if not torch.equal(ema, beta * codebook_loss(latent, quantized)):
            raise AssertionError("the EMA variant at beta=%r is not beta*codebook" % beta)
    return "beta composes exactly, and the EMA variant drops the commitment term"


def test_levels_are_averaged_not_summed() -> str:
    parts = [(torch.tensor(0.4), torch.tensor(0.8)),
             (torch.tensor(0.2), torch.tensor(0.0))]
    one = quantization_total([parts[0]], 0.5)
    if abs(float(one) - (0.4 + 0.5 * 0.8)) > 1e-7:
        raise AssertionError("a single level is not its own term: %r" % float(one))
    two = quantization_total(parts, 0.5)
    expected = ((0.4 + 0.5 * 0.8) + (0.2 + 0.5 * 0.0)) / 2
    if abs(float(two) - expected) > 1e-7:
        raise AssertionError("two levels are not averaged: %r vs %r" % (float(two), expected))
    ema = quantization_total([(None, torch.tensor(0.8))], 0.5)
    if abs(float(ema) - 0.4) > 1e-7:
        raise AssertionError("an EMA level was not scored as beta*codebook: %r" % float(ema))
    try:
        quantization_total([], 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("no levels was accepted")
    return "levels average, the EMA level drops commitment, and no levels raises"


def test_the_total_reports_its_parts_and_adds_them() -> str:
    target, reconstructed = _pair(8)
    parts = [(torch.tensor(0.4), torch.tensor(0.8))]
    criterion = losses.build_loss({"beta": 0.25})
    total, components = criterion(reconstructed, target, parts)
    if set(components) != {"reconstruction", "quantization"}:
        raise AssertionError("the components are not named: %r" % sorted(components))
    summed = None
    for value in components.values():
        summed = value if summed is None else summed + value
    if not torch.equal(total, summed):
        raise AssertionError("the total is not the exact sum of its parts")
    only_recon = losses.build_loss({"beta": 0.25, "losses": ["reconstruction"]})
    total2, components2 = only_recon(reconstructed, target, parts)
    if set(components2) != {"reconstruction"} or not torch.equal(total2, components2["reconstruction"]):
        raise AssertionError("a single enabled term was not honoured")
    return "the total is the exact sum of the named parts, and subsets work"


def test_the_registry_is_consistent() -> str:
    for name in ("reconstruction", "commitment", "codebook", "quantization"):
        entry = losses.LOSSES.get(name)
        if entry is None:
            raise AssertionError("the registry lost %r" % name)
        if not entry.get("description"):
            raise AssertionError("%r has no description" % name)
        if not callable(entry.get("function")):
            raise AssertionError("%r has no callable" % name)
    for bad in ("smoothness", "l2"):
        try:
            losses.build_loss({"beta": 0.25, "losses": [bad]})
        except losses.LossError:
            continue
        raise AssertionError("an unknown loss name %r was accepted" % bad)
    for nested in ("commitment", "codebook"):
        try:
            losses.build_loss({"beta": 0.25, "losses": [nested]})
        except losses.LossError as exc:
            if "quantization" not in str(exc):
                raise AssertionError("enabling %r does not point at quantization: %s"
                                     % (nested, exc))
            continue
        raise AssertionError("%r was enabled on its own" % nested)
    return "every term is registered with a definition, and bad names are refused"


TESTS = (test_every_loss_is_zero_at_equality,
         test_every_loss_is_non_negative,
         test_shifting_both_sides_leaves_the_loss_alone,
         test_a_common_scale_enters_squared,
         test_every_loss_matches_an_explicit_elementwise_mean,
         test_float64_values_are_exact,
         test_every_loss_leaves_its_inputs_untouched,
         test_every_loss_refuses_a_shape_mismatch,
         test_every_loss_refuses_an_empty_batch,
         test_every_loss_refuses_an_integer_pair,
         test_the_two_conjugate_terms_update_opposite_sides,
         test_reconstruction_moves_the_reconstruction_only_by_default,
         test_quantization_is_commitment_plus_beta_codebook,
         test_levels_are_averaged_not_summed,
         test_the_total_reports_its_parts_and_adds_them,
         test_the_registry_is_consistent)


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            print("PASS %s: %s" % (test.__name__, test()), flush=True)
        except (AssertionError, ValueError, TypeError) as exc:
            failures += 1
            print("FAIL %s: %s" % (test.__name__, exc), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
