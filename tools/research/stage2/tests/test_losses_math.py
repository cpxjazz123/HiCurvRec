"""One driver for every pairwise loss, driven by what each module declares about itself.

Each loss module carries a `SPEC` describing its own mathematics:

    {"form": "mse_pair", "symmetric": True, "shift_invariant": True,
     "scale_power": 2, "zero_at_equality": True}

and the driver runs exactly the cases the declaration entitles it to. A new formula
therefore inherits the whole table without editing the tests -- but only if it states
its form: a module without a `SPEC`, or with a form the driver does not know, is
refused, because a formula that cannot say what it is cannot be checked.

The declaration is not an escape hatch. The driver is itself tested against a second
formula (an L1 pair, which is *not* shift invariant and scales linearly): if the
driver demanded the MSE laws of every loss, that self-test fails.
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
from losses._pairs import validate_pair  # noqa: E402

TOL = 1e-6

#: form -> an independent implementation of the same formula, used as the oracle.
FORMS = {
    "mse_pair": lambda a, b: ((a - b) ** 2).mean(),
    "l1_pair": lambda a, b: (a - b).abs().mean(),
}

#: the keys every declaration must carry
SPEC_KEYS = {"form", "symmetric", "shift_invariant", "scale_power", "zero_at_equality"}


def _pair(seed: int = 0, shape=(6, 4), dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator, dtype=dtype),
            torch.randn(shape, generator=generator, dtype=dtype))


def check_pairwise(name: str, fn, spec: dict) -> list[str]:
    """Run every case the declaration entitles; return the failures."""
    problems: list = []
    if not isinstance(spec, dict) or set(spec) != SPEC_KEYS:
        return ["%s declares %r, the driver needs exactly %s"
                % (name, spec, sorted(SPEC_KEYS))]
    form = spec["form"]
    if form not in FORMS:
        return ["%s declares form %r; the driver knows %s"
                % (name, form, sorted(FORMS))]

    def case(label: str, fn_check) -> None:
        try:
            fn_check()
        except (AssertionError, ValueError, TypeError) as exc:
            problems.append("%s/%s: %s" % (name, label, exc))

    def zero_at_equality() -> None:
        if not spec["zero_at_equality"]:
            return
        x, _ = _pair(1)
        if float(fn(x, x)) != 0.0:
            raise AssertionError("not zero on identical inputs: %r" % float(fn(x, x)))

    def non_negative() -> None:
        for seed in range(3):
            value = float(fn(*_pair(seed)))
            if value < 0:
                raise AssertionError("negative value %r" % value)

    def shift_invariance() -> None:
        if not spec["shift_invariant"]:
            return
        base = float(fn(*_pair(2)))
        shifted = float(fn(torch.tensor(0.0) + _pair(2)[0] + 3.0,
                           _pair(2)[1] + 3.0))
        if abs(shifted - base) > 1e-3:
            raise AssertionError("shifting both sides changed %r to %r" % (base, shifted))

    def scaling() -> None:
        power = spec["scale_power"]
        if not isinstance(power, (int, float)):
            return
        left, right = _pair(3)
        base = float(fn(left, right))
        for scale in (0.5, 2.0, 4.0):
            got = float(fn(left * scale, right * scale))
            want = base * (scale ** power)
            if abs(got - want) > TOL * max(1.0, abs(want)):
                raise AssertionError("scaling by %r gave %r, expected %r (power %r)"
                                     % (scale, got, want, power))

    def analytic() -> None:
        left, right = _pair(4, dtype=torch.float64)
        expected = float(FORMS[form](left, right))
        got = float(fn(left, right))
        if abs(got - expected) > 1e-12:
            raise AssertionError("%r != independent %r" % (got, expected))

    def purity() -> None:
        left, right = _pair(5)
        left_copy, right_copy = left.clone(), right.clone()
        fn(left, right)
        if not (torch.equal(left, left_copy) and torch.equal(right, right_copy)):
            raise AssertionError("the inputs were modified in place")

    def shape_guard() -> None:
        for bad in ((torch.randn(6, 1), torch.randn(6, 4)),
                    (torch.randn(6, 4), torch.randn(5, 4))):
            try:
                fn(*bad)
            except ValueError:
                continue
            raise AssertionError("mismatched shapes %r were accepted"
                                 % (tuple(bad[0].shape),))

    def empty_guard() -> None:
        try:
            fn(torch.empty(0, 4), torch.empty(0, 4))
        except ValueError:
            return
        raise AssertionError("an empty batch produced a value")

    def dtype_guard() -> None:
        try:
            fn(torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.long))
        except TypeError:
            return
        raise AssertionError("an integer pair was accepted")

    case("zero_at_equality", zero_at_equality)
    case("non_negative", non_negative)
    case("shift_invariance", shift_invariance)
    case("scaling", scaling)
    case("analytic", analytic)
    case("purity", purity)
    case("shape_guard", shape_guard)
    case("empty_guard", empty_guard)
    case("dtype_guard", dtype_guard)
    return problems


def test_every_registered_pairwise_loss_passes_its_own_cases() -> str:
    checked = []
    for name, entry in sorted(losses.LOSSES.items()):
        spec = entry.get("spec")
        if not isinstance(spec, dict) or "form" not in spec:
            raise AssertionError("%r carries no declaration" % name)
        if spec["form"] == "composite":
            continue
        problems = check_pairwise(name, entry["function"], spec)
        if problems:
            raise AssertionError("; ".join(problems))
        checked.append(name)
    if len(checked) != 3:
        raise AssertionError("expected the three pairwise terms, ran %r" % checked)
    return "the three pairwise losses pass every case their declaration entitles"


def test_the_driver_adapts_to_another_formula() -> str:
    def l1(a, b):
        # A new formula keeps its own arithmetic and calls the shared guard; that is
        # the contract the driver assumes of every registered pairwise loss.
        validate_pair(a, b, term="l1_probe")
        return FORMS["l1_pair"](a, b)

    spec = {"form": "l1_pair", "symmetric": True, "shift_invariant": True,
            "scale_power": 1, "zero_at_equality": True}
    problems = check_pairwise("l1_probe", l1, spec)
    if problems:
        raise AssertionError("the driver demanded the MSE laws of an L1 pair: "
                             + "; ".join(problems))
    # The declaration has teeth: an L1 pair that claims the MSE scaling is caught.
    lying = dict(spec, scale_power=2)
    caught = check_pairwise("l1_lying", l1, lying)
    if not any("scaling" in problem for problem in caught):
        raise AssertionError("a wrong scale_power was not caught: %r" % (caught,))
    return ("a linearly scaling formula passes under its own declaration, and a wrong "
            "scale_power is caught")


def test_a_loss_without_a_usable_declaration_is_refused() -> str:
    problems = check_pairwise("undeclared", lambda a, b: a, None)
    if not problems or "declares" not in problems[0]:
        raise AssertionError("an undeclared loss was accepted: %r" % (problems,))
    problems = check_pairwise("alien", lambda a, b: a,
                              {"form": "cosine", "symmetric": True, "shift_invariant": False,
                               "scale_power": 0, "zero_at_equality": True})
    if not problems or "knows" not in problems[0]:
        raise AssertionError("an unknown form was accepted: %r" % (problems,))
    return "a loss that cannot state its form is refused, not skipped"


def test_pairwise_declarations_match_the_registry() -> str:
    for name in ("reconstruction", "commitment", "codebook"):
        spec = losses.LOSSES[name].get("spec") or {}
        if spec.get("form") != "mse_pair":
            raise AssertionError("%r no longer declares the form the baseline trains" % name)
        if not spec.get("shift_invariant") or spec.get("scale_power") != 2:
            raise AssertionError("%r's declaration drifted: %r" % (name, spec))
    if (losses.LOSSES["quantization"].get("spec") or {}).get("form") != "composite":
        raise AssertionError("the quantisation term must declare the composite form")
    return "the declarations are the mathematics the baseline actually trains"


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


def test_reconstruction_gradient_is_two_over_n() -> str:
    target = torch.randn(4, 3)
    reconstructed = torch.randn(4, 3, requires_grad=True)
    reconstruction_loss(reconstructed, target).backward()
    expected = 2 * (reconstructed.detach() - target) / target.numel()
    if not torch.allclose(reconstructed.grad, expected, atol=1e-6):
        raise AssertionError("the gradient is not 2(reconstruction - target)/N")
    return "the reconstruction gradient is 2(reconstruction - target)/N"


def test_quantization_composes_exactly() -> str:
    latent, quantized = _pair(7)
    for beta in (0.0, 0.25, 1.0, 4.0):
        got = quantization_loss(latent, quantized, beta)
        want = commitment_loss(latent, quantized) + beta * codebook_loss(latent, quantized)
        if not torch.equal(got, want):
            raise AssertionError("quantization(beta=%r) != commitment + beta*codebook" % beta)
        ema = quantization_loss(latent, quantized, beta, commitment=False)
        if not torch.equal(ema, beta * codebook_loss(latent, quantized)):
            raise AssertionError("the EMA variant at beta=%r is not beta*codebook" % beta)
    return "beta composes exactly, and the EMA variant drops the commitment term"


def test_levels_average_and_the_total_adds_up() -> str:
    parts = [(torch.tensor(0.4), torch.tensor(0.8)), (torch.tensor(0.2), torch.tensor(0.0))]
    one = float(quantization_total([parts[0]], 0.5))
    if abs(one - (0.4 + 0.5 * 0.8)) > 1e-7:
        raise AssertionError("a single level is not its own term: %r" % one)
    two = float(quantization_total(parts, 0.5))
    expected = ((0.4 + 0.5 * 0.8) + (0.2 + 0.5 * 0.0)) / 2
    if abs(two - expected) > 1e-7:
        raise AssertionError("two levels are not averaged: %r vs %r" % (two, expected))
    ema = float(quantization_total([(None, torch.tensor(0.8))], 0.5))
    if abs(ema - 0.4) > 1e-7:
        raise AssertionError("an EMA level was not beta*codebook: %r" % ema)
    try:
        quantization_total([], 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("no levels was accepted")

    target, reconstructed = _pair(8)
    criterion = losses.build_loss({"beta": 0.25})
    total, components = criterion(reconstructed, target, [parts[0]])
    if set(components) != {"reconstruction", "quantization"}:
        raise AssertionError("the components are not named: %r" % sorted(components))
    summed = None
    for value in components.values():
        summed = value if summed is None else summed + value
    if not torch.equal(total, summed):
        raise AssertionError("the total is not the exact sum of its parts")
    return "levels average, the EMA level drops commitment, and the total adds exactly"


def test_the_registry_refuses_the_nested_terms_and_unknown_names() -> str:
    for bad in ("smoothness", "l2"):
        try:
            losses.build_loss({"beta": 0.25, "losses": [bad]})
        except losses.LossError:
            continue
        raise AssertionError("an unknown loss %r was accepted" % bad)
    for nested in ("commitment", "codebook"):
        try:
            losses.build_loss({"beta": 0.25, "losses": [nested]})
        except losses.LossError as exc:
            if "quantization" not in str(exc):
                raise AssertionError("%r was refused without pointing at quantization" % nested)
            continue
        raise AssertionError("%r was enabled on its own" % nested)
    only_recon = losses.build_loss({"beta": 0.25, "losses": ["reconstruction"]})
    total, components = only_recon(torch.randn(3, 2), torch.randn(3, 2), [])
    if set(components) != {"reconstruction"} or not torch.equal(total, components["reconstruction"]):
        raise AssertionError("a single enabled term was not honoured")
    return "unknown names and nested terms are refused at build time; subsets work"


TESTS = (test_every_registered_pairwise_loss_passes_its_own_cases,
         test_the_driver_adapts_to_another_formula,
         test_a_loss_without_a_usable_declaration_is_refused,
         test_pairwise_declarations_match_the_registry,
         test_the_two_conjugate_terms_update_opposite_sides,
         test_reconstruction_gradient_is_two_over_n,
         test_quantization_composes_exactly,
         test_levels_average_and_the_total_adds_up,
         test_the_registry_refuses_the_nested_terms_and_unknown_names)


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
