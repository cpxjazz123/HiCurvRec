"""N-version differential verification of a mechanism's formula.

Three independent derivations of the same specification must agree, and each
must agree with its own declared formula. The checks are deterministic:

1. **symbolic equivalence** - the output expressions of two derivations are
   compared in SymPy; `x**2` and `x**2 + 10` share a gradient and are still
   different functions, so this check is what separates them;
2. **forward agreement** - the two implementations are evaluated on the same
   double-precision inputs and compared with ``torch.testing.assert_close``;
3. **gradient agreement** - the gradients with respect to every input are
   compared on the same inputs;
4. **self-consistency** - each implementation is compared with a reference
   rebuilt from ITS OWN statements (sympytorch) and both sides go through
   ``torch.autograd.gradcheck``; code that disagrees with its own formula is
   refused even when every implementation agrees with every other.

What this cannot do: an argmin or a straight-through surrogate is not a
differentiable function of its inputs, so it has no numeric derivative to
compare; those surfaces must be listed in ``excluded`` with a reason and stay
with the four independent suites. Three derivations that all misread the same
specification agree with each other and are still wrong: this verifier is
evidence, not proof, and the frozen specification remains the authority.

Usage::

    python -m tools.research.rounds.differential --implementations <file.json>

The file holds ``{"implementations": [{"name": ..., "symbols": [...],
"args": [...], "statements": [...], "output": ..., "code": "..."},
...], "excluded": [...]}``; one JSON verdict line is printed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

VALUE_RTOL = 1e-6
VALUE_ATOL = 1e-8
GRAD_RTOL = 1e-5
GRAD_ATOL = 1e-7
SAMPLES = 12


def _expressions(implementation: dict) -> dict:
    import sympy

    namespace: dict = {}
    for line in implementation.get("statements") or []:
        if "=" not in line:
            raise ValueError(f"a statement must be name = expression: {line!r}")
        name, expression = line.split("=", 1)
        namespace[name.strip()] = sympy.sympify(expression, locals=namespace)
    output = implementation.get("output")
    if not output or output not in namespace:
        raise ValueError(f"the output {output!r} is not one of the statements")
    return namespace


def symbolic_difference(left: dict, right: dict) -> dict:
    """Whether two derivations state the same function of the same inputs."""
    import sympy

    try:
        left_scope = _expressions(left)
        right_scope = _expressions(right)
    except Exception as exc:  # noqa: BLE001
        return {"equivalent": False, "reason": f"a derivation does not parse: {exc}"}
    if list(left.get("symbols") or []) != list(right.get("symbols") or []):
        return {"equivalent": False,
                "reason": "the two derivations use different symbol lists"}
    difference = sympy.simplify(left_scope[left["output"]] -
                                right_scope[right["output"]])
    try:
        equivalent = bool(difference.equals(0))
    except Exception:  # noqa: BLE001
        equivalent = bool(difference == 0)
    return {"equivalent": equivalent,
            "residual": str(difference)[:200] if not equivalent else "0"}


def _inputs(implementation: dict, *, seed: int):
    import torch

    generator = torch.Generator().manual_seed(seed)
    arguments, tensors = {}, []
    for spec in implementation.get("args") or []:
        if spec.get("scalar") is not None:
            arguments[spec["name"]] = float(spec["scalar"])
            continue
        width = int(spec.get("width") or 1)
        radius = float(spec.get("radius") or 0.1)
        raw = (torch.rand(SAMPLES, width, generator=generator, dtype=torch.float64) * 2 - 1)
        if spec.get("sampler", "ball") == "ball":
            norms = raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
            raw = raw / norms * (radius * torch.rand(SAMPLES, 1, generator=generator,
                                                     dtype=torch.float64))
        else:
            raw = raw * radius
        arguments[spec["name"]] = raw.requires_grad_(True)
        tensors.append(arguments[spec["name"]])
    return arguments, tensors


def _module_from_source(implementation: dict):
    """Import the derivation's own torch code from a temporary module."""
    code = implementation.get("code")
    if not code:
        raise ValueError("the derivation carries no code")
    directory = Path(tempfile.mkdtemp(prefix="derivation_"))
    path = directory / (str(implementation.get("name") or "derivation") + ".py")
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("derivation_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute


def _reference_from_statements(implementation: dict):
    import sympytorch

    scope = _expressions(implementation)
    module = sympytorch.SymPyModule(expressions=[scope[implementation["output"]]])
    symbols = list(implementation["symbols"])

    def reference(**arguments):
        columns, offset = {}, 0
        for spec in implementation.get("args") or []:
            value = arguments[spec["name"]]
            if spec.get("scalar") is not None:
                columns[symbols[offset]] = value
                offset += 1
                continue
            width = int(spec.get("width") or 1)
            for index in range(width):
                columns[symbols[offset + index]] = value[:, index]
            offset += width
        result = module(**columns)
        return result.squeeze(-1) if result.dim() > 1 else result

    return reference


def _compare_pair(left_fn, right_fn, arguments, tensors, *, seed):
    import torch

    left_value, right_value = left_fn(**arguments), right_fn(**arguments)
    report = {"status": "OK",
              "value_max_abs_diff": float(
                  (left_value.detach() - right_value.detach()).abs().max())}
    left_grads = torch.autograd.grad(left_value.sum(), tensors, allow_unused=True)
    right_grads = torch.autograd.grad(right_value.sum(), tensors, allow_unused=True)
    worst = 0.0
    for left, right in zip(left_grads, right_grads):
        if left is not None and right is not None:
            worst = max(worst, float((left.detach() - right.detach()).abs().max()))
    report["gradient_max_abs_diff"] = worst
    names = [name for name, value in arguments.items() if not isinstance(value, float)]

    def bound(function):
        def wrapped(*values):
            bound_arguments = dict(arguments)
            bound_arguments.update(dict(zip(names, values)))
            return function(**bound_arguments)
        return wrapped

    try:
        torch.testing.assert_close(left_value.detach(), right_value.detach(),
                                   rtol=VALUE_RTOL, atol=VALUE_ATOL)
    except AssertionError as exc:
        report.update(status="FAIL", reason="values: " + str(exc).splitlines()[0][:200])
        return report
    for left, right in zip(left_grads, right_grads):
        if left is None or right is None:
            if left is not right:
                report.update(status="FAIL",
                              reason="one implementation has a gradient the other lacks")
                return report
            continue
        try:
            torch.testing.assert_close(left.detach(), right.detach(),
                                       rtol=GRAD_RTOL, atol=GRAD_ATOL)
        except AssertionError as exc:
            report.update(status="FAIL", reason="gradients: " + str(exc).splitlines()[0][:200])
            return report
    report["gradcheck_left"] = bool(torch.autograd.gradcheck(
        bound(left_fn), tuple(tensors), eps=1e-6, atol=1e-5, rtol=1e-4))
    report["gradcheck_right"] = bool(torch.autograd.gradcheck(
        bound(right_fn), tuple(tensors), eps=1e-6, atol=1e-5, rtol=1e-4))
    return report


def compare(payload: dict, *, seed: int = 17) -> dict:
    """The full verdict over N derivations."""
    implementations = list(payload.get("implementations") or [])
    excluded = list(payload.get("excluded") or [])
    if len(implementations) < 2:
        return {"status": "NOT_APPLICABLE",
                "reason": "fewer than two derivations take part", "pairs": [],
                "excluded": excluded}
    pairs, failures = [], []
    for index, left in enumerate(implementations):
        for right in implementations[index + 1:]:
            entry = {"left": left.get("name"), "right": right.get("name")}
            symbolic = symbolic_difference(left, right)
            entry["symbolic"] = symbolic
            if not symbolic["equivalent"]:
                entry["status"] = "FAIL"
                entry["reason"] = ("the two derivations are not the same function: "
                                   + ("different symbols" if "symbols" in symbolic.get("reason", "")
                                      else "the difference is " + symbolic.get("residual", "")))
                failures.append(entry)
                pairs.append(entry)
                continue
            try:
                report = _compare_pair(_module_from_source(left),
                                       _module_from_source(right),
                                       *_inputs(left, seed=seed), seed=seed)
                entry.update(report)
                if report.get("status") != "OK":
                    failures.append(entry)
            except Exception as exc:  # noqa: BLE001
                entry.update(status="FAIL",
                             reason=f"{type(exc).__name__}: {exc}"[:300])
                failures.append(entry)
            pairs.append(entry)
    self_checks = []
    for implementation in implementations:
        entry = {"name": implementation.get("name")}
        try:
            report = _compare_pair(_module_from_source(implementation),
                                   _reference_from_statements(implementation),
                                   *_inputs(implementation, seed=seed), seed=seed)
            entry.update(report)
            if report.get("status") != "OK":
                failures.append(entry)
        except Exception as exc:  # noqa: BLE001
            entry.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}"[:300])
            failures.append(entry)
        self_checks.append(entry)
    return {"status": "FAIL" if failures else "AGREE", "pairs": pairs,
            "self_checks": self_checks, "excluded": excluded,
            "reason": "" if not failures else failures[0].get("reason", "")}


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementations", required=True)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args(argv)
    payload = json.loads(Path(arguments.implementations).read_text(encoding="utf-8"))
    verdict = compare(payload, seed=arguments.seed)
    print(json.dumps(verdict, default=str))
    return 0 if verdict.get("status") in ("AGREE", "NOT_APPLICABLE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
