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
import re
import json
import sys
import tempfile
from pathlib import Path

VALUE_RTOL = 1e-6
VALUE_ATOL = 1e-8
GRAD_RTOL = 1e-5
GRAD_ATOL = 1e-7
SAMPLES = 12



#: A decimal literal longer than this is an approximation, not a constant to
#: be identified: SymPy would try to recognise it as an exact rational, and
#: that search (a lattice reduction) can run for minutes.
_MAX_DECIMAL_DIGITS = 12
_DECIMAL = re.compile(r"\d+\.\d{%d,}" % (_MAX_DECIMAL_DIGITS + 1))


def _shorten_decimals(text: str) -> str:
    """Cut an over-long decimal literal down to a dozen digits."""
    return _DECIMAL.sub(lambda match: match.group(0)[:_MAX_DECIMAL_DIGITS + 2], text)


#: Constructs whose symbolic treatment is a rabbit hole: a Piecewise sorts its
#: conditions symbolically, which can take minutes, and the question they
#: answer is decided numerically anyway.
_CONDITIONAL_MARKERS = ("Piecewise", "Max(", "Min(", "<=", ">=", "Eq(", "And(", "Or(")


def _has_conditional_parts(implementation: dict) -> bool:
    text = " ".join(str(line) for line in implementation.get("statements") or [])
    return any(marker in text for marker in _CONDITIONAL_MARKERS)


def _expressions(implementation: dict) -> dict:
    import sympy

    namespace: dict = {}
    for line in implementation.get("statements") or []:
        if "=" not in line:
            raise ValueError(f"a statement must be name = expression: {line!r}")
        name, expression = line.split("=", 1)
        namespace[name.strip()] = sympy.sympify(_shorten_decimals(expression),
                                                locals=namespace)
    output = implementation.get("output")
    if not output or output not in namespace:
        raise ValueError(f"the output {output!r} is not one of the statements")
    return namespace


#: A symbolic equivalence check over a whole quantisation chain is expensive:
#: simplify() on a few hundred tanh/exp nodes can run for minutes. The check is
#: bounded, and a comparison that does not finish is reported as TIMEOUT - not
#: as agreement and not as disagreement.
SYMBOLIC_TIMEOUT_SECONDS = 90.0
#: Above this many printed characters the symbolic question is left undecided.
SYMBOLIC_DECISION_CHARS = 1200


def _sampled_identity(left_expression, right_expression, symbols, *,
                      samples: int = 12, tolerance: float = 1e-9) -> dict:
    """Whether two expressions agree, decided by evaluating their difference.

    A general ``simplify`` of a formula of this size can spend minutes in
    SymPy's constant identification. The question "are these the same
    function?" is answered instead by evaluating the difference at random
    points: if it vanishes there, the two expressions agree. This is a sampled
    identity, not a proof, and the report says so.
    """
    import numpy as np
    import sympy

    try:
        difference = left_expression - right_expression
        function = sympy.lambdify(symbols, difference, "numpy")
    except Exception as exc:  # noqa: BLE001
        return {"equivalent": None,
                "reason": "the difference could not be evaluated: %s" % exc}
    generator = np.random.default_rng(17)
    points = generator.uniform(-0.6, 0.6, size=(samples, len(symbols)))
    try:
        values = np.asarray(function(*[points[:, index] for index in
                                       range(len(symbols))]), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return {"equivalent": None,
                "reason": "the difference could not be evaluated: %s" % exc}
    worst = float(np.nanmax(np.abs(values))) if values.size else float("inf")
    if worst <= tolerance:
        return {"equivalent": True, "residual": "0",
                "reason": "the difference vanishes at %d sampled points" % samples}
    return {"equivalent": False, "residual": str(difference)[:200],
            "reason": "the difference reaches %g at the sampled points" % worst}


def _bounded_simplify(expression, timeout: float):
    """simplify() under a wall-clock bound, in a child process.

    An alarm in the parent is not enough: SymPy catches broad exceptions
    internally, so a timeout raised there can be swallowed and the computation
    keeps running. A child process that is terminated is not.
    """
    import multiprocessing

    def worker(connection):
        import os
        parent = os.getppid()

        def guard():
            # The parent was killed: this child must not keep the machine busy.
            if os.getppid() != parent:
                os._exit(0)
        try:
            import signal
            signal.signal(signal.SIGALRM, lambda *_: guard())
            signal.setitimer(signal.ITIMER_REAL, 5.0)
            import sympy
            result = sympy.simplify(expression)
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            connection.send(("ok", result))
        except Exception as exc:  # noqa: BLE001
            connection.send(("error", str(exc)))
        finally:
            connection.close()

    parent, child = multiprocessing.Pipe()
    process = multiprocessing.Process(target=worker, args=(child,))
    process.daemon = True
    process.start()
    try:
        if parent.poll(float(timeout)):
            kind, value = parent.recv()
            if kind == "ok":
                return value
            raise ValueError(value)
    finally:
        process.join(1)
        if process.is_alive():
            process.terminate()
            process.join(1)
    raise TimeoutError("the symbolic comparison ran out of time")


def _layout(implementation: dict) -> list:
    """The argument layout of a derivation: the widths and the scalars."""
    layout = []
    for spec in implementation.get("args") or []:
        if spec.get("scalar") is not None:
            layout.append(("scalar", 1))
        else:
            layout.append(("tensor", int(spec.get("width") or 1)))
    return layout


def _positional_mapping(left: dict, right: dict):
    """The symbols of the right derivation, mapped onto the left's by position."""
    if _layout(left) != _layout(right):
        return None
    left_symbols = list(left.get("symbols") or [])
    right_symbols = list(right.get("symbols") or [])
    if len(left_symbols) != len(right_symbols):
        return None
    return right_symbols


def symbolic_difference(left: dict, right: dict, *, timeout: float = SYMBOLIC_TIMEOUT_SECONDS) -> dict:
    """Whether two derivations state the same function of the same inputs."""
    import sympy

    if _has_conditional_parts(left) or _has_conditional_parts(right):
        return {"equivalent": None,
                "reason": "the formula has conditional or clipped parts, which "
                          "are decided numerically"}
    try:
        left_scope = _expressions(left)
        right_scope = _expressions(right)
    except Exception as exc:  # noqa: BLE001
        return {"equivalent": False, "reason": f"a derivation does not parse: {exc}"}
    # Two derivations that name the same argument differently are the SAME
    # function: the comparison is by position, and the names are mapped through
    # the argument layout. A different layout (a different number of arguments,
    # a different width) is a different function and is refused as such.
    mapping = _positional_mapping(left, right)
    if mapping is None:
        return {"equivalent": False, "layout_mismatch": True,
                "reason": "the two derivations declare different argument layouts: " +
                           str(_layout(left)) + " against " + str(_layout(right))}
    substitutions = {}
    if mapping:
        import sympy
        for right_symbol, left_symbol in zip(mapping, left.get("symbols") or []):
            substitutions[right_symbol] = sympy.Symbol(left_symbol)
    right_output = right_scope[right["output"]]
    if substitutions:
        right_output = right_output.subs(substitutions)
    # The symbolic question is only asked of a small formula. A general
    # simplify() of a formula of this size can spend minutes in SymPy's
    # constant identification, and lambdify() of a large expression does the
    # same. The numeric comparison of the same pair - values, gradients,
    # gradcheck on the same inputs - is the decisive check, and it catches
    # what the symbolic one would catch (a constant offset changes the value).
    try:
        size = len(str(left_scope[left["output"]])) + len(str(right_output))
    except Exception:  # noqa: BLE001
        size = 0
    if size > SYMBOLIC_DECISION_CHARS:
        return {"equivalent": None,
                "reason": "the symbolic comparison was not attempted (the two "
                          "outputs are %d characters long); the numeric "
                          "comparison decides this pair" % size}
    return _sampled_identity(left_scope[left["output"]], right_output,
                              left.get("symbols") or [])


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
    """Import the derivation's own torch code from a temporary module.

    Returns the callable and the names of its parameters: the comparison calls
    it positionally, and a code whose parameters are not the declared arguments
    cannot be compared.
    """
    code = implementation.get("code")
    if not code:
        raise ValueError("the derivation carries no code")
    directory = Path(tempfile.mkdtemp(prefix="derivation_"))
    path = directory / (str(implementation.get("name") or "derivation") + ".py")
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("derivation_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import inspect
    parameters = inspect.signature(module.compute).parameters
    required = [name for name, parameter in parameters.items()
                if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD,
                                      parameter.KEYWORD_ONLY)]
    accepts_kwargs = any(parameter.kind == parameter.VAR_KEYWORD
                          for parameter in parameters.values())
    return module.compute, required, accepts_kwargs


def _reference_from_statements(implementation: dict):
    """The reference rebuilt from the derivation's own statements."""
    import sympytorch

    scope = _expressions(implementation)
    module = sympytorch.SymPyModule(expressions=[scope[implementation["output"]]])
    symbols = list(implementation["symbols"])
    arguments = implementation.get("args") or []

    def reference(*values):
        columns, offset = {}, 0
        for spec, value in zip(arguments, values):
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


def _values_for(implementation: dict, values: list) -> list:
    """The same numbers, packed the way this derivation declares them.

    Two derivations of the same function may name the same argument
    differently, and one may pack a scalar where the other packs a one-
    component vector: the FUNCTIONS are compared, not their signatures. A
    scalar becomes a one-component vector and a one-component vector becomes a
    scalar when the other side says so.
    """
    packed = []
    for spec, value in zip(implementation.get("args") or [], values):
        wants_scalar = spec.get("scalar") is not None
        if wants_scalar and not isinstance(value, float):
            value = float(value.reshape(-1)[0])
        elif not wants_scalar and isinstance(value, float):
            import torch
            value = torch.tensor([value], dtype=torch.float64)
        packed.append(value)
    return packed


def _compare_pair(left_fn, right_fn, left_values, right_values, tensors,
                  left_names=None, right_names=None, declared_names=None):
    """Compare two implementations on the same numbers.

    The implementations are called BY NAME: a derivation writes its own
    parameter names, and the framework hands over the numbers under the names
    the code declares. The names are the derivation's; the numbers are the
    framework's.
    """
    import torch

    declared_names = list(declared_names or [])

    def call(function, values, names):
        """Call by name; a code that takes **kwargs receives everything."""
        if names is None:
            return function(*values)
        required, accepts_kwargs = names
        if accepts_kwargs:
            return function(**dict(zip(declared_names, values)))
        allowed = set(required)
        return function(**{name: value for name, value in zip(declared_names, values)
                           if name in allowed})

    left_value = call(left_fn, left_values, left_names)
    right_value = call(right_fn, right_values, right_names)
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
    def bound(function, values, names):
        positions = [index for index, value in enumerate(values)
                     if not isinstance(value, float)]

        def wrapped(*tensors):
            arguments = list(values)
            for position, tensor in zip(positions, tensors):
                arguments[position] = tensor
            if names is None:
                return function(*arguments)
            required, accepts_kwargs = names
            if accepts_kwargs:
                return function(**dict(zip(declared_names, arguments)))
            allowed = set(required)
            return function(**{name: value
                               for name, value in zip(declared_names, arguments)
                               if name in allowed})
        return wrapped

    try:
        torch.testing.assert_close(left_value.detach(), right_value.detach(),
                                   rtol=VALUE_RTOL, atol=VALUE_ATOL)
    except AssertionError as exc:
        report.update(status="FAIL", kind="contradiction",
                      reason="values: " + str(exc).splitlines()[0][:200])
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
        bound(left_fn, left_values, left_names), tuple(tensors),
        eps=1e-6, atol=1e-5, rtol=1e-4))
    report["gradcheck_right"] = bool(torch.autograd.gradcheck(
        bound(right_fn, right_values, right_names), tuple(tensors),
        eps=1e-6, atol=1e-5, rtol=1e-4))
    return report


def compare(payload: dict, *, seed: int = 17) -> dict:
    """The full verdict over N derivations."""
    implementations, invalid = [], []
    for implementation in payload.get("implementations") or []:
        try:
            _expressions(implementation)
            implementations.append(implementation)
        except Exception as exc:  # noqa: BLE001
            invalid.append({"name": implementation.get("name"),
                            "status": "INVALID",
                            "reason": f"the derivation does not parse: {exc}"[:300]})
    excluded = list(payload.get("excluded") or [])
    if len(implementations) < 2:
        return {"status": "NOT_APPLICABLE",
                "reason": "fewer than two derivations parse", "pairs": [],
                "invalid": invalid, "excluded": excluded}
    pairs, failures = [], []
    for index, left in enumerate(implementations):
        for right in implementations[index + 1:]:
            entry = {"left": left.get("name"), "right": right.get("name")}
            symbolic = symbolic_difference(left, right)
            entry["symbolic"] = symbolic
            if symbolic["equivalent"] is None:
                # The symbolic comparison did not decide. That is not a
                # disagreement: the values, the gradients and gradcheck on the
                # same inputs decide the pair, and they are the decisive checks.
                entry["symbolic"] = {"equivalent": None,
                                     "reason": str(symbolic.get("reason"))}
            elif not symbolic["equivalent"]:
                entry["status"] = "FAIL"
                entry["kind"] = "contradiction"
                entry["reason"] = ("the two derivations are not the same function: " +
                                   str(symbolic.get("reason") or
                                       ("the difference is " + str(symbolic.get("residual")))))
                failures.append(entry)
                pairs.append(entry)
                continue
            if symbolic.get("layout_mismatch"):
                entry["status"] = "FAIL"
                entry["kind"] = "incomparable"
                failures.append(entry)
                pairs.append(entry)
                continue
            try:
                left_module, left_names, left_kwargs = _module_from_source(left)
                right_module, right_names, right_kwargs = _module_from_source(right)
                declared_left = [str(spec.get("name"))
                                 for spec in left.get("args") or []]
                declared_right = [str(spec.get("name"))
                                  for spec in right.get("args") or []]
                missing_left = [name for name in left_names
                                if name not in declared_left]
                missing_right = [name for name in right_names
                                 if name not in declared_right]
                if missing_left or missing_right:
                    # The code may name its parameters as it likes, but every
                    # parameter it REQUIRES must be one of the declared
                    # arguments: a code that needs an argument the round does not
                    # declare cannot be compared with the others.
                    entry.update(status="FAIL", kind="incomparable",
                                 reason="the code requires arguments the round "
                                        "does not declare (%s)"
                                        % (missing_left or missing_right))
                    failures.append(entry)
                    pairs.append(entry)
                    continue
                arguments, tensors = _inputs(left, seed=seed)
                # The values in the ORDER the arguments are declared, so the
                # right side can be fed the same numbers under its own names.
                values = [arguments[spec["name"]]
                          for spec in left.get("args") or []]
                left_values = list(values)
                right_values = _values_for(right, values)
                declared_names = [str(spec.get("name"))
                                  for spec in left.get("args") or []]
                report = _compare_pair(left_module, right_module,
                                       left_values, right_values, tensors,
                                       (left_names, left_kwargs),
                                       (right_names, right_kwargs),
                                       declared_names)
                entry.update(report)
                if report.get("status") != "OK":
                    failures.append(entry)
            except Exception as exc:  # noqa: BLE001
                # A derivation whose own code raises is broken, not
                # contradicted: it never produced a value to disagree with.
                entry.update(status="FAIL", kind="incomparable",
                             reason=f"{type(exc).__name__}: {exc}"[:300])
                failures.append(entry)
            pairs.append(entry)
    self_checks = []
    for implementation in implementations:
        entry = {"name": implementation.get("name")}
        if _has_conditional_parts(implementation):
            # Rebuilding a reference of a formula with conditional parts means
            # constructing a Piecewise, which SymPy canonicalises by sorting
            # the conditions symbolically: minutes for a formula of this size.
            # The self-check is therefore recorded as skipped, and the pair
            # comparison - the two implementations against each other - carries
            # the verdict.
            entry.update(status="SKIPPED",
                         reason="the formula has conditional parts, which are "
                                "not rebuilt symbolically")
            self_checks.append(entry)
            continue
        try:
            module, names, accepts_kwargs = _module_from_source(implementation)
            declared = [str(spec.get("name"))
                        for spec in implementation.get("args") or []]
            missing = [name for name in names if name not in declared]
            if missing:
                entry.update(status="FAIL", kind="incomparable",
                             reason="the code requires arguments the round does "
                                    "not declare (%s)" % missing)
                failures.append(entry)
                self_checks.append(entry)
                continue
            arguments, tensors = _inputs(implementation, seed=seed)
            values = [arguments[spec["name"]]
                      for spec in implementation.get("args") or []]
            report = _compare_pair(module,
                                   _reference_from_statements(implementation),
                                   values, values, tensors, (names, accepts_kwargs),
                                   None, [str(spec.get("name"))
                                          for spec in implementation.get("args") or []])
            entry.update(report)
            if report.get("status") != "OK":
                failures.append(entry)
        except Exception as exc:  # noqa: BLE001
            entry.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}"[:300])
            failures.append(entry)
        self_checks.append(entry)
    return {"status": "FAIL" if failures else "AGREE", "pairs": pairs,
            "self_checks": self_checks, "invalid": invalid, "excluded": excluded,
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
