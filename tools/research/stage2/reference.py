"""The formula reference: every declared equation, built twice and compared.

A mechanism's continuous surface is declared, in the candidate tree, as SymPy
statements (`reference_contract.json`).  The framework builds an independent
reference from those statements (sympytorch), then compares the candidate's own
function against it in double precision: the values, the gradients and
`torch.autograd.gradcheck` on both sides.  The agent that writes the mechanism
therefore does not also get to write the only statement of what it computes.

What the contract covers, and what it does not: the differentiable surface
(distances, kernels, penalties, loss terms, residual updates).  The discrete
parts - a quantizer's argmin, a straight-through estimator's surrogate - are
not differentiable functions of the inputs and cannot be compared this way;
they must be listed under "excluded" with a reason, and stay with the four
independent suites and the one-epoch smoke.

Contract shape (one entry per declared equation)::

    {"calls": [{"name": "distance",
                "callable": "tools.research.geometry.poincare_contract:distance",
                "symbols": ["x0", "x1", "y0", "y1", "c"],
                "args": [{"name": "x", "width": 2, "radius": 0.4},
                         {"name": "y", "width": 2, "radius": 0.4},
                         {"name": "c", "scalar": 1.0}],
                "statements": ["x2 = x0**2 + x1**2", ...],
                "output": "d"}],
     "excluded": [{"surface": "quantizer argmin",
                   "reason": "an argmin is not differentiable"}]}

Usage: ``python -m tools.research.stage2.reference --tree <candidate tree>``
prints one JSON line with the verdict.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from pathlib import Path

CONTRACT_NAME = "reference_contract.json"

#: A decimal literal longer than this is an approximation: SymPy would try to
#: recognise it as an exact rational, and that search can run for minutes.
_DECIMAL = re.compile(r"\d+\.\d{15,}")

#: The tolerance of a value comparison between two double-precision paths.
VALUE_RTOL = 1e-6
VALUE_ATOL = 1e-8
#: Gradients are compared a little more loosely: the two paths reach the same
#: formula by different algebra, and float64 loses a few digits in the tail.
GRAD_RTOL = 1e-5
GRAD_ATOL = 1e-7


def _statements(contract_call: dict) -> tuple[dict, object]:
    import sympy

    namespace: dict = {}
    for line in contract_call.get("statements") or []:
        if "=" not in line:
            raise ValueError(f"a statement must be name = expression: {line!r}")
        name, expression = line.split("=", 1)
        namespace[name.strip()] = sympy.sympify(
            _DECIMAL.sub(lambda match: match.group(0)[:14], expression),
            locals=namespace)
    output = contract_call.get("output")
    if not output or output not in namespace:
        raise ValueError(f"the output {output!r} is not one of the statements")
    return namespace, namespace[output]


def build_reference(contract_call: dict):
    """The independent implementation, from the SymPy statements alone."""
    import sympytorch

    namespace, expression = _statements(contract_call)
    symbols = [sympy_symbol for sympy_symbol in contract_call["symbols"]]
    module = sympytorch.SymPyModule(expressions=[expression])

    def reference(**arguments):
        import torch
        for spec in contract_call["args"]:
            value = arguments[spec["name"]]
            if spec.get("positive") and (value <= 0).any():
                raise ValueError("the argument %r must be positive" % spec["name"])
            if spec.get("choices"):
                allowed = torch.tensor(spec["choices"], dtype=value.dtype)
                if not torch.isin(value, allowed).all():
                    raise ValueError("the argument %r leaves its declared choices" % spec["name"])
        if contract_call.get("codeword_h_max") is not None and "codes" in arguments:
            h = arguments["codes"][..., 16:]
            if (h.norm(dim=-1) > float(contract_call["codeword_h_max"])).any():
                raise ValueError("the codeword H norm leaves the declared domain")
        columns = {}
        for spec in contract_call["args"]:
            value = arguments[spec["name"]]
            width = int(spec.get("width") or 1)
            if spec.get("scalar") is not None:
                columns[symbols[_consumed(contract_call, spec)]] = value
                continue
            for index in range(width):
                columns[symbols[_consumed(contract_call, spec) + index]] = value[:, index]
        result = module(**columns)
        return result.squeeze(-1) if result.dim() > 1 else result

    return reference, namespace


def _consumed(contract_call: dict, spec: dict) -> int:
    """The offset of this argument's first symbol in the declared symbol list."""
    offset = 0
    for other in contract_call["args"]:
        if other is spec:
            return offset
        offset += 1 if other.get("scalar") is not None else int(other.get("width") or 1)
    return offset


def _load_callable(contract_call: dict):
    module_name, _, attribute = contract_call["callable"].partition(":")
    return getattr(importlib.import_module(module_name), attribute)


def _inputs(contract_call: dict, *, seed: int):
    """Double-precision inputs inside the declared domain, one sample row each."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    arguments, tensors = {}, []
    for spec in contract_call["args"]:
        if spec.get("scalar") is not None:
            arguments[spec["name"]] = float(spec["scalar"])
            continue
        width = int(spec["width"] or 1)
        radius = float(spec.get("radius") or 0.1)
        count = int(spec.get("samples") or 16)
        if spec.get("choices"):
            choices = torch.tensor(spec["choices"], dtype=torch.float64)
            raw = choices[torch.randint(0, len(choices), (count, width), generator=generator)]
        elif spec.get("positive"):
            low = float(spec.get("min", 0.1))
            high = float(spec.get("max", 1.0))
            raw = torch.rand(count, width, generator=generator, dtype=torch.float64) * (high - low) + low
        else:
            raw = (torch.rand(count, width, generator=generator, dtype=torch.float64) * 2 - 1)
        if spec.get("sampler", "ball") == "ball":
            # Inside the ball of the declared radius: scale each row's norm.
            norms = raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
            raw = raw / norms * (radius * torch.rand(count, 1, generator=generator,
                                                     dtype=torch.float64))
        if not (spec.get("choices") or spec.get("positive")):
            raw = raw * radius
        raw = raw.requires_grad_(True)
        arguments[spec["name"]] = raw
        tensors.append(raw)
    return arguments, tensors


def _gradients(result, tensors):
    import torch

    return torch.autograd.grad(result.sum(), tensors, allow_unused=True)


def compare(tree: Path, contract: dict, *, seed: int = 17) -> dict:
    """Compare the candidate's function against the reference, call by call."""
    import torch

    calls = contract.get("calls") or []
    excluded = contract.get("excluded") or []
    if not calls:
        return {"status": "NOT_APPLICABLE", "reason": "the contract declares no equation",
                "excluded": excluded, "calls": []}
    verdicts = []
    for contract_call in calls:
        entry = {"name": contract_call.get("name"), "status": "OK"}
        try:
            reference, _ = build_reference(contract_call)
            candidate = _load_callable(contract_call)
            arguments, tensors = _inputs(contract_call, seed=seed)
        except Exception as exc:  # noqa: BLE001
            entry.update(status="FAIL", reason=f"the call could not be prepared: {exc}")
            verdicts.append(entry)
            continue
        try:
            candidate_value = candidate(**arguments)
            reference_value = reference(**arguments)
            entry["samples"] = len(reference_value)
            entry["value_max_abs_diff"] = float(
                (candidate_value.detach() - reference_value.detach()).abs().max())
            # A contract may declare the tolerance its surface needs: two
            # algebraic forms of one formula differ in the last digits, and the
            # declared tolerance says how much that difference is.
            torch.testing.assert_close(candidate_value.detach(), reference_value.detach(),
                                       rtol=float(contract_call.get("rtol") or VALUE_RTOL),
                                       atol=float(contract_call.get("atol") or VALUE_ATOL))
            candidate_grads = _gradients(candidate_value, tensors)
            reference_grads = _gradients(reference_value, tensors)
            worst, mismatch = 0.0, None
            for left, right in zip(candidate_grads, reference_grads):
                if left is None or right is None:
                    if left is not right:
                        mismatch = "one path has a gradient the other lacks"
                    continue
                worst = max(worst, float((left.detach() - right.detach()).abs().max()))
            entry["gradient_max_abs_diff"] = worst
            if mismatch:
                raise AssertionError(mismatch)
            for left, right in zip(candidate_grads, reference_grads):
                if left is None or right is None:
                    continue
                torch.testing.assert_close(
                    left.detach(), right.detach(),
                    rtol=float(contract_call.get("grad_rtol") or GRAD_RTOL),
                    atol=float(contract_call.get("grad_atol") or GRAD_ATOL))
            tensor_names = [spec["name"] for spec in contract_call["args"]
                            if spec.get("scalar") is None]

            def bound(function, names):
                def wrapped(*values):
                    bound_arguments = dict(arguments)
                    bound_arguments.update(dict(zip(names, values)))
                    return function(**bound_arguments)
                return wrapped

            entry["gradcheck_candidate"] = bool(torch.autograd.gradcheck(
                bound(candidate, tensor_names), tuple(tensors),
                eps=1e-6, atol=1e-5, rtol=1e-4))
            entry["gradcheck_reference"] = bool(torch.autograd.gradcheck(
                bound(reference, tensor_names), tuple(tensors),
                eps=1e-6, atol=1e-5, rtol=1e-4))
        except AssertionError as exc:
            entry.update(status="FAIL", reason=str(exc).splitlines()[0][:300])
        except Exception as exc:  # noqa: BLE001
            entry.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}"[:300])
        verdicts.append(entry)
    failed = [entry for entry in verdicts if entry["status"] != "OK"]
    return {"status": "FAIL" if failed else "OK", "calls": verdicts,
            "excluded": excluded, "reason": "" if not failed else failed[0]["reason"]}


def coupling_verdict(contract: dict, *, seed: int = 17) -> dict:
    """Whether the candidate's BEHAVIOUR term can move the computation.

    The evidence probe answers a general question: a positive row scalar leaves
    every arg-min unchanged, so a mechanism whose behaviour enters as a row
    scalar can never change the assignment. That question is about a proxy, not
    about this candidate. The candidate answers it with its own code: the
    contract declares the callable, the argument the behaviour enters, and a
    control argument; the framework calls both and requires the result to
    change. A behaviour that cannot move the computation is refused here.
    """
    declaration = contract.get("coupling") or {}
    if not declaration.get("callable"):
        return {"status": "NOT_APPLICABLE",
                "reason": "the contract declares no coupling surface"}
    try:
        function = _load_callable(declaration)
    except Exception as exc:  # noqa: BLE001
        return {"status": "FAIL", "reason": "the coupling surface could not be loaded: %s" % exc}
    named = declaration.get("behaviour_arguments")
    if named is None:
        single = declaration.get("behaviour_argument")
        named = [single] if single else []
    behaviour_arguments = [str(name) for name in named if name]
    control = str(declaration.get("control_argument") or "")
    if not behaviour_arguments:
        return {"status": "FAIL", "reason": "the coupling declaration names no behaviour argument"}
    import torch

    generator = torch.Generator().manual_seed(seed)
    arguments = {}
    for spec in declaration.get("args") or []:
        if spec.get("scalar") is not None:
            arguments[spec["name"]] = float(spec["scalar"])
            continue
        width = int(spec.get("width") or 1)
        raw = torch.rand(8, width, generator=generator, dtype=torch.float64) * 0.2 - 0.1
        arguments[spec["name"]] = raw
    try:
        first = function(**arguments)
    except Exception as exc:  # noqa: BLE001
        return {"status": "FAIL", "reason": "the coupling surface could not be called: %s" % exc}
    # Every behaviour argument is perturbed: an anchor that the sampled rows
    # never select would otherwise look like a behaviour that does nothing.
    difference = 0.0
    for name in behaviour_arguments:
        changed = dict(arguments)
        changed[name] = arguments[name] + 0.5
        try:
            second = function(**changed)
        except Exception as exc:  # noqa: BLE001
            return {"status": "FAIL",
                    "reason": "the coupling surface could not be called: %s" % exc}
        difference = max(difference,
                         float((first.detach() - second.detach()).abs().max()))
    control_difference = None
    if control:
        changed = dict(arguments)
        value = arguments[control]
        changed[control] = value + 0.5 if not hasattr(value, "dtype") else value + 0.5
        try:
            third = function(**changed)
        except Exception as exc:  # noqa: BLE001
            return {"status": "FAIL",
                    "reason": "the control argument could not be applied: %s" % exc}
        control_difference = float((first.detach() - third.detach()).abs().max())
    moved = difference > 1e-6 and (control_difference is None or control_difference > 1e-6)
    return {"status": "PASS" if moved else "FAIL",
            "behaviour_argument": ", ".join(behaviour_arguments),
            "control_argument": control,
            "max_abs_difference": difference,
            "control_max_abs_difference": control_difference,
            "reason": ("the behaviour moves the computation" if moved else
                       "the behaviour argument does not change the computation, so it "
                       "cannot move the assignment")}


def check_tree(tree: Path, *, seed: int = 17) -> dict:
    """Read the tree's contract and compare it; an absent contract is recorded."""
    path = Path(tree) / CONTRACT_NAME
    if not path.exists():
        return {"status": "NOT_APPLICABLE",
                "reason": "the candidate tree declares no reference_contract.json",
                "contract": str(path)}
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return {"status": "FAIL", "reason": f"the contract is not readable JSON: {exc}",
                "contract": str(path)}
    verdict = compare(Path(tree), contract, seed=seed)
    verdict["contract"] = str(path)
    verdict["coupling"] = coupling_verdict(contract, seed=seed)
    return verdict


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args(argv)
    verdict = check_tree(Path(arguments.tree), seed=arguments.seed)
    print(json.dumps(verdict, default=str))
    return 0 if verdict.get("status") in ("OK", "NOT_APPLICABLE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
