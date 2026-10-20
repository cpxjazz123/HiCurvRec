"""Mechanism-specific independent checks, driven by a proposal's own declarations.

The forward oracle of the MVG compares an implementation against a *reference
implementation* -- the strongest check there is, and the wrong tool for a campaign: a
researcher-proposed mechanism has no reference implementation, and writing one per
mechanism is the research itself.

So the campaign checks what a proposal can declare and a checker can execute without
knowing the mechanism's internals:

* the **invariants** it claims (a permutation of the codebook leaves the loss unchanged;
  swapping the source and the target changes it; masking every pair gives zero);
* the **gradient targets** it claims (nonzero on the source residual, on the target
  residual, on the codebook);
* the **forbidden substitutions** it declares (a static audit of the code).

The mechanism's own code is never asked whether it is right. It is asked to behave.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class Check:
    """One executed check and what it found."""

    id: str
    ok: bool
    detail: str
    measurements: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ok": self.ok,
            "detail": self.detail,
            "measurements": self.measurements,
        }


def load_module(path: Path, name: str) -> types.ModuleType:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_case(inputs: Dict[str, Any], dtype: torch.dtype = torch.float64) -> Dict[str, torch.Tensor]:
    """The synthetic case a proposal declares, built deterministically from its seeds."""
    case: Dict[str, torch.Tensor] = {}
    for name in ("source", "target", "codebook"):
        spec = inputs.get(name) or {}
        shape = spec.get("shape") or [4, 4]
        seed = int(spec.get("seed", 1000 + len(case)))
        generator = torch.Generator().manual_seed(seed)
        case[name] = torch.randn(*[int(s) for s in shape], generator=generator, dtype=dtype)
    mask_spec = inputs.get("mask") or {}
    pairs = int(case["source"].shape[0])
    fraction = float(mask_spec.get("valid_fraction", 0.75))
    generator = torch.Generator().manual_seed(int(mask_spec.get("seed", 77)))
    keep = torch.rand(pairs, generator=generator) < fraction
    keep[0] = True  # a masked mean over nothing is not a case
    case["mask"] = keep.to(dtype)
    return case


def _loss(entry, case: Dict[str, torch.Tensor], parameters: Dict[str, float]) -> torch.Tensor:
    return entry(
        case["source"],
        case["target"],
        case["codebook"],
        case["mask"],
        **parameters,
    )


def _transform(name: str, case: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], str]:
    """Apply one declared transformation to the case."""
    out = dict(case)
    if name == "permute_codebook":
        entries = int(case["codebook"].shape[0])
        generator = torch.Generator().manual_seed(4242)
        permutation = torch.randperm(entries, generator=generator)
        out["codebook"] = case["codebook"][permutation]
        return out, "the codebook rows were permuted"
    if name == "swap_source_target":
        out["source"] = case["target"]
        out["target"] = case["source"]
        return out, "the source and the target were exchanged"
    if name == "mask_all":
        out["mask"] = torch.zeros_like(case["mask"])
        return out, "every pair was masked out"
    raise ValueError(f"unknown transformation {name!r}")


def check_invariants(
    entry,
    case: Dict[str, torch.Tensor],
    invariants: Sequence[Dict[str, Any]],
    parameters: Dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> List[Check]:
    """Run the invariants the proposal declares."""
    checks: List[Check] = []
    base = float(_loss(entry, case, parameters).detach())
    for item in invariants:
        name = str(item.get("transform", ""))
        expect = str(item.get("expect", "unchanged"))
        try:
            transformed, what = _transform(name, case)
        except ValueError as exc:
            checks.append(Check(str(item.get("id", name)), False, str(exc)))
            continue
        value = float(_loss(entry, transformed, parameters).detach())
        if expect == "unchanged":
            ok = abs(value - base) <= tolerance
            detail = (
                f"{what}: the loss changed by {abs(value - base):.3e} (tolerance {tolerance:.1e})"
                if not ok
                else f"{what}: the loss is unchanged"
            )
        elif expect == "changed":
            threshold = float(item.get("min_delta", tolerance))
            ok = abs(value - base) > threshold
            detail = (
                f"{what}: the loss changed by only {abs(value - base):.3e} (expected more than "
                f"{threshold:.1e})"
                if not ok
                else f"{what}: the loss changed by {abs(value - base):.3e}"
            )
        elif expect == "zero":
            ok = abs(value) <= tolerance
            detail = (
                f"{what}: the loss is {value:.3e}, not zero"
                if not ok
                else f"{what}: the loss is zero"
            )
        else:
            ok = False
            detail = f"unknown expectation {expect!r}"
        checks.append(
            Check(
                str(item.get("id", name)),
                ok,
                detail,
                {"base": base, "value": value, "expect": expect},
            )
        )
    return checks


def check_gradient_targets(
    entry,
    case: Dict[str, torch.Tensor],
    targets: Dict[str, str],
    parameters: Dict[str, float],
) -> List[Check]:
    """Check that the gradient reaches exactly the targets the proposal declares."""
    checks: List[Check] = []
    tensors = {
        "source": case["source"].detach().clone().requires_grad_(True),
        "target": case["target"].detach().clone().requires_grad_(True),
        "codebook": case["codebook"].detach().clone().requires_grad_(True),
    }
    differentiable = dict(case)
    differentiable.update(tensors)
    loss = _loss(entry, differentiable, parameters)
    loss.backward()
    for name, expectation in sorted(targets.items()):
        tensor = tensors.get(name)
        if tensor is None:
            checks.append(Check(f"gradient_{name}", False, f"unknown gradient target {name!r}"))
            continue
        grad = tensor.grad
        if grad is None:
            checks.append(
                Check(f"gradient_{name}", False, "no gradient reached the declared target")
            )
            continue
        _ = differentiable
        norm = float(grad.detach().abs().max())
        finite = bool(torch.isfinite(grad).all())
        if expectation == "nonzero":
            ok = finite and norm > 0.0
            detail = (
                f"the gradient on {name} is {norm:.3e}"
                + ("" if finite else " and is not finite")
            )
        elif expectation == "zero":
            ok = finite and norm == 0.0
            detail = f"the gradient on {name} is {norm:.3e} (declared zero)"
        else:  # "finite"
            ok = finite
            detail = f"the gradient on {name} is finite ({norm:.3e})"
        checks.append(
            Check(f"gradient_{name}", ok, detail, {"max_abs": norm, "finite": finite})
        )
    return checks


def check_forbidden_substitutions(
    source: str, forbidden: Sequence[Dict[str, Any]]
) -> List[Check]:
    """The static audit: the declared substitutions must not be in the code."""
    code = _code_only(source)
    checks: List[Check] = []
    for entry in forbidden:
        pattern = str(entry.get("pattern", ""))
        found = re.search(pattern, code) if pattern else None
        checks.append(
            Check(
                str(entry.get("id", pattern)),
                found is None,
                "the code does not contain it"
                if found is None
                else f"the code contains {found.group(0)!r}: {entry.get('why', '')}",
                {"pattern": pattern},
            )
        )
    return checks


def _code_only(source: str) -> str:
    """The source with comments and string literals removed: an audit of code, not prose."""
    import io
    import tokenize

    pieces: List[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    return "\n".join(pieces)


def diagnose_gradient_failure(
    entry,
    case: Dict[str, torch.Tensor],
    targets: Dict[str, str],
    parameters: Dict[str, float],
) -> Dict[str, Any]:
    """Why did a declared gradient target not receive a gradient?

    Three causes are separable by measurement, and the fourth by inspection of the call
    site:

    * **impossible specification** -- the loss does not depend on the target at all, so no
      implementation of this formula can reach it. Falsifiable: perturb the target and see
      whether the loss moves.
    * **missing gradient (wiring)** -- the loss depends on the target in the forward pass
      but the backward pass does not reach it: a detach, a stop-gradient, a ``no_grad``, or
      a call site that hands over a detached tensor.
    * **saturation** -- the backward pass reaches it but the local derivative is zero at
      this operating point (a ``relu``/``clamp`` boundary), while a finite step in the
      input does move the loss.

    The diagnosis is per target, and it is a measurement, not an opinion.
    """
    diagnosis: Dict[str, Any] = {}

    def loss_value(overrides: Dict[str, torch.Tensor]) -> float:
        probe = dict(case)
        probe.update(overrides)
        return float(_loss(entry, probe, parameters).detach())

    base = loss_value({})
    for name, expectation in sorted(targets.items()):
        if name not in ("source", "target", "codebook"):
            continue
        tensor = case[name]
        step = 1e-3
        plus = dict(case)
        plus[name] = tensor + step
        minus = dict(case)
        minus[name] = tensor - step
        forward_move = abs(loss_value(plus) - loss_value(minus))
        requires_grad = tensor.detach().clone().requires_grad_(True)
        differentiable = dict(case)
        differentiable[name] = requires_grad
        failure = ""
        try:
            value = _loss(entry, differentiable, parameters)
            if not value.requires_grad:
                # The loss has no graph at all: this input does not reach it, and the
                # forward measurement above already says whether anything does.
                raise RuntimeError("the loss carries no graph")
            grad = torch.autograd.grad(value, requires_grad, allow_unused=True)[0]
            local = 0.0 if grad is None else float(grad.abs().max())
        except Exception as exc:  # noqa: BLE001
            local = 0.0
            failure = f"{type(exc).__name__}: {exc}"

        if failure:
            if forward_move <= 1e-12:
                verdict = "impossible specification"
                reason = (
                    f"the loss does not depend on {name} at all (a finite step moves it by "
                    f"{forward_move:.3e}) and carries no graph through it, so no "
                    "implementation of this formula can deliver a gradient to it; the "
                    "declaration is wrong, not the code"
                )
            else:
                verdict = "missing gradient (wiring)"
                reason = (
                    f"the loss depends on {name} (a finite step moves it by "
                    f"{forward_move:.3e}) but the backward pass cannot reach it ({failure}): "
                    "a detach, a stop-gradient, a no_grad block, or a detached call site"
                )
            diagnosis[name] = {
                "expectation": expectation,
                "base_loss": base,
                "forward_dependence": forward_move,
                "local_gradient": 0.0,
                "verdict": verdict,
                "reason": reason,
            }
            continue
        if expectation != "nonzero":
            if local > 0.0:
                verdict = "declaration incorrect (measured nonzero)"
                reason = (
                    f"{name} was declared zero but carries a gradient of {local:.3e}; the "
                    "measurement contradicts the declaration, so the declaration is what has "
                    "to change"
                )
            elif forward_move > 1e-12:
                verdict = "undeclared influence"
                reason = (
                    f"{name} was declared zero but a finite step moves the loss by "
                    f"{forward_move:.3e}"
                )
            else:
                verdict = "matches the declaration"
            diagnosis[name] = {
                "expectation": expectation,
                "base_loss": base,
                "forward_dependence": forward_move,
                "local_gradient": local,
                "verdict": verdict,
                "reason": reason,
            }
            continue
        if forward_move <= 1e-12:
            verdict = "impossible specification"
            reason = (
                f"the loss does not depend on {name} at all (a finite step moves it by "
                f"{forward_move:.3e}), so no implementation of this formula can deliver a "
                "gradient to it; the declaration is wrong, not the code"
            )
        elif local <= 0.0:
            verdict = "missing gradient (wiring)"
            reason = (
                f"the loss depends on {name} (a finite step moves it by {forward_move:.3e}) "
                "but the backward pass does not reach it: a detach, a stop-gradient, a "
                "no_grad block, or a call site that passes a detached tensor"
            )
        elif forward_move > 1e-6 and local < 1e-9:
            verdict = "saturation"
            reason = f"{name} moves the loss finitely but its local derivative is {local:.3e}"
        else:
            verdict = "delivered"
            reason = f"the gradient reaches {name} ({local:.3e})"
        diagnosis[name] = {
            "expectation": expectation,
            "base_loss": base,
            "forward_dependence": forward_move,
            "local_gradient": local,
            "verdict": verdict,
            "reason": reason,
        }
    return diagnosis


def verify_mechanism(
    tree: Path,
    proposal: Dict[str, Any],
    *,
    module: str = "",
    entrypoint: str = "",
) -> Dict[str, Any]:
    """Run every check a proposal declares against the implementation in ``tree``."""
    tree = Path(tree)
    module_name = module or proposal.get("module", "")
    function_name = entrypoint or proposal.get("entrypoint", "")
    if not module_name or not function_name:
        return {
            "ok": False,
            "detail": "the proposal declares no module and entrypoint to check",
            "checks": [],
        }
    path = tree / f"{module_name}.py"
    if not path.is_file():
        return {"ok": False, "detail": f"no mechanism module at {path}", "checks": []}
    try:
        loaded = load_module(path, f"campaign_mechanism_{module_name}")
    except Exception as exc:
        return {
            "ok": False,
            "detail": f"the mechanism module does not import: {type(exc).__name__}: {exc}",
            "checks": [],
        }
    entry = getattr(loaded, function_name, None)
    if entry is None:
        return {
            "ok": False,
            "detail": f"the mechanism module defines no {function_name!r}",
            "checks": [],
        }
    parameters = {
        str(k): float(v)
        for k, v in (proposal.get("parameters") or {}).items()
        if isinstance(v, (int, float))
    }
    inputs = proposal.get("inputs") or {}
    case = build_case(inputs)
    checks: List[Check] = []
    checks += check_invariants(
        entry, case, proposal.get("invariants") or [], parameters,
        tolerance=float(proposal.get("tolerance", 1e-6)),
    )
    targets_declared = proposal.get("gradient_targets") or {}
    checks += check_gradient_targets(entry, case, targets_declared, parameters)
    checks += check_forbidden_substitutions(
        path.read_text(encoding="utf-8"), proposal.get("forbidden_substitutions") or []
    )
    failed = [check for check in checks if not check.ok]
    report = {
        "ok": not failed,
        "detail": (
            f"all {len(checks)} declared checks hold"
            if not failed
            else f"{len(failed)} of {len(checks)} declared checks failed"
        ),
        "checks": [check.as_dict() for check in checks],
        "failed": [check.id for check in failed],
    }
    if any(check.id.startswith("gradient_") and not check.ok for check in checks):
        report["diagnosis"] = diagnose_gradient_failure(entry, case, targets_declared, parameters)
    return report


__all__ = [
    "Check",
    "build_case",
    "check_forbidden_substitutions",
    "check_gradient_targets",
    "check_invariants",
    "load_module",
    "verify_mechanism",
]
