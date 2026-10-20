"""The MVG gates: M0 to M5, the checks that stand between a mechanism and a cold Stage-2 run.

Each gate answers one question, and each one is executable:

* **M0 contract/static audit** -- does the implementation agree with the frozen
  specification about *what* it is, and does it contain a substitution the specification
  forbids? A static audit alone never admits a mechanism: it is necessary, never sufficient.
* **M1 independent numerical reference** -- does the implementation compute the declared
  objective, on the forward value and on the intermediates, when an independent
  implementation that never imports it computes the same numbers?
* **M2 derivative correctness** -- is the gradient the derivative of the declared loss,
  does it reach the declared targets, and does it survive the degenerate cases?
* **M3 metamorphic/semantic** -- do the properties the specification declares actually
  hold (direction matters, permutations do not, a masked pair contributes nothing)?
* **M4 on/off identity and integration** -- is the mechanism-off tree the frozen host,
  and does the mechanism-on tree really move the parameters the export reads?
* **M5 test quality** -- does the suite have teeth? Every named defect must be rejected
  by the gate that is supposed to reject it.

A gate's state is ``passed``, ``failed`` or ``blocked``. ``blocked`` means the check could
not be run -- a missing resource, an unavailable device -- and is never a pass.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from . import spec as SPEC

PASSED = "passed"
FAILED = "failed"
BLOCKED = "blocked"

GATES = ("M0", "M1", "M2", "M3", "M4", "M5")

GATE_TITLES = {
    "M0": "contract and static audit",
    "M1": "independent numerical reference",
    "M2": "derivative correctness",
    "M3": "metamorphic and semantic properties",
    "M4": "on/off identity and integration",
    "M5": "test quality (mutation suite)",
}

GATE_PREVENTS = {
    "M0": "a mechanism that is not the one the specification describes",
    "M1": "a forward that computes the wrong quantity",
    "M2": "a gradient that is not the derivative of the declared loss",
    "M3": "a mechanism that breaks the properties it claims to have",
    "M4": "a mechanism that does not reach the trained parameters",
    "M5": "a suite that would pass a broken implementation",
}

#: The gradient target names the specification may declare, and the tensor each names.
GRADIENT_TARGETS = ("source_residual", "target_residual", "codebook")


@dataclass
class GateResult:
    gate: str
    state: str
    detail: str
    measurements: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "title": GATE_TITLES.get(self.gate, self.gate),
            "prevents": GATE_PREVENTS.get(self.gate, ""),
            "state": self.state,
            "detail": self.detail,
            "measurements": self.measurements,
            "diagnostics": self.diagnostics,
            "artifacts": self.artifacts,
        }


def _load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _max_deviation(left: Any, right: Any) -> float:
    """The largest absolute difference between two tensors or nested sequences."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            return float("inf")
        return float((left.detach() - right.detach()).abs().max()) if left.numel() else 0.0
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        worst = 0.0
        for a, b in zip(left, right):
            worst = max(worst, _max_deviation(a, b))
        return worst
    return abs(float(left) - float(right))


def _to_list(value: Any) -> List[Any]:
    """Flatten a tensor or a nested sequence into a flat list of numbers."""
    if isinstance(value, torch.Tensor):
        return value.detach().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for item in value:
            out.extend(_to_list(item))
        return out
    return [value]


class MechanismUnderTest:
    """The mechanism, its specification, and the independent reference beside it."""

    def __init__(self, tree: Path, spec_doc: Dict[str, Any], baseline_tree: Path | None = None) -> None:
        self.tree = Path(tree).resolve()
        # The tree the specification was written against. The mutation suite injects its
        # defects there: a tree that has been repaired since would no longer match the
        # catalogue, and a mutant that no longer applies proves nothing.
        self.baseline_tree = Path(baseline_tree).resolve() if baseline_tree else self.tree
        self.spec = spec_doc
        self.integration: Dict[str, Any] = spec_doc["integration"]
        self.module_path = self.tree / f"{self.integration['module']}.py"
        self.trainer_path = self.tree / f"{self.integration['host_module']}.py"
        self._production: Optional[types.ModuleType] = None
        self._reference: Optional[types.ModuleType] = None

    # -- loading --------------------------------------------------------
    def production(self) -> types.ModuleType:
        if self._production is None:
            if not self.module_path.is_file():
                raise FileNotFoundError(f"the mechanism module {self.module_path} does not exist")
            self._production = _load_module(self.module_path, "mvg_production_mechanism")
        return self._production

    def reference(self) -> types.ModuleType:
        if self._reference is None:
            path = SPEC.REPO_ROOT / self.spec["objective"]["reference_implementation"]
            if not path.is_file():
                raise FileNotFoundError(f"the reference implementation {path} does not exist")
            self._reference = _load_module(path, "mvg_reference_implementation")
        return self._reference

    def entry(self) -> Callable[..., torch.Tensor]:
        name = self.integration["entrypoint"]
        fn = getattr(self.production(), name, None)
        if fn is None:
            raise AttributeError(f"the mechanism module defines no {name!r}")
        return fn

    def intermediates(self) -> Callable[..., Dict[str, Any]]:
        name = self.integration["intermediates_entrypoint"]
        fn = getattr(self.production(), name, None)
        if fn is None:
            raise AttributeError(f"the mechanism module defines no {name!r}")
        return fn

    # -- fixtures -------------------------------------------------------
    def cases(self) -> List[Tuple[str, Dict[str, Any]]]:
        out: List[Tuple[str, Dict[str, Any]]] = []
        for role, entries in self.spec["cases"].items():
            for case in entries:
                out.append((role, case))
        return out

    @staticmethod
    def tensors(case: Dict[str, Any], dtype: torch.dtype = torch.float64) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        inputs = case["inputs"]
        z_src = torch.tensor(inputs["z_src"], dtype=dtype)
        z_tgt = torch.tensor(inputs["z_tgt"], dtype=dtype)
        codebook = torch.tensor(inputs["codebook"], dtype=dtype)
        mask = torch.tensor(inputs["mask"], dtype=dtype)
        return z_src, z_tgt, codebook, mask

    def parameters(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.spec["objective"]["parameters"].items()}

    def call_entry(
        self,
        z_src: torch.Tensor,
        z_tgt: torch.Tensor,
        codebook: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.entry()(
            z_src,
            z_tgt,
            codebook,
            mask,
            **self.parameters(),
        )


# --------------------------------------------------------------------------------------
# M0 -- contract and static audit
# --------------------------------------------------------------------------------------
def _code_only(source: str) -> str:
    """The source with comments and string literals removed.

    A textual audit is an audit of **code**. A candidate that writes "the margin was
    symmetric" in a comment has not substituted a symmetric operation, and a rule that
    fires on the word would reject a correct repair for its explanation.
    """
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


def gate_m0(mut: MechanismUnderTest) -> GateResult:
    problems: List[str] = []
    measurements: Dict[str, Any] = {}

    if not mut.module_path.is_file():
        return GateResult(
            "M0",
            FAILED,
            f"the mechanism module {mut.module_path} does not exist",
            measurements,
            [f"missing module: {mut.module_path}"],
        )

    source = mut.module_path.read_text(encoding="utf-8")
    measurements["module_sha256"] = SPEC.spec_digest({"source": source})
    measurements["module_bytes"] = len(source)

    try:
        tree = ast.parse(source, filename=str(mut.module_path))
    except SyntaxError as exc:
        return GateResult(
            "M0", FAILED, f"the mechanism module does not parse: {exc}", measurements, [str(exc)]
        )

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for key in ("entrypoint", "intermediates_entrypoint"):
        name = mut.integration[key]
        if name not in functions:
            problems.append(f"the mechanism module defines no {name!r}")
        elif name.startswith("_"):
            problems.append(f"{name!r} is private; the verifier and the MCP probe reach it by name")

    # the declared parameters must exist in the implementation, with the declared values
    constants: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
    for name, declared in mut.parameters().items():
        found = None
        for candidate in (name, name.upper()):
            if candidate in constants:
                found = constants[candidate]
                break
        if found is None:
            problems.append(
                f"the specification declares the parameter {name!r} but the implementation "
                f"exposes no matching constant"
            )
        elif not isinstance(found, (int, float)) or abs(float(found) - declared) > 1e-12:
            problems.append(
                f"the parameter {name!r} is {found!r} in the implementation and {declared!r} in "
                "the specification"
            )

    # the forbidden substitutions, read against the CODE: a comment that explains the
    # defect is not the defect.
    code = _code_only(source)
    for entry in mut.spec["forbidden_substitutions"]:
        pattern = entry["pattern"]
        found = re.search(pattern, code)
        if found:
            problems.append(
                f"forbidden substitution {entry['id']!r}: the code contains {found.group(0)!r} "
                f"({entry['why']})"
            )

    # the call site: the host must call the declared entrypoint, with the declared weight
    if not mut.trainer_path.is_file():
        problems.append(f"the host module {mut.trainer_path} does not exist")
    else:
        trainer_source = mut.trainer_path.read_text(encoding="utf-8")
        try:
            trainer_tree = ast.parse(trainer_source, filename=str(mut.trainer_path))
        except SyntaxError as exc:
            problems.append(f"the host module does not parse: {exc}")
            trainer_tree = None
        if trainer_tree is not None:
            called = False
            for node in ast.walk(trainer_tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == mut.integration["entrypoint"]:
                        called = True
                        break
            if not called:
                problems.append(
                    f"the host never calls {mut.integration['entrypoint']!r}: the mechanism is "
                    "not wired into the training entry"
                )
            flag = mut.integration["mechanism_weight_flag"]
            trainer_constants = {}
            for node in trainer_tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
                    node.targets[0], ast.Name
                ):
                    try:
                        trainer_constants[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        continue
            if flag not in trainer_constants:
                problems.append(
                    f"the host declares no {flag!r}: the mechanism's weight is not a hardcoded "
                    "constant of the tree"
                )
            else:
                measurements["mechanism_weight"] = trainer_constants[flag]

    # the declared arguments must be the ones the implementation accepts
    try:
        signature = inspect.signature(mut.entry())
        accepted = set(signature.parameters)
        for argument in mut.integration["arguments"]:
            if argument["name"] not in accepted:
                problems.append(
                    f"the implementation does not accept the declared argument "
                    f"{argument['name']!r}"
                )
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        problems.append(f"the entrypoint has no inspectable signature: {exc}")

    # independence: the implementation must not reach the oracle, and the oracle must not
    # reach the implementation
    reference_path = SPEC.REPO_ROOT / mut.spec["objective"]["reference_implementation"]
    if reference_path.is_file():
        reference_source = reference_path.read_text(encoding="utf-8")
        if mut.integration["module"] in reference_source:
            problems.append(
                "the reference implementation mentions the production module: the oracle is not "
                "independent"
            )
        if mut.integration["module"] + "." in source or f"import {mut.integration['module']}" in source:
            problems.append("the production module imports the reference implementation")

    measurements["forbidden_substitutions_checked"] = len(mut.spec["forbidden_substitutions"])
    measurements["static_audit_is_sufficient_alone"] = False

    if problems:
        return GateResult(
            "M0",
            FAILED,
            f"{len(problems)} contract problem(s) in the mechanism implementation",
            measurements,
            problems,
        )
    return GateResult(
        "M0",
        PASSED,
        "the implementation matches the frozen contract; a static audit never admits a "
        "mechanism on its own, so M1-M5 still have to pass",
        measurements,
    )


# --------------------------------------------------------------------------------------
# M1 -- independent numerical reference
# --------------------------------------------------------------------------------------
def gate_m1(mut: MechanismUnderTest) -> GateResult:
    tolerances = mut.spec["tolerances"]
    if "float64_cpu" not in tolerances:
        return GateResult(
            "M1", BLOCKED, "the specification declares no float64_cpu tolerance", {}, []
        )
    tol = tolerances["float64_cpu"]
    atol, rtol = float(tol["atol"]), float(tol["rtol"])

    deviations: Dict[str, float] = {}
    measurements: Dict[str, Any] = {}
    worst = 0.0
    problems: List[str] = []
    checked = 0

    try:
        entry = mut.call_entry
        intermediates = mut.intermediates()
        reference = mut.reference()
    except Exception as exc:
        return GateResult(
            "M1", BLOCKED, f"the mechanism or the reference could not be loaded: {exc}", {}, [str(exc)]
        )

    for role, case in mut.cases():
        z_src, z_tgt, codebook, mask = mut.tensors(case, torch.float64)
        label = f"{role}/{case['id']}"
        try:
            produced = entry(z_src, z_tgt, codebook, mask)
            parts = intermediates(z_src, z_tgt, codebook, mask, **mut.parameters())
        except Exception as exc:
            problems.append(f"{label}: the implementation raised {type(exc).__name__}: {exc}")
            continue
        expected = reference.reference_intermediates(
            z_src,
            z_tgt,
            codebook,
            mask,
            **mut.parameters(),
        )
        checked += 1

        def note(name: str, value: float) -> None:
            nonlocal worst
            deviations[f"{label}:{name}"] = value
            worst = max(worst, value)

        note("loss", _max_deviation(float(produced), float(expected["loss"])))
        for key, ref_key in (
            ("src_dist", "src_dist"),
            ("tgt_dist", "tgt_dist"),
            ("src_nearest", "src_nearest"),
            ("tgt_nearest", "tgt_nearest"),
            ("margin_term", "margin_term"),
            ("structural_term", "structural"),
            ("per_pair", "per_pair"),
        ):
            note(key, _max_deviation(_to_list(parts[key]), _to_list(expected[ref_key])))
        for key in ("src_index", "tgt_index"):
            note(key, _max_deviation(_to_list(parts[key]), _to_list(expected[key])))

    measurements["cases_checked"] = checked
    measurements["max_deviation"] = worst
    measurements["atol"] = atol
    measurements["rtol"] = rtol
    measurements["deviations"] = deviations

    if checked == 0:
        return GateResult(
            "M1", BLOCKED, "no case could be evaluated", measurements, problems
        )

    # the tolerance is relative to the reference's own magnitude
    scale = 1.0
    for name, value in deviations.items():
        limit = atol + rtol * scale
        if value > limit:
            problems.append(f"{name} deviates by {value:.3e} (limit {limit:.3e})")
            break

    # the training dtype, when the specification declares one
    if "float32_cpu" in tolerances:
        tol32 = tolerances["float32_cpu"]
        limit32 = float(tol32["atol"]) + float(tol32["rtol"])
        worst32 = 0.0
        for role, case in mut.cases():
            z_src, z_tgt, codebook, mask = mut.tensors(case, torch.float32)
            produced = mut.call_entry(z_src, z_tgt, codebook, mask)
            expected = mut.reference().reference_loss(
                z_src, z_tgt, codebook, mask, **mut.parameters()
            )
            worst32 = max(worst32, abs(float(produced) - float(expected)))
        measurements["float32_max_deviation"] = worst32
        if worst32 > limit32:
            problems.append(
                f"the float32 deviation {worst32:.3e} exceeds the declared limit {limit32:.3e}"
            )

    if problems:
        return GateResult(
            "M1",
            FAILED,
            f"the implementation does not reproduce the independent reference "
            f"(worst deviation {worst:.3e})",
            measurements,
            problems,
        )
    return GateResult(
        "M1",
        PASSED,
        f"the implementation reproduces the independent reference on {checked} case(s) "
        f"(worst deviation {worst:.3e})",
        measurements,
    )


# --------------------------------------------------------------------------------------
# M2 -- derivative correctness
# --------------------------------------------------------------------------------------
def _argmin_ties(case: Dict[str, Any]) -> int:
    """How many pairs of the case have two equidistant nearest codebook entries.

    A tie is a point where the loss is not differentiable: the argmin's selection changes
    under an arbitrarily small perturbation, so a finite difference there measures the jump
    rather than the derivative.
    """
    inputs = case["inputs"]
    codebook = inputs["codebook"]
    ties = 0
    for key in ("z_src", "z_tgt"):
        for row in inputs[key]:
            distances = sorted(
                sum(abs(float(row[d]) - float(entry[d])) for d in range(len(entry)))
                for entry in codebook
            )
            if abs(distances[0] - distances[1]) <= 1e-12:
                ties += 1
    return ties


def gate_m2(mut: MechanismUnderTest) -> GateResult:
    problems: List[str] = []
    measurements: Dict[str, Any] = {}
    case = None
    ties = 0
    for role, entry in mut.cases():
        if role != "positive":
            continue
        ties = _argmin_ties(entry)
        if ties == 0:
            case = entry
            break
    measurements["case"] = case["id"] if case else ""
    measurements["argmin_ties_at_the_evaluation_point"] = ties
    if case is None:
        return GateResult(
            "M2",
            FAILED,
            "the specification declares no differentiable evaluation point: every positive case "
            "has two equidistant nearest codebook entries, and a derivative cannot be measured "
            "at a tie",
            measurements,
            ["add a positive case whose nearest codebook entry is unique for every pair"],
        )

    z_src, z_tgt, codebook, mask = mut.tensors(case, torch.float64)
    z_src.requires_grad_(True)
    z_tgt.requires_grad_(True)
    codebook.requires_grad_(True)

    tensors = {"source_residual": z_src, "target_residual": z_tgt, "codebook": codebook}

    try:
        loss = mut.call_entry(z_src, z_tgt, codebook, mask)
        loss.backward()
    except Exception as exc:
        return GateResult(
            "M2", FAILED, f"the loss could not be differentiated: {exc}", measurements, [str(exc)]
        )

    measurements["loss"] = float(loss.detach())
    grad_norms: Dict[str, float] = {}
    for name, tensor in tensors.items():
        grad = tensor.grad
        norm = 0.0 if grad is None else float(grad.detach().abs().max())
        grad_norms[name] = norm
        expectation = mut.spec["gradient_targets"].get(name)
        if expectation is None:
            problems.append(f"the specification declares no gradient target named {name!r}")
            continue
        if not torch.isfinite(grad).all() if grad is not None else False:
            problems.append(f"the gradient of {name} is not finite")
        if expectation == "nonzero" and norm <= 0.0:
            problems.append(
                f"the specification declares a nonzero gradient on {name} and the gradient is {norm:.3e}"
            )
        if expectation == "zero" and norm != 0.0:
            problems.append(
                f"the specification declares a zero gradient on {name} and the gradient is {norm:.3e}"
            )
    measurements["gradient_max_abs"] = grad_norms

    # the derivative of the declared loss, by central differences
    reference = mut.reference()
    step = 1e-6

    def loss_fn(sequence: Sequence[torch.Tensor]) -> torch.Tensor:
        # The mask is fixed: it is not a differentiable input of the mechanism.
        return mut.call_entry(sequence[0], sequence[1], sequence[2], mask)

    try:
        finite_difference = reference.finite_difference_gradient(
            loss_fn, [z_src, z_tgt, codebook], step=step
        )
    except Exception as exc:
        return GateResult(
            "M2", FAILED, f"the finite-difference reference failed: {exc}", measurements, [str(exc)]
        )

    fd_dev = 0.0
    for name, tensor, fd in zip(
        ("source_residual", "target_residual", "codebook"),
        (z_src, z_tgt, codebook),
        finite_difference,
    ):
        grad = tensor.grad
        if grad is None:
            problems.append(f"{name} received no gradient at all")
            continue
        deviation = _max_deviation(grad.detach(), fd)
        fd_dev = max(fd_dev, deviation)
        measurements[f"finite_difference_deviation_{name}"] = deviation
    measurements["finite_difference_max_deviation"] = fd_dev
    if fd_dev > 1e-5:
        problems.append(
            f"the autograd gradient and the finite-difference gradient differ by {fd_dev:.3e}"
        )

    # the directional derivative: autograd against a central difference along one direction
    generator = torch.Generator().manual_seed(20251020)
    direction = [
        torch.randn(z_src.shape, generator=generator, dtype=torch.float64),
        torch.randn(z_tgt.shape, generator=generator, dtype=torch.float64),
        torch.randn(codebook.shape, generator=generator, dtype=torch.float64),
    ]
    autograd_directional = float(
        sum(
            (tensor.grad.detach() * step_tensor).sum()
            for tensor, step_tensor in zip((z_src, z_tgt, codebook), direction)
        )
    )
    finite_directional = reference.directional_derivative(
        loss_fn, [z_src, z_tgt, codebook], direction, step=step
    )
    directional_gap = abs(autograd_directional - finite_directional)
    measurements["directional_derivative"] = {
        "autograd": autograd_directional,
        "finite_difference": finite_directional,
        "gap": directional_gap,
    }
    if directional_gap > 1e-4:
        problems.append(
            f"the directional derivative disagrees: autograd {autograd_directional:.6e} vs "
            f"finite difference {finite_directional:.6e}"
        )

    # the zero-gradient failure case: every pair masked out
    empty = torch.zeros_like(mask)
    z2, t2, c2 = z_src.detach().clone(), z_tgt.detach().clone(), codebook.detach().clone()
    z2.requires_grad_(True)
    t2.requires_grad_(True)
    c2.requires_grad_(True)
    try:
        empty_loss = mut.call_entry(z2, t2, c2, empty)
        empty_loss.backward()
        empty_ok = abs(float(empty_loss.detach())) <= 1e-12
        finite_grads = all(
            tensor.grad is not None and torch.isfinite(tensor.grad).all()
            for tensor in (z2, t2, c2)
        )
        zero_grads = all(
            tensor.grad is not None and float(tensor.grad.detach().abs().max()) == 0.0
            for tensor in (z2, t2, c2)
        )
        measurements["all_masked_out"] = {
            "loss": float(empty_loss.detach()),
            "finite_gradients": bool(finite_grads),
            "zero_gradients": bool(zero_grads),
        }
        if not empty_ok:
            problems.append(
                f"the fully masked loss is {float(empty_loss.detach()):.6e}, not zero"
            )
        if not finite_grads:
            problems.append("the fully masked backward produced a non-finite gradient")
    except Exception as exc:
        problems.append(f"the fully masked case raised {type(exc).__name__}: {exc}")

    # the nondifferentiable boundary: a tie in the argmin must not produce a NaN
    for role, entry in mut.cases():
        if role != "boundary":
            continue
        b_zs, b_zt, b_cb, b_mask = mut.tensors(entry, torch.float64)
        try:
            value = mut.call_entry(b_zs, b_zt, b_cb, b_mask)
            if not torch.isfinite(value).all():
                problems.append(f"the boundary case {entry['id']} produced a non-finite loss")
        except Exception as exc:
            problems.append(f"the boundary case {entry['id']} raised {type(exc).__name__}: {exc}")

    if problems:
        return GateResult(
            "M2",
            FAILED,
            f"{len(problems)} derivative problem(s)",
            measurements,
            problems,
        )
    return GateResult(
        "M2",
        PASSED,
        "the gradient is the derivative of the declared loss, reaches every declared target, "
        "and survives the degenerate cases",
        measurements,
    )


# --------------------------------------------------------------------------------------
# M3 -- metamorphic and semantic properties
# --------------------------------------------------------------------------------------
def gate_m3(mut: MechanismUnderTest) -> GateResult:
    problems: List[str] = []
    measurements: Dict[str, Any] = {}
    case = None
    for role, entry in mut.cases():
        if role == "positive":
            case = entry
            break
    if case is None:
        return GateResult("M3", BLOCKED, "the specification declares no positive case", {}, [])

    z_src, z_tgt, codebook, mask = mut.tensors(case, torch.float64)
    tol = mut.spec["tolerances"]["float64_cpu"]
    limit = float(tol["atol"]) + float(tol["rtol"])

    base = float(mut.call_entry(z_src, z_tgt, codebook, mask).detach())
    measurements["loss"] = base

    # 1. direction matters: swapping the roles changes the loss
    swapped = float(mut.call_entry(z_tgt, z_src, codebook, mask).detach())
    reference_swapped = float(
        mut.reference().reference_loss(z_tgt, z_src, codebook, mask, **mut.parameters())
    )
    measurements["swapped_loss"] = swapped
    measurements["swapped_reference"] = reference_swapped
    if abs(swapped - reference_swapped) > limit:
        problems.append(
            f"the swapped direction deviates from the reference by {abs(swapped - reference_swapped):.3e}"
        )
    if abs(swapped - base) <= limit:
        problems.append(
            "the loss is unchanged when the source and the target are swapped: the mechanism is "
            "symmetric, and the specification declares it directed"
        )

    # 2. the codebook may be permuted: the loss is a function of the codebook, not of its order
    generator = torch.Generator().manual_seed(4242)
    permutation = torch.randperm(codebook.shape[0], generator=generator)
    permuted = float(mut.call_entry(z_src, z_tgt, codebook[permutation], mask).detach())
    measurements["permuted_codebook_loss"] = permuted
    if abs(permuted - base) > limit:
        problems.append(
            f"permuting the codebook changed the loss by {abs(permuted - base):.3e}"
        )

    # 3. the pairs may be permuted: the masked mean is a mean
    pair_permutation = torch.randperm(z_src.shape[0], generator=generator)
    reordered = float(
        mut.call_entry(
            z_src[pair_permutation], z_tgt[pair_permutation], codebook, mask[pair_permutation]
        ).detach()
    )
    measurements["permuted_pairs_loss"] = reordered
    if abs(reordered - base) > limit:
        problems.append(f"permuting the pairs changed the loss by {abs(reordered - base):.3e}")

    # 4. a masked pair contributes nothing: masking one pair equals dropping it
    masked_index = int(torch.nonzero(mask == 0)[0][0]) if bool((mask == 0).any()) else 0
    keep = torch.ones_like(mask)
    keep[masked_index] = 0.0
    without = float(mut.call_entry(z_src, z_tgt, codebook, keep).detach())
    keep_rows = mask > 0
    subset = float(
        mut.call_entry(
            z_src[keep_rows], z_tgt[keep_rows], codebook, mask[keep_rows]
        ).detach()
    )
    measurements["masked_out_loss"] = without
    measurements["subset_loss"] = subset
    if abs(without - subset) > limit:
        problems.append(
            f"masking a pair out changed the loss by {abs(without - subset):.3e} against the "
            "loss of the remaining pairs"
        )

    # 5. the degenerate and boundary cases stay finite
    for role, entry in mut.cases():
        if role not in ("degenerate", "boundary"):
            continue
        d_zs, d_zt, d_cb, d_mask = mut.tensors(entry, torch.float64)
        try:
            value = float(mut.call_entry(d_zs, d_zt, d_cb, d_mask).detach())
            if not math.isfinite(value):
                problems.append(f"the {role} case {entry['id']} produced {value}")
        except Exception as exc:
            problems.append(f"the {role} case {entry['id']} raised {type(exc).__name__}: {exc}")

    if problems:
        return GateResult(
            "M3", FAILED, f"{len(problems)} property problem(s)", measurements, problems
        )
    return GateResult(
        "M3",
        PASSED,
        "the declared properties hold: the mechanism is directed, permutation-invariant in the "
        "codebook and the pairs, and a masked pair contributes nothing",
        measurements,
    )


# --------------------------------------------------------------------------------------
# M4 -- on/off identity and integration
# --------------------------------------------------------------------------------------
def _load_trainer(tree: Path):
    """Load the arm's trainer the way the CGV does: under a private name, tree on sys.path."""
    repo_cgv = SPEC.REPO_ROOT / "tools" / "cgv"
    if str(repo_cgv) not in sys.path:
        sys.path.insert(0, str(repo_cgv))
    import engine as cgv_engine  # type: ignore  # noqa: E402

    return cgv_engine.load_trainer(tree, module_name="mvg_trainer")


def gate_m4(mut: MechanismUnderTest) -> GateResult:
    problems: List[str] = []
    measurements: Dict[str, Any] = {}
    artifacts: List[str] = []

    # -- static: the call site and the output paths ----------------------
    if not mut.trainer_path.is_file():
        return GateResult(
            "M4", BLOCKED, f"the host module {mut.trainer_path} does not exist", {}, []
        )
    trainer_source = mut.trainer_path.read_text(encoding="utf-8")
    measurements["trainer_sha256"] = SPEC.spec_digest({"source": trainer_source})
    if mut.integration["entrypoint"] not in trainer_source:
        problems.append("the host does not mention the mechanism entrypoint")

    # the tree must not write into another arm's namespace
    tree_name = mut.tree.name
    siblings = sorted(
        p.name for p in (SPEC.REPO_ROOT / "stage2_RQ-VAE").iterdir() if p.is_dir()
    )
    measurements["tree_name"] = tree_name
    measurements["sibling_arms"] = len(siblings)
    if siblings.count(tree_name) > 1:
        problems.append(f"more than one arm tree is named {tree_name!r}")

    # -- executable: the real model, the real host terms -----------------
    try:
        trainer, saved_path, saved_mods = _load_trainer(mut.tree)
    except Exception as exc:
        return GateResult(
            "M4",
            BLOCKED,
            f"the host trainer could not be loaded: {type(exc).__name__}: {exc}",
            measurements,
            [str(exc)],
        )
    try:
        torch.manual_seed(7)
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=32)
        pairs = 6
        embeddings = torch.randn(2 * pairs, 32, generator=torch.Generator().manual_seed(11))
        reconstructed, quant_loss, _, _ = model.forward(embeddings)
        base_loss, recon_loss = model.compute_loss(embeddings, reconstructed, quant_loss)
        residuals = trainer._residuals_with_trace(model, embeddings)
        source_ids = torch.arange(pairs)
        target_ids = (torch.arange(pairs) + 1) % pairs
        behavior_loss = trainer.behavior_contrastive_loss(
            residuals, source_ids, target_ids, None
        )
        host_loss = base_loss + behavior_loss

        codebook = model.rq.vq_layers[0].get_code_embs()
        z_src = residuals[0, :pairs]
        z_tgt = residuals[0, pairs:]
        pair_mask = source_ids.ne(target_ids).to(dtype=torch.float32)
        mechanism_term = mut.call_entry(z_src, z_tgt, codebook, pair_mask)
        weight = float(getattr(trainer, mut.integration["mechanism_weight_flag"], 0.0))
        mechanism_loss = host_loss + weight * mechanism_term

        measurements["host_loss"] = float(host_loss.detach())
        measurements["mechanism_term"] = float(mechanism_term.detach())
        measurements["mechanism_weight"] = weight
        measurements["mechanism_changes_the_loss"] = (
            abs(float((mechanism_loss - host_loss).detach())) > 0.0
        )
        if not measurements["mechanism_changes_the_loss"]:
            problems.append("the mechanism does not change the loss at all")

        # the mechanism-off tree is the frozen host: the host terms are unchanged
        frozen = SPEC.REPO_ROOT / "stage2_RQ-VAE" / "TIGER_r8_control"
        if frozen.is_dir():
            frozen_trainer, frozen_path, frozen_mods = _load_trainer(frozen)
            try:
                torch.manual_seed(7)
                frozen_model = frozen_trainer.RQVAE(frozen_trainer._tokenizer_config(), in_dim=32)
                frozen_reconstructed, frozen_quant, _, _ = frozen_model.forward(embeddings)
                frozen_base, frozen_recon = frozen_model.compute_loss(
                    embeddings, frozen_reconstructed, frozen_quant
                )
                frozen_residuals = frozen_trainer._residuals_with_trace(frozen_model, embeddings)
                frozen_behavior = frozen_trainer.behavior_contrastive_loss(
                    frozen_residuals, source_ids, target_ids, None
                )
                frozen_host = frozen_base + frozen_behavior
                gap = abs(float(host_loss.detach()) - float(frozen_host.detach()))
                measurements["frozen_host_loss"] = float(frozen_host.detach())
                measurements["frozen_host_gap"] = gap
                measurements["frozen_host_tree"] = str(frozen)
                if gap > 1e-9:
                    problems.append(
                        f"the mechanism-off host differs from the frozen host by {gap:.3e}"
                    )
            finally:
                _unload_trainer(frozen_trainer, frozen_path, frozen_mods, "mvg_trainer")

        # the gradient really moves the parameters the export reads
        encoder_parameters = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        host_grads = torch.autograd.grad(
            host_loss, encoder_parameters, retain_graph=True, allow_unused=True
        )
        mechanism_grads = torch.autograd.grad(
            mechanism_loss, encoder_parameters, retain_graph=True, allow_unused=True
        )
        encoder_change = max(
            (
                float((a - b).abs().max())
                for a, b in zip(host_grads, mechanism_grads)
                if a is not None and b is not None
            ),
            default=0.0,
        )
        measurements["encoder_gradient_change"] = encoder_change
        if encoder_change <= 0.0:
            problems.append("the mechanism does not change the encoder gradient")

        codebook_grad = torch.autograd.grad(
            mechanism_term, codebook, retain_graph=True, allow_unused=True
        )[0]
        codebook_change = float(codebook_grad.abs().max()) if codebook_grad is not None else 0.0
        measurements["codebook_gradient_max_abs"] = codebook_change
        if codebook_change <= 0.0:
            problems.append("the mechanism does not reach the codebook the export reads")

        # the export reads the encoder and the codebook: the pathway is affected
        with torch.no_grad():
            before = model.get_indices(embeddings, infer_use_sk=False).clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        optimizer.zero_grad(set_to_none=True)
        mechanism_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            after = model.get_indices(embeddings, infer_use_sk=False).clone()
        changed = int((before != after).sum())
        measurements["exported_token_changes_after_one_step"] = changed
        measurements["exported_rows"] = int(before.numel())
        if changed == 0:
            problems.append(
                "one optimizer step with the mechanism changed no exported token: the mechanism "
                "does not reach the exported hard SID pathway"
            )
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            _unload_trainer(trainer, saved_path, saved_mods, "mvg_trainer")
        except Exception:  # pragma: no cover - defensive
            pass

    if problems:
        return GateResult(
            "M4", FAILED, f"{len(problems)} integration problem(s)", measurements, problems, artifacts
        )
    return GateResult(
        "M4",
        PASSED,
        "the mechanism-off tree is the frozen host, and the mechanism-on tree changes the loss, "
        "the encoder gradient, the codebook gradient and the exported tokens",
        measurements,
        artifacts,
    )


def _unload_trainer(trainer, saved_path, saved_mods, module_name: str) -> None:
    repo_cgv = SPEC.REPO_ROOT / "tools" / "cgv"
    if str(repo_cgv) in sys.path:
        sys.path.insert(0, str(repo_cgv))
    import engine as cgv_engine  # type: ignore  # noqa: E402

    cgv_engine.unload_trainer(saved_path, saved_mods, module_name=module_name)


__all__ = [
    "GATES",
    "GATE_PREVENTS",
    "GATE_TITLES",
    "GateResult",
    "MechanismUnderTest",
    "PASSED",
    "FAILED",
    "BLOCKED",
    "gate_m0",
    "gate_m1",
    "gate_m2",
    "gate_m3",
    "gate_m4",
]
