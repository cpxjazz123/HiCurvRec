"""Mechanism test contracts: what would prove THIS mechanism failed, decided early.

A research agent that proposes a mechanism must also say what would make it inert,
because that answer is the test. The contract is data: it names templates from this
library and the callables of the candidate they apply to. The templates are written
once; a new mechanism usually fills parameters instead of writing a new verifier.

Every template returns ``{"status": "PASS" | "FAIL" | "NOT_RUN", "detail", ...}``.
A template may import the candidate, but it must never use the candidate to compute
its own expectation: an independent formula compares against the canonical module,
a sensitivity check compares two runs against each other, and a range check uses the
frozen training statistics.

``negative_control`` is the check on the check: the contract names a small source
edit that must make a named template FAIL. A template that still passes the broken
variant is measuring nothing.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

#: The templates, by name. A contract may only use these.
TEMPLATES = ("SignalSensitivity", "IndependentFormula", "RangeAndSaturation",
             "AblationEffect", "AssignmentConsistency", "GradientFlow", "DataBoundary")


def load_contract(path: Path) -> dict:
    """Read and validate one mechanism contract."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_contract(payload)


def validate_contract(payload: dict) -> dict:
    """Refuse a contract that names an unknown template or has no mechanism."""
    if not str(payload.get("mechanism") or "").strip():
        raise ValueError("a mechanism contract needs a mechanism name")
    tests = payload.get("tests") or []
    if not tests:
        raise ValueError("a mechanism contract needs at least one test")
    for entry in tests:
        template = str(entry.get("template") or "")
        if template not in TEMPLATES:
            raise ValueError("unknown template: " + template)
    control = payload.get("negative_control") or {}
    if control:
        if not str(control.get("edit") or "").strip():
            raise ValueError("a negative control needs the edit that breaks the mechanism")
        if str(control.get("must_fail") or "") not in TEMPLATES:
            raise ValueError("the negative control must name the template it breaks")
    return payload


# --- helpers ------------------------------------------------------------
def _import_candidate(tree: Path, module: str):
    """Import a module of the candidate, from the candidate's own root."""
    tree = Path(tree)
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    spec = importlib.util.spec_from_file_location(
        "cand_" + module.replace(".", "_"), tree / (module.replace(".", "/") + ".py"))
    if spec is None or spec.loader is None:
        raise ImportError("cannot import " + module + " from " + str(tree))
    module_obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    return module_obj


def _resolve(tree: Path, dotpath: str):
    """``module.attr`` or ``module.Class.attr`` inside the candidate."""
    parts = str(dotpath).split(".")
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj = _import_candidate(tree, module_name)
        except Exception:                                  # noqa: BLE001
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute)
        return obj
    raise ImportError("cannot resolve " + dotpath + " in " + str(tree))


def _torch():
    import torch
    return torch


# --- templates ----------------------------------------------------------
def signal_sensitivity(tree: Path, *, set: dict, observe: dict,
                       low: float = 0.05, high: float = 1.3) -> dict:
    """Change the behaviour input, and see whether the observed quantity moves.

    The contract names a callable that turns the behaviour input into the geometric
    quantity (``set``) and a callable that observes the computation (``observe``).
    Two different behaviour inputs must produce two different observations; the same
    observation means the behaviour reaches nothing.
    """
    torch = _torch()
    setter = _resolve(tree, set["callable"])
    observer = _resolve(tree, observe["callable"])
    argument = str(set.get("arg") or "value")
    with torch.no_grad():
        first = observer(**{argument: setter(float(low))})
        second = observer(**{argument: setter(float(high))})
    difference = float((first - second).abs().max()) if torch.is_tensor(first) else 0.0
    if difference <= 1e-9:
        return {"status": "FAIL", "template": "SignalSensitivity",
                "detail": "the observed quantity is identical for two behaviour inputs "
                          "(max difference %.3e)" % difference,
                "difference": difference}
    return {"status": "PASS", "template": "SignalSensitivity",
            "detail": "the observation moves with the behaviour input",
            "difference": difference}


def independent_formula(tree: Path, *, candidate: str, reference: str,
                        grid: dict = None) -> dict:
    """Compare the candidate's operation with the canonical one on a grid.

    The expectation comes from the framework's own module, never from the code under
    test: that is what makes it independent.
    """
    torch = _torch()
    import importlib
    implementation = _resolve(tree, candidate)
    reference_module = importlib.import_module("tools.research.geometry.poincare_contract")
    expected = getattr(reference_module, reference.split(".")[-1])
    values = (grid or {}).get("values") or [0.05, 0.2, 0.5]
    curvature = float((grid or {}).get("curvature") or 1.0)
    worst = 0.0
    for value in values:
        point = torch.tensor([[float(value), 0.0]])
        other = torch.tensor([[0.0, float(value)]])
        got = implementation(point, other, curvature)
        if str(reference) == "closed_form_1d":
            # The 1-D Poincare distance written by hand: an expectation that does
            # not come from any module under test.
            root = math.sqrt(curvature)
            a, b = float(point[0, 0]), float(other[0, 0])
            ratio = root * abs(b - a) / abs(1 - curvature * a * b)
            want = torch.tensor([(2.0 / root) * math.atanh(min(ratio, 1 - 1e-9))])
        else:
            want = expected(point, other, curvature)
        worst = max(worst, float((got.reshape(-1) - want.reshape(-1)).abs().max()))
    if worst > 1e-5:
        return {"status": "FAIL", "template": "IndependentFormula",
                "detail": "the candidate differs from the canonical " + reference +
                          " by up to %.3e" % worst, "difference": worst}
    return {"status": "PASS", "template": "IndependentFormula",
            "detail": "the candidate matches the canonical " + reference,
            "difference": worst}


def range_and_saturation(tree: Path, *, callable: str, range_source: dict,
                         tolerance: float = 1e-6) -> dict:
    """Does the quantity saturate over the range the training data will produce?

    A quantity pinned to a constant over the whole feasible range cannot carry the
    behaviour signal, whatever the formula says.
    """
    function = _resolve(tree, callable)
    low = float(range_source.get("low", 0.02))
    high = float(range_source.get("high", 0.99))
    steps = int(range_source.get("steps", 9))
    values = []
    for index in range(steps):
        q = low + (high - low) * index / max(1, steps - 1)
        values.append(float(function(q)))
    span = max(values) - min(values)
    if span <= tolerance:
        return {"status": "FAIL", "template": "RangeAndSaturation",
                "detail": "the quantity is constant over [%.3f, %.3f]: %.6f" %
                          (low, high, values[0]), "span": span}
    return {"status": "PASS", "template": "RangeAndSaturation",
            "detail": "the quantity varies over [%.3f, %.3f]: span %.6f" %
                      (low, high, span), "span": span}


def assignment_consistency(tree: Path, *, compare: dict) -> dict:
    """The exported SID is the code the training path assigns."""
    training = str(compare.get("training") or "tests.test_radial_export")
    import subprocess
    done = subprocess.run([sys.executable, "-m", "unittest", training],
                          cwd=str(Path(tree)), capture_output=True, text=True,
                          timeout=900)
    status = "PASS" if done.returncode == 0 else "FAIL"
    tail = (done.stdout + done.stderr).strip().splitlines()[-8:]
    return {"status": status, "template": "AssignmentConsistency",
            "detail": " | ".join(tail)[:1500], "exit_code": done.returncode}


def delegate(tree: Path, *, template: str, module: str) -> dict:
    """Templates whose check is a candidate-side test named by the contract."""
    import subprocess
    done = subprocess.run([sys.executable, "-m", "unittest", module], cwd=str(Path(tree)),
                          capture_output=True, text=True, timeout=900)
    status = "PASS" if done.returncode == 0 else "FAIL"
    tail = (done.stdout + done.stderr).strip().splitlines()[-8:]
    return {"status": status, "template": template, "detail": " | ".join(tail)[:1500],
            "exit_code": done.returncode}


def run_tests(tree: Path, contract: dict) -> dict:
    """Run every test the contract names."""
    tree = Path(tree)
    results = {}
    for entry in contract.get("tests") or []:
        name = str(entry.get("template"))
        try:
            if name == "SignalSensitivity":
                results[name] = signal_sensitivity(tree, **{k: v for k, v in entry.items()
                                                            if k != "template"})
            elif name == "IndependentFormula":
                results[name] = independent_formula(tree, **{k: v for k, v in entry.items()
                                                             if k != "template"})
            elif name == "RangeAndSaturation":
                results[name] = range_and_saturation(tree, **{k: v for k, v in entry.items()
                                                              if k != "template"})
            elif name == "AssignmentConsistency":
                results[name] = assignment_consistency(tree, **{k: v for k, v in entry.items()
                                                                if k != "template"})
            elif name in ("GradientFlow", "AblationEffect", "DataBoundary"):
                results[name] = delegate(tree, template=name,
                                         module=str(entry.get("module") or ""))
            else:
                results[name] = {"status": "NOT_RUN", "template": name,
                                 "detail": "unhandled template"}
        except Exception as exc:                           # noqa: BLE001
            results[name] = {"status": "FAIL", "template": name,
                             "detail": type(exc).__name__ + ": " + str(exc)}
    statuses = [entry.get("status") for entry in results.values()]
    return {"tests": results,
            "ran": all(status in ("PASS", "FAIL") for status in statuses),
            "passed": bool(statuses) and all(status == "PASS" for status in statuses)}
