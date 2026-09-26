#!/usr/bin/env python3
"""FCCR-1 mechanism-contract preflight.

Run with NO CLI arguments from:
  stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/

Static preflight: blocks Stage2 when source contradicts the
fixed closed-form curvature contract.
"""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

EXPECTED = {
    "contract_version": "FCCR-1",
    "curvature_source": "closed_form",
    "curvature_trainable": False,
    "curvature_time_varying": False,
    "uses_cyclic_schedule": False,
    "uses_curvature_regularization": False,
    "new_curvature_conditioned_optimizer": False,
    "new_curvature_conditioned_aux_loss": False,
}

def fail(msg: str) -> None:
    raise RuntimeError(f"MECHANISM_CONTRACT_FAIL: {msg}")

def iter_id_from_cwd(work: Path) -> str:
    m = re.fullmatch(r"curvature_RQ-VAE_iter(\d+)", work.name)
    if not m:
        fail(f"run from an iteration directory, got {work}")
    return m.group(1)

def required_artifacts(work: Path, iter_id: str) -> dict[str, Path]:
    logs = work / "logs"
    paths = {
        "protocol": logs / f"protocol_manifest_iter{iter_id}.md",
        "hypothesis": logs / f"hypothesis_iter{iter_id}.md",
        "manifest": logs / f"mechanism_manifest_iter{iter_id}.md",
        "contract": logs / f"mechanism_contract_iter{iter_id}.json",
        "diff": logs / f"one_factor_diff_iter{iter_id}.md",
    }
    for label, path in paths.items():
        if not path.is_file():
            fail(f"missing mandatory preflight artifact ({label}): {path}")
    return paths

def load_contract(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot parse contract JSON {path}: {exc}")

    for key, expected in EXPECTED.items():
        if payload.get(key) != expected:
            fail(f"contract field {key}={payload.get(key)!r}; expected {expected!r}")

    if payload.get("formula_inputs") != ["behavior_branching", "raw_residual_median"]:
        fail("formula_inputs must be exactly ['behavior_branching', 'raw_residual_median']")

    values = payload.get("final_curvature_values")
    if not isinstance(values, list) or len(values) != 3:
        fail("final_curvature_values must contain exactly 3 numbers")
    for value in values:
        if not isinstance(value, (int, float)):
            fail(f"non-numeric curvature value: {value!r}")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            fail(f"invalid curvature value: {value}")
    return payload

def semantic_manifest_checks(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    for term in ("raw residual", "behavior branching", "normalized layer scale"):
        if term not in text:
            fail(f"mechanism manifest missing semantic/provenance term: {term!r}")

def name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = name_of(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""

def inspect_source(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        fail(f"cannot parse {path}: {exc}")

    trainable = []
    step_dependent = []
    fixed_buffers = []
    reg_weights = []

    curvature_tokens = (
        "c_layer_scale",
        "log_c",
        "curvature_scale",
        "learnable_c",
        "c_param",
    )

    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            cname = name_of(node.func)
            if cname.endswith("register_buffer") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value in {"fixed_c", "_fixed_c"}:
                    fixed_buffers.append(f"{path}:{getattr(node, 'lineno', '?')}")

            if cname in {"nn.Parameter", "torch.nn.Parameter"}:
                line = getattr(node, "lineno", 1)
                lo = max(0, line - 3)
                hi = min(len(lines), line + 1)
                context = "\n".join(lines[lo:hi])
                if any(tok in context for tok in curvature_tokens):
                    trainable.append(f"{path}:{line}: {context.strip()}")

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                tname = name_of(target)
                if tname.endswith("CURVATURE_REG_WEIGHT"):
                    try:
                        reg_weights.append(float(ast.literal_eval(node.value)))
                    except Exception:
                        pass

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_c":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in {
                    "_curriculum_step", "curriculum_step", "global_step"
                }:
                    step_dependent.append(f"{path}:{getattr(sub, 'lineno', '?')}:{sub.attr}")

    return {
        "trainable": trainable,
        "step_dependent": step_dependent,
        "fixed_buffers": fixed_buffers,
        "reg_weights": reg_weights,
    }

def main() -> None:
    work = Path.cwd().resolve()
    iter_id = iter_id_from_cwd(work)
    paths = required_artifacts(work, iter_id)
    contract = load_contract(paths["contract"])
    semantic_manifest_checks(paths["manifest"])

    source_files = [
        work / "curvature_RQ-VAE.py",
        work / "modules" / "quantize.py",
        work / "modules" / "rqvae.py",
    ]
    source_files = [p for p in source_files if p.is_file()]
    if not source_files:
        fail("no active Python source files found")

    trainable, step_dependent, fixed_buffers, reg_weights = [], [], [], []
    for path in source_files:
        result = inspect_source(path)
        trainable.extend(result["trainable"])
        step_dependent.extend(result["step_dependent"])
        fixed_buffers.extend(result["fixed_buffers"])
        reg_weights.extend(result["reg_weights"])

    if trainable:
        fail("trainable curvature parameter found:\n" + "\n".join(trainable[:8]))

    if step_dependent:
        fail("get_c() is step/time dependent:\n" + "\n".join(step_dependent[:8]))

    if not fixed_buffers:
        fail("no register_buffer('fixed_c' or '_fixed_c', ...) found")

    if any(abs(v) > 1e-15 for v in reg_weights):
        fail(f"non-zero CURVATURE_REG_WEIGHT found: {reg_weights}")

    values = [float(x) for x in contract["final_curvature_values"]]
    print("MECHANISM_CONTRACT_PASS")
    print(f"iter={iter_id}")
    print("contract=FCCR-1")
    print(f"fixed_curvature={values}")
    print("curvature_trainable=false")
    print("curvature_time_varying=false")
    print("uses_cyclic_schedule=false")
    print("uses_curvature_regularization=false")
    print(f"fixed_buffer_sites={fixed_buffers}")

if __name__ == "__main__":
    main()
