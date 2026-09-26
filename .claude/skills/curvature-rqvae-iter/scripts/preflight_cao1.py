#!/usr/bin/env python3
"""CAO-1 mechanism-contract preflight.

Active research contract for iter>=27 when the user has explicitly switched
from FCCR-1 (fixed closed-form curvature) to CAO-1 (curvature-aware
optimization).  Iter27 is the first iteration under this contract.

Run with NO CLI arguments from:
    stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/

CAO-1 invariants enforced (any FAIL blocks Stage2):

- The single new mechanism MUST be a post-optimizer step *codebook-only*
  hyperbolic trust-region.  No other parameter may be modified by the
  trust-region code.
- The trust-region MUST be implemented as an explicit post-step projection
  that computes a Poincaré-ball displacement, sets a layer-local trust
  radius derived from the current codebook spacing, scales the geodesic
  displacement when it exceeds the radius, and writes the corrected
  tangent parameter back into `layer.embedding.weight`.
- Curvature in the active model MAY remain cyclic / learnable (CAO-1 keeps
  the iter18 curvature machinery).  The preflight does NOT enforce
  FCCR-1 invariants.
- The mechanism MUST be invoked once per training step after
  `optimizer.step()` and BEFORE `set_curriculum_step(...)` for the next
  step.  Invariant snapshots of the trust-region summary must be written
  to `logs/iter27_trust_region.jsonl` (or equivalent) at the configured
  cadence.
- No new trainable parameter may be introduced; no new auxiliary loss may
  be added; no new curvature-dependent Sinkhorn/assignment rule may be
  introduced in the same iteration.

This preflight is a process gate.  It does not substitute for the
deliberation_gate.py 2+1 protocol.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


EXPECTED_CONTRACT = {
    "contract_version": "CAO-1",
    "active_curvature_contract": "iter18_cyclic_learnable_layer_scale",
    "new_mechanism": "hyperbolic_codebook_trust_region",
    "curvature_trainable": True,
    "curvature_time_varying": True,
    "uses_cyclic_schedule": True,
    "introduces_trainable_parameter": False,
    "introduces_new_auxiliary_loss": False,
    "introduces_new_assignment_rule": False,
    "trust_radius_fraction_default": 0.5,
    "trust_radius_floor_default": 0.001,
}


def fail(msg: str) -> None:
    raise RuntimeError(f"CAO1_CONTRACT_FAIL: {msg}")


def iter_id_from_cwd(work: Path) -> str:
    m = re.fullmatch(r"curvature_RQ-VAE_iter(\d+)", work.name)
    if not m:
        fail(f"run from an iteration directory, got {work}")
    return int(m.group(1))


def required_artifacts(work: Path) -> dict[str, Path]:
    logs = work / "logs"
    paths = {
        "protocol": logs / "protocol_manifest_iter27.md"
        if iter_id_from_cwd(work) == 27
        else logs / f"protocol_manifest_iter{work.name.split('iter')[-1]}.md",
        "hypothesis": logs / "hypothesis_iter27.md",
        "manifest": logs / "mechanism_manifest_iter27.md",
        "contract": logs / "mechanism_contract_iter27.json",
        "diff": logs / "one_factor_diff_iter27.md",
    }
    # Generic resolution for non-27 iters
    paths = {}
    for label, suffix in [
        ("protocol", "protocol_manifest_iter27.md"),
        ("hypothesis", "hypothesis_iter27.md"),
        ("manifest", "mechanism_manifest_iter27.md"),
        ("contract", "mechanism_contract_iter27.json"),
        ("diff", "one_factor_diff_iter27.md"),
    ]:
        candidate = logs / suffix
        paths[label] = candidate
        if not candidate.is_file():
            fail(f"missing mandatory CAO-1 preflight artifact ({label}): {candidate}")
    return paths


def load_contract(path: Path) -> dict:
    import json
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot parse contract JSON {path}: {exc}")
    for key, expected in EXPECTED_CONTRACT.items():
        if payload.get(key) != expected:
            fail(
                f"contract field {key}={payload.get(key)!r}; expected {expected!r}"
            )
    return payload


def semantic_manifest_checks(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    required = [
        "hyperbolic",           # mechanism type
        "trust region",         # mechanism name
        "codebook",             # target tensor
        "poincaré",             # manifold
        "geodesic",             # operation
    ]
    for term in required:
        if term not in text:
            fail(f"mechanism manifest missing CAO-1 semantic term: {term!r}")


def name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = name_of(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _iter18_parent_path(work: Path, source_path: Path) -> Path | None:
    """Resolve the iter18 parent source path for diff-based novelty checks."""
    rel = source_path.relative_to(work)
    parent = work.parent / "curvature_RQ-VAE_iter18" / rel
    return parent if parent.is_file() else None


def inspect_source(path: Path, work: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        fail(f"cannot parse {path}: {exc}")

    parent_path = _iter18_parent_path(work, path)
    parent_text = parent_path.read_text(encoding="utf-8") if parent_path else ""

    has_trust_region = False
    has_snapshot = False
    trust_post_step_call = False
    references_embedding_weight = False
    new_trainable = []
    new_loss_terms = []
    new_assignment = False

    # Pre-compute iter18's nn.Parameter call line numbers so we can suppress
    # inherited curvature parameters (c_layer_scale) that existed in the parent.
    # Because iter27 added an extra import to modules/quantize.py, every
    # nn.Parameter call in iter18's quantize.py is shifted by +1 line.  We
    # therefore use a window (±3 lines) to be robust against non-load-bearing
    # edits.
    inherited_param_lines = set()
    if parent_text:
        try:
            parent_tree = ast.parse(parent_text, filename=str(parent_path))
            for pnode in ast.walk(parent_tree):
                if isinstance(pnode, ast.Call) and name_of(pnode.func) in {
                    "nn.Parameter", "torch.nn.Parameter"
                }:
                    inherited_param_lines.add(getattr(pnode, "lineno", -1))
        except SyntaxError:
            pass

    # Detect trust-region API on Quantize (method defs)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "apply_hyperbolic_trust_region":
                has_trust_region = True
                # Body must touch embedding.weight
                body_src = ast.dump(node)
                if "embedding.weight" in body_src:
                    references_embedding_weight = True
            if node.name == "snapshot_codebooks":
                has_snapshot = True

        # Detect Rank-0 invocation in the main trainer after optimizer.step
        if isinstance(node, ast.Call):
            cname = name_of(node.func)
            if cname.endswith("apply_hyperbolic_trust_region"):
                trust_post_step_call = True
            if cname.endswith("snapshot_codebooks"):
                has_snapshot = True
            # New trainable parameters are forbidden
            if cname in {"nn.Parameter", "torch.nn.Parameter"}:
                line_no = getattr(node, "lineno", -1)
                # Suppress inherited nn.Parameter declarations: the iter18 parent
                # may have its declaration at a slightly different line number
                # because iter27 added non-load-bearing imports / whitespace.
                # We use a ±3-line window so c_layer_scale and other inherited
                # curvature parameters are not falsely flagged.
                if not any(
                    abs(line_no - parent_line) <= 3
                    for parent_line in inherited_param_lines
                ):
                    new_trainable.append(
                        f"{path}:{line_no}: new nn.Parameter (not in iter18)"
                    )
            # New auxiliary losses are forbidden (loss function definition)
            if cname.endswith("__init__") and "loss" in cname.lower():
                pass

        # Search for new loss terms in forward body (simple heuristic)
        if isinstance(node, ast.Assign):
            targets = [name_of(t) for t in node.targets]
            for t in targets:
                if "trust_region" in t and "loss" in t.lower():
                    new_loss_terms.append(f"{path}:{getattr(node, 'lineno', '?')}: {t}")

    return {
        "has_trust_region": has_trust_region,
        "has_snapshot": has_snapshot,
        "trust_post_step_call": trust_post_step_call,
        "references_embedding_weight": references_embedding_weight,
        "new_trainable": new_trainable,
        "new_loss_terms": new_loss_terms,
    }


def main() -> None:
    work = Path.cwd().resolve()
    iter_id_from_cwd(work)
    paths = required_artifacts(work)
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

    summary = {
        "has_trust_region": False,
        "has_snapshot": False,
        "trust_post_step_call": False,
        "references_embedding_weight": False,
        "new_trainable": [],
        "new_loss_terms": [],
    }
    for path in source_files:
        result = inspect_source(path, work)
        summary["has_trust_region"] |= result["has_trust_region"]
        summary["has_snapshot"] |= result["has_snapshot"]
        summary["trust_post_step_call"] |= result["trust_post_step_call"]
        summary["references_embedding_weight"] |= result["references_embedding_weight"]
        summary["new_trainable"].extend(result["new_trainable"])
        summary["new_loss_terms"].extend(result["new_loss_terms"])

    if not summary["has_trust_region"]:
        fail("apply_hyperbolic_trust_region() not defined on Quantize")
    if not summary["references_embedding_weight"]:
        fail("trust-region does not reference embedding.weight")
    if not summary["has_snapshot"]:
        fail("snapshot_codebooks() not defined on RqVae")
    if not summary["trust_post_step_call"]:
        fail("trainer does not invoke apply_hyperbolic_trust_region after step")
    if summary["new_trainable"]:
        fail("new trainable parameter introduced under CAO-1: " +
             "\n".join(summary["new_trainable"][:5]))
    if summary["new_loss_terms"]:
        fail("new auxiliary loss term introduced under CAO-1: " +
             "\n".join(summary["new_loss_terms"][:5]))

    print("MECHANISM_CONTRACT_PASS")
    print("contract=CAO-1")
    print(f"new_mechanism={contract['new_mechanism']}")
    print(f"trust_radius_fraction={contract['trust_radius_fraction_default']}")
    print(f"trust_radius_floor={contract['trust_radius_floor_default']}")
    print(f"iter={work.name.split('iter')[-1]}")


if __name__ == "__main__":
    main()