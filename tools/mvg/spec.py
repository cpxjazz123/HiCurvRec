"""The frozen ``MechanismSpec``: schema, parser, validator and digest.

A specification is a CLAIM about a mechanism that is written down *before* the mechanism
exists, so that the implementation can be checked against something other than itself. This
module is the parser and the validator; the normative schema is
``tools/mvg/schemas/mechanism_spec.schema.json``.

Two rules shape the validator:

* a key nobody reads is a claim nobody checks, so every key is either understood and acted
  on here, or absent;
* the validator is deliberately hand-written and dependency-free: it has to run in the
  project's own Python 3.9 environment, next to the trainer, with nothing installed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

SCHEMA_VERSION = 1

MVG_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVG_DIR.parent.parent
SCHEMA_PATH = MVG_DIR / "schemas" / "mechanism_spec.schema.json"
SPECS_DIR = MVG_DIR / "specs"

CASE_ROLES = ("positive", "negative", "boundary", "degenerate")
GRADIENT_EXPECTATIONS = ("nonzero", "zero", "finite")
CGV_GROUPS = (
    "mechanism",
    "model",
    "forward_contract",
    "gradient_contract",
    "update_contract",
    "distributed_contract",
    "runtime_contract",
)

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class SpecError(ValueError):
    """A specification that cannot be trusted."""


def canonical_json(doc: Dict[str, Any]) -> str:
    """A canonical rendering: sorted keys, no insignificant whitespace."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_digest(doc: Dict[str, Any]) -> str:
    """The sha256 of the canonical rendering -- the identity of this spec version."""
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matrix_ok(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    width = None
    for row in value:
        if not isinstance(row, list) or not row:
            return False
        if not all(_is_number(v) for v in row):
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
    return True


def validate_spec(doc: Any) -> List[str]:
    """Return every problem in the specification. An empty list means it is trustworthy."""
    problems: List[str] = []
    if not isinstance(doc, dict):
        return ["the specification must be an object"]

    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version must be {SCHEMA_VERSION}, got {doc.get('schema_version')!r}"
        )

    mech = doc.get("mechanism")
    if not isinstance(mech, dict):
        problems.append("mechanism: missing")
    else:
        for key in ("id", "version", "summary", "directionality", "masking",
                    "normalization", "coefficient_policy"):
            if not isinstance(mech.get(key), str) or not mech.get(key, "").strip():
                problems.append(f"mechanism.{key}: missing or empty")
        if isinstance(mech.get("id"), str) and not _NAME.match(mech["id"]):
            problems.append(f"mechanism.id {mech['id']!r} is not snake_case")
        if not isinstance(mech.get("permitted_approximations", []), list):
            problems.append("mechanism.permitted_approximations must be a list")

    integration = doc.get("integration")
    if not isinstance(integration, dict):
        problems.append("integration: missing")
    else:
        for key in ("tree", "module", "entrypoint", "intermediates_entrypoint", "call_site",
                    "host_module", "mechanism_weight_flag"):
            if not isinstance(integration.get(key), str) or not integration.get(key, "").strip():
                problems.append(f"integration.{key}: missing or empty")
        args = integration.get("arguments")
        if not isinstance(args, list) or not args:
            problems.append("integration.arguments: missing")
        else:
            for arg in args:
                if not isinstance(arg, dict) or not arg.get("name") or not arg.get("shape"):
                    problems.append("integration.arguments: every argument needs name and shape")

    objective = doc.get("objective")
    if not isinstance(objective, dict):
        problems.append("objective: missing")
    else:
        if not isinstance(objective.get("equation"), str) or not objective.get("equation", "").strip():
            problems.append("objective.equation: missing")
        if not isinstance(objective.get("symbols"), dict) or not objective.get("symbols"):
            problems.append("objective.symbols: missing")
        if not isinstance(objective.get("parameters"), dict) or not objective.get("parameters"):
            problems.append("objective.parameters: missing")
        elif not all(_is_number(v) for v in objective["parameters"].values()):
            problems.append("objective.parameters: every value must be a number")
        if not isinstance(objective.get("reference_implementation"), str):
            problems.append("objective.reference_implementation: missing")
        if not isinstance(objective.get("reference_symbols"), dict) or not objective.get(
            "reference_symbols"
        ):
            problems.append("objective.reference_symbols: missing")

    targets = doc.get("gradient_targets")
    if not isinstance(targets, dict) or not targets:
        problems.append("gradient_targets: missing")
    else:
        for name, expectation in targets.items():
            if expectation not in GRADIENT_EXPECTATIONS:
                problems.append(
                    f"gradient_targets.{name}: {expectation!r} is not one of {GRADIENT_EXPECTATIONS}"
                )

    forbidden = doc.get("forbidden_substitutions")
    if not isinstance(forbidden, list) or not forbidden:
        problems.append(
            "forbidden_substitutions: at least one is required; a specification that forbids "
            "nothing cannot decide whether an implementation substituted the operation"
        )
    else:
        for entry in forbidden:
            if not isinstance(entry, dict) or not entry.get("id") or not entry.get("pattern"):
                problems.append("forbidden_substitutions: every entry needs id and pattern")
                continue
            if not isinstance(entry.get("why"), str) or not entry.get("why", "").strip():
                problems.append(f"forbidden_substitutions.{entry.get('id')}: needs a why")
            try:
                re.compile(entry["pattern"])
            except re.error as exc:
                problems.append(
                    f"forbidden_substitutions.{entry.get('id')}: pattern does not compile: {exc}"
                )

    invariants = doc.get("invariants")
    if not isinstance(invariants, list) or not invariants:
        problems.append("invariants: at least one is required")
    else:
        for entry in invariants:
            if not isinstance(entry, dict) or not entry.get("id") or not entry.get("statement"):
                problems.append("invariants: every entry needs id and statement")
            elif entry.get("kind") not in ("numeric", "metamorphic", "structural"):
                problems.append(f"invariants.{entry.get('id')}: unknown kind {entry.get('kind')!r}")

    cases = doc.get("cases")
    if not isinstance(cases, dict):
        problems.append("cases: missing")
    else:
        for role in CASE_ROLES:
            entries = cases.get(role)
            if not isinstance(entries, list) or not entries:
                problems.append(f"cases.{role}: at least one case is required")
                continue
            for case in entries:
                if not isinstance(case, dict) or not case.get("id"):
                    problems.append(f"cases.{role}: every case needs an id")
                    continue
                if case.get("role") != role:
                    problems.append(
                        f"cases.{role}.{case.get('id')}: role field says {case.get('role')!r}"
                    )
                inputs = case.get("inputs")
                if not isinstance(inputs, dict):
                    problems.append(f"cases.{role}.{case.get('id')}: inputs missing")
                    continue
                for key in ("z_src", "z_tgt", "codebook"):
                    if not _matrix_ok(inputs.get(key)):
                        problems.append(
                            f"cases.{role}.{case.get('id')}.inputs.{key}: must be a matrix of numbers"
                        )
                mask = inputs.get("mask")
                if not isinstance(mask, list) or not mask or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in mask
                ):
                    problems.append(
                        f"cases.{role}.{case.get('id')}.inputs.mask: must be a vector of numbers"
                    )
                elif isinstance(inputs.get("z_src"), list) and len(mask) != len(inputs["z_src"]):
                    problems.append(
                        f"cases.{role}.{case.get('id')}: mask length {len(mask)} != batch "
                        f"{len(inputs['z_src'])}"
                    )
                if isinstance(inputs.get("z_src"), list) and isinstance(
                    inputs.get("z_tgt"), list
                ):
                    if len(inputs["z_src"]) != len(inputs["z_tgt"]):
                        problems.append(
                            f"cases.{role}.{case.get('id')}: z_src and z_tgt must have the same rows"
                        )
                    elif inputs["z_src"] and inputs["codebook"]:
                        dim = len(inputs["z_src"][0])
                        if len(inputs["codebook"][0]) != dim:
                            problems.append(
                                f"cases.{role}.{case.get('id')}: codebook dim "
                                f"{len(inputs['codebook'][0])} != residual dim {dim}"
                            )

    tolerances = doc.get("tolerances")
    if not isinstance(tolerances, dict) or not tolerances:
        problems.append("tolerances: missing")
    else:
        for name, tol in tolerances.items():
            if not isinstance(tol, dict) or not _is_number(tol.get("atol")) or not _is_number(
                tol.get("rtol")
            ):
                problems.append(f"tolerances.{name}: needs numeric atol and rtol")
            elif tol["atol"] <= 0 or tol["rtol"] <= 0:
                problems.append(f"tolerances.{name}: atol and rtol must be positive")

    cgv = doc.get("cgv_contract")
    if not isinstance(cgv, dict):
        problems.append("cgv_contract: missing")
    else:
        for group in CGV_GROUPS:
            if not isinstance(cgv.get(group), dict):
                problems.append(f"cgv_contract.{group}: missing")

    provenance = doc.get("provenance")
    if not isinstance(provenance, dict):
        problems.append("provenance: missing")
    else:
        for key in ("authored_by", "authored_at", "review"):
            if not isinstance(provenance.get(key), str) or not provenance.get(key, "").strip():
                problems.append(f"provenance.{key}: missing")
        for key in ("source_digests", "dataset_digests"):
            if not isinstance(provenance.get(key), dict):
                problems.append(f"provenance.{key}: missing")

    return problems


def load_spec(path: Path | str) -> Dict[str, Any]:
    """Read, validate and return a specification. Raises ``SpecError`` when unusable."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read the specification {path}: {exc}") from exc
    problems = validate_spec(doc)
    if problems:
        raise SpecError(
            f"the specification {path} is not trustworthy:\n  - "
            + "\n  - ".join(problems)
        )
    return doc


def spec_path(mechanism_id: str) -> Path:
    return SPECS_DIR / f"{mechanism_id}.json"


__all__ = [
    "CASE_ROLES",
    "CGV_GROUPS",
    "GRADIENT_EXPECTATIONS",
    "MVG_DIR",
    "REPO_ROOT",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "SPECS_DIR",
    "SpecError",
    "canonical_json",
    "load_spec",
    "schema",
    "spec_digest",
    "spec_path",
    "validate_spec",
]
