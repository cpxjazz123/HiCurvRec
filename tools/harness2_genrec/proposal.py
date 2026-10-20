"""The frozen mechanism proposal.

A proposal is what an agent says it is going to build, written down and **frozen before
the implementation exists**. That order is the whole point: the implementation is then
checked against a claim that was made before the implementation could shape it.

The proposal is the mechanism's own declaration, and it is not editable by the agent
that implements it. Once it is frozen its digest is part of the run's manifest, and a
change to it is a new proposal that invalidates the previous admission.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

SCHEMA_VERSION = 1

TARGET_PATHS = ("encoder", "codebook", "hard_sid_export")
TEST_KINDS = ("numeric", "metamorphic", "integration", "gradient", "distributed")

#: The independent checks a proposal may ask for. Kept here as a literal so that validating a
#: proposal never imports the judge registry -- the registry pulls in torch, and this
#: package has to run in the framework's own environment. A test keeps the two in step.
KNOWN_JUDGES = (
    "contract_audit",
    "forward_oracle",
    "derivative_evidence",
    "property_checks",
    "integration_on_off",
)

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ProposalError(ValueError):
    """A proposal that cannot be trusted."""


def canonical_json(doc: Dict[str, Any]) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def proposal_digest(doc: Dict[str, Any]) -> str:
    """The identity of this proposal version."""
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def validate_proposal(doc: Any) -> List[str]:
    """Every problem in the proposal. An empty list means it may be frozen."""
    problems: List[str] = []
    if not isinstance(doc, dict):
        return ["the proposal must be an object"]
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}, got {doc.get('schema_version')!r}")

    for key in ("mechanism_id", "version", "title", "equation"):
        value = doc.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{key}: missing or empty")
    mechanism_id = doc.get("mechanism_id")
    if isinstance(mechanism_id, str) and not _NAME.match(mechanism_id):
        problems.append(f"mechanism_id {mechanism_id!r} is not snake_case")

    symbols = doc.get("symbols")
    if not isinstance(symbols, dict) or not symbols:
        problems.append("symbols: missing")

    parameters = doc.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        problems.append("parameters: missing")
    else:
        for name, value in parameters.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"parameters.{name}: must be a number")

    targets = doc.get("target_paths")
    if not isinstance(targets, list) or not targets:
        problems.append("target_paths: missing")
    else:
        for target in targets:
            if target not in TARGET_PATHS:
                problems.append(
                    f"target_paths: {target!r} is not one of {list(TARGET_PATHS)}"
                )

    for key in ("hypotheses", "expected_failure_modes"):
        value = doc.get(key)
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            problems.append(f"{key}: at least one non-empty string is required")

    tests = doc.get("tests")
    if not isinstance(tests, list) or not tests:
        problems.append("tests: at least one mechanism-specific test is required")
    else:
        for test in tests:
            if not isinstance(test, dict) or not test.get("id") or not test.get("statement"):
                problems.append("tests: every test needs an id and a statement")
                continue
            if test.get("kind") not in TEST_KINDS:
                problems.append(
                    f"tests.{test.get('id')}: kind {test.get('kind')!r} is not one of "
                    f"{list(TEST_KINDS)}"
                )

    judges = doc.get("judges")
    if not isinstance(judges, list) or not judges:
        problems.append("judges: at least one independent check is required")
    else:
        unknown = [name for name in judges if name not in KNOWN_JUDGES]
        if unknown:
            problems.append(
                f"judges: {unknown} are not judges this repository knows; known judges are "
                f"{sorted(KNOWN_JUDGES)}"
            )

    forbidden = doc.get("forbidden_substitutions")
    if not isinstance(forbidden, list) or not forbidden:
        problems.append(
            "forbidden_substitutions: at least one is required; a proposal that forbids "
            "nothing cannot decide whether the implementation substituted the operation"
        )
    else:
        for entry in forbidden:
            if not isinstance(entry, dict) or not entry.get("id") or not entry.get("pattern"):
                problems.append("forbidden_substitutions: every entry needs id and pattern")
                continue
            try:
                re.compile(entry["pattern"])
            except re.error as exc:
                problems.append(
                    f"forbidden_substitutions.{entry.get('id')}: pattern does not compile: {exc}"
                )

    provenance = doc.get("provenance")
    if not isinstance(provenance, dict):
        problems.append("provenance: missing")
    else:
        for key in ("authored_by", "authored_at"):
            if not isinstance(provenance.get(key), str) or not provenance.get(key, "").strip():
                problems.append(f"provenance.{key}: missing")
        if not isinstance(provenance.get("frozen_digest"), str):
            problems.append("provenance.frozen_digest: missing")

    return problems


def load_proposal(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProposalError(f"cannot read the proposal {path}: {exc}") from exc
    problems = validate_proposal(doc)
    if problems:
        raise ProposalError(
            f"the proposal {path} is not trustworthy:\n  - " + "\n  - ".join(problems)
        )
    return doc


def freeze(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return the proposal with its digest in place, ready to be written down."""
    frozen = dict(doc)
    provenance = dict(frozen.get("provenance") or {})
    provenance["frozen_digest"] = proposal_digest(
        {k: v for k, v in frozen.items() if k != "provenance"}
    )
    frozen["provenance"] = provenance
    problems = validate_proposal(frozen)
    if problems:
        raise ProposalError(
            "the frozen proposal is not valid:\n  - " + "\n  - ".join(problems)
        )
    return frozen


__all__ = [
    "KNOWN_JUDGES",
    "ProposalError",
    "SCHEMA_VERSION",
    "TARGET_PATHS",
    "TEST_KINDS",
    "canonical_json",
    "freeze",
    "load_proposal",
    "proposal_digest",
    "validate_proposal",
]
