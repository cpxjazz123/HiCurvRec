"""The frozen mathematical specification of a mechanism.

The Formula Agent writes the specification, the Math Reviewers check it
against the idea it came from, and the freeze writes the file and its hash.
After the freeze the specification is the ONLY authority the implementation may
follow: the reference contract takes its statements from it, and the
implementation may not reinterpret it.

Three responsibilities are kept apart on purpose:

* the IDEA agent decides WHAT is researched;
* the FORMULA agent decides HOW the idea is written as mathematics - and it
  may not change the idea, only complete it; if the idea lacks a definition the
  mechanism needs, it answers ``SPECIFICATION_INCOMPLETE`` and says which
  definitions are missing;
* the CODING agent implements the frozen specification faithfully.

The specification must cover every computation the mechanism touches, not only
a loss: the metric, the behaviour's entry point, the assignment, the gradient
path, the reconstruction, the switches and the surfaces that are deliberately
out of scope.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPEC_NAME = "math_spec.json"

#: The sections a specification must have. A missing one is a specification
#: that would leave an implementation free to interpret.
REQUIRED_SECTIONS = ("mechanism", "computations", "inputs", "outputs",
                     "parameters", "dimensions", "gradient_rules",
                     "training_integration", "switches", "excluded")

#: The answer of a Formula Agent that finds the idea incomplete.
INCOMPLETE = "SPECIFICATION_INCOMPLETE"


def validate(spec: dict) -> list:
    """The problems of a specification; an empty list means it is complete."""
    problems = []
    if not isinstance(spec, dict):
        return ["the specification is not an object"]
    for section in REQUIRED_SECTIONS:
        if section not in spec:
            problems.append("the section %r is missing" % section)
        elif not spec[section]:
            problems.append("the section %r is empty" % section)
    computations = spec.get("computations")
    if isinstance(computations, list):
        if not computations:
            problems.append("no computation is stated")
        for entry in computations:
            if not isinstance(entry, dict):
                problems.append("a computation is not an object")
                continue
            for key in ("name", "definition", "inputs", "output"):
                if not entry.get(key):
                    problems.append("the computation %r has no %r"
                                    % (entry.get("name", "?"), key))
    dimensions = spec.get("dimensions")
    if isinstance(dimensions, dict):
        for name, size in dimensions.items():
            if not str(name):
                problems.append("a dimension has no name")
            if size in (None, "", 0):
                problems.append("the dimension %r has no size" % name)
    else:
        problems.append("the dimensions are not stated as a mapping")
    rules = spec.get("gradient_rules")
    if isinstance(rules, list):
        if not rules:
            problems.append("no gradient rule is stated")
        for rule in rules:
            if not isinstance(rule, dict) or not rule.get("operator")                     or not rule.get("rule"):
                problems.append("a gradient rule does not name its operator and its rule")
    else:
        problems.append("the gradient rules are not stated as a list")
    integration = spec.get("training_integration")
    if isinstance(integration, dict):
        for key in ("loss", "reconstruction", "optimizer"):
            if not integration.get(key):
                problems.append("the training integration does not state %r" % key)
    else:
        problems.append("the training integration is not stated")
    surface = spec.get("continuous_surface")
    if isinstance(surface, dict):
        statements = surface.get("statements")
        if not statements:
            problems.append("the continuous surface states no statements")
        for statement in statements or []:
            if "=" not in str(statement):
                problems.append("the statement %r is not an assignment" % statement)
        if not surface.get("symbols"):
            problems.append("the continuous surface states no symbols")
        if not surface.get("args"):
            problems.append("the continuous surface states no arguments")
        if not surface.get("callable"):
            problems.append("the continuous surface names no callable")
    else:
        problems.append("the continuous surface is not stated")
    return problems


def _canonical(spec: dict) -> str:
    """The one serialisation the hash is taken over: written and read alike."""
    return json.dumps(spec, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def freeze(directory: Path, spec: dict) -> dict:
    """Write the specification and its hash; the hash is what the code cites."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = _canonical(spec)
    path = directory / SPEC_NAME
    path.write_text(payload, encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def load(directory: Path) -> dict:
    """The frozen specification of a round, with the hash of the bytes on disk."""
    path = Path(directory) / SPEC_NAME
    if not path.is_file():
        return {}
    payload = path.read_text(encoding="utf-8")
    return {"spec": json.loads(payload),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "path": str(path)}


def matches_frozen(directory: Path, recorded_sha256: str) -> dict:
    """Whether the frozen file on disk is the one the record cites."""
    frozen = load(directory)
    if not frozen:
        return {"matches": False, "reason": "no frozen specification on disk"}
    if frozen["sha256"] != recorded_sha256:
        return {"matches": False,
                "reason": "the frozen file changed after the freeze (%s -> %s)"
                          % (str(recorded_sha256)[:12], frozen["sha256"][:12])}
    return {"matches": True, "sha256": frozen["sha256"], "spec": frozen["spec"]}


def statements(spec: dict) -> list:
    """The SymPy statements of the continuous surface, in the frozen order."""
    surface = (spec or {}).get("continuous_surface") or {}
    return list(surface.get("statements") or [])


def output_name(spec: dict) -> str:
    """The name of the surface's output expression."""
    surface = (spec or {}).get("continuous_surface") or {}
    return str(surface.get("output") or "")
