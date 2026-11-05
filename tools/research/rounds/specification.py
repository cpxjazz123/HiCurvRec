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
                     "parameters", "switches", "excluded")

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
    surface = spec.get("continuous_surface")
    if isinstance(surface, dict):
        statements = surface.get("statements")
        if not statements:
            problems.append("the continuous surface states no statements")
        for statement in statements or []:
            if "=" not in str(statement):
                problems.append("the statement %r is not an assignment" % statement)
    return problems


def freeze(directory: Path, spec: dict) -> dict:
    """Write the specification and its hash; the hash is what the code cites."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(spec, indent=1, sort_keys=True, ensure_ascii=False)
    path = directory / SPEC_NAME
    path.write_text(payload + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def load(directory: Path) -> dict:
    """The frozen specification of a round, with its hash."""
    path = Path(directory) / SPEC_NAME
    if not path.is_file():
        return {}
    payload = path.read_text(encoding="utf-8")
    return {"spec": json.loads(payload),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "path": str(path)}


def statements(spec: dict) -> list:
    """The SymPy statements of the continuous surface, in the frozen order."""
    surface = (spec or {}).get("continuous_surface") or {}
    return list(surface.get("statements") or [])


def output_name(spec: dict) -> str:
    """The name of the surface's output expression."""
    surface = (spec or {}).get("continuous_surface") or {}
    return str(surface.get("output") or "")
