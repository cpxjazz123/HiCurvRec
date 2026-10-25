"""The versioned weighted rubric (Issue #122).

The weights and the anchors of the scientific review are a policy, not code:
they live in a JSON file with a version and a digest, so a decision can be
reproduced later and a change of policy can never be mistaken for a change of
evidence. A round pins the rubric it was scored with; the file itself is only
edited before a new round starts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RUBRIC_FILE = Path(__file__).with_name("rubric_v1.json")

#: The literals the evidence bundle uses when something is not known.
UNKNOWN = "UNKNOWN"
NOT_CHECKED = "NOT_CHECKED"


class RubricError(ValueError):
    """The rubric file is unusable or does not match the code's expectations."""


def load_rubric(path: Path = None) -> dict:
    """Load and validate the rubric policy."""
    path = Path(path) if path is not None else RUBRIC_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RubricError("cannot read the rubric: " + str(exc)) from exc
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise RubricError("the rubric names no dimensions")
    names, weights = [], []
    for entry in dimensions:
        name = str(entry.get("name", "")).strip()
        weight = entry.get("weight")
        if not name or not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise RubricError("a rubric dimension is missing its name or weight")
        if not entry.get("anchors"):
            raise RubricError("dimension " + name + " has no anchored meanings")
        names.append(name)
        weights.append(float(weight))
    if len(set(names)) != len(names):
        raise RubricError("the rubric repeats a dimension")
    total = round(sum(weights), 6)
    if total != 100.0:
        raise RubricError("the weights must sum to 100, got " + str(total))
    payload["dimension_names"] = names
    payload["weight_by_dimension"] = dict(zip(names, weights))
    payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["path"] = str(path)
    return payload


def rubric_version(payload: dict) -> str:
    return str(payload.get("version", ""))


def weighted_score(medians: dict, payload: dict) -> float | None:
    """sum(weight[d] * median[d]) / 100 over the rubric's dimensions.

    A missing dimension makes the score undefined: an incomplete review never
    becomes a number by averaging whatever happened to be filled.
    """
    weights = payload["weight_by_dimension"]
    total = 0.0
    for dimension, weight in weights.items():
        value = medians.get(dimension)
        if value is None:
            return None
        total += weight * float(value)
    return round(total / 100.0, 6)
