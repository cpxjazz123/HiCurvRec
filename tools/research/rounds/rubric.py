"""The versioned selection policy of the scientific review (Issue #122).

The reviewers rank the proposals and the policy that turns their ranks into one
mechanism is a file, not code: it has a version and a digest, so a decision can
be reproduced later and a change of policy can never be mistaken for a change of
evidence. A round pins the policy it was decided with; the file itself is only
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
    """The policy file is unusable or does not match the code's expectations."""


def load_rubric(path: Path = None) -> dict:
    """Load and validate the selection policy."""
    path = Path(path) if path is not None else RUBRIC_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RubricError("cannot read the policy: " + str(exc)) from exc
    if not str(payload.get("version", "")).strip():
        raise RubricError("the policy has no version")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise RubricError("the policy names no selection rule")
    if not isinstance(selection.get("max_selected"), int) or selection["max_selected"] < 1:
        raise RubricError("the policy must keep at least one mechanism")
    if not isinstance(selection.get("min_reviewers"), int) or selection["min_reviewers"] < 1:
        raise RubricError("the policy must ask for at least one reviewer")
    if not selection.get("tie_break_order"):
        raise RubricError("the policy names no tie-break order")
    payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["path"] = str(path)
    return payload


def rubric_version(payload: dict) -> str:
    return str(payload.get("version", ""))
