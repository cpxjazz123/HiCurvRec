"""The immutable decision record (Issue #122).

When a round finishes scoring, its decision is written once, with everything
that produced it: the rubric identity, the proposal and evidence digests, the
three raw reviews, the weighted matrix, the pairwise result and the winner with
its backups. Later recalculation -- a fixed bug, a changed policy -- writes a
separate ``rescored_audit_<timestamp>.json`` and never edits the original
decision or the results that followed from it.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

DECISION_FILE = "decision.json"


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def proposal_digest(idea: dict) -> str:
    return _digest({key: idea.get(key) for key in sorted(idea) if key != "fingerprint"})


def build_snapshot(*, round_id: str, evaluation: dict, reports: list,
                   bundles: dict, proposals: list, source_commit: str = None) -> dict:
    """The decision record of one round, with its own hash."""
    rubric_version = evaluation.get("rubric_version")
    rubric_sha = evaluation.get("rubric_sha256")
    snapshot = {
        "schema": 1,
        "round_id": round_id,
        "decision_at": time.time(),
        "rubric_version": rubric_version,
        "rubric_sha256": rubric_sha,
        "scoring_source_commit": source_commit,
        "proposals": [{"idea_id": idea["idea_id"],
                       "proposal_sha256": proposal_digest(idea),
                       "evidence_sha256": (bundles.get(idea["idea_id"]) or {}).get("sha256")}
                      for idea in proposals],
        "reviews": [{key: report.get(key) for key in
                     ("scorer", "rubric_version", "rubric_sha256", "scores")}
                    for report in reports],
        "matrix": evaluation.get("matrix"),
        "pairwise": evaluation.get("pairwise"),
        "selection": evaluation.get("selection"),
        "lineage": {},
    }
    snapshot["decision_hash"] = _digest({key: value for key, value in snapshot.items()
                                        if key != "decision_hash"})
    return snapshot


def write_snapshot(directory: Path, snapshot: dict) -> Path:
    """Write the decision once. An existing decision is never overwritten."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DECISION_FILE
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("decision_hash") != snapshot.get("decision_hash"):
            raise RuntimeError("a decision for this round already exists and differs; "
                               "write a rescored audit instead")
        return path
    from ..core.evidence import atomic_json
    atomic_json(path, snapshot)
    return path


def write_audit(directory: Path, payload: dict, *, reason: str) -> Path:
    """A recalculation, recorded beside the decision and never instead of it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = directory / ("rescored_audit_" + stamp + ".json")
    record = {"schema": 1, "reason": reason, "written_at": time.time(),
              "kind": "rescored_audit", "audit_hash": _digest(payload), "payload": payload}
    from ..core.evidence import atomic_json
    atomic_json(path, record)
    return path


def load_snapshot(directory: Path) -> dict:
    path = Path(directory) / DECISION_FILE
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
