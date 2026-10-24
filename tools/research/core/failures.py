"""Structured failure registry inside the single Issue117 evidence space.

Issue #118 P1-D. Every reproducible engineering failure is recorded with what was
expected, what happened, how to reproduce it and which regression proves the fix.
Entries are deduplicated by a stable fingerprint over the failure's identity, not
by natural-language similarity, so the same root cause injected twice becomes one
entry with two occurrences instead of two stories.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

#: Registry location RELATIVE to the evidence root (``settings.STATE_ROOT``),
#: so a caller cannot end up with the state path duplicated inside itself.
REGISTRY = Path("failures") / "registry.json"

ERROR_CLASSES = (
    "GENERATION_OUTPUT", "SCHEMA", "MATH", "FALSIFIER", "TEST_AUTHOR", "DDP",
    "ENTRYPOINT", "HASH_BINDING", "PYTHON_RUNTIME", "INFRA_TRANSIENT",
    "SCIENTIFIC_FATAL",
)
STATUSES = ("open", "fix_pending", "fixed", "verified")

#: Which error class a preflight layer failure belongs to.
LAYER_CLASS = {
    "output": "GENERATION_OUTPUT",
    "contract": "SCHEMA",
    "math": "MATH",
    "falsifier": "FALSIFIER",
}

FINGERPRINT_FIELDS = ("stage", "error_class", "rule", "expected", "actual")


def fingerprint(entry: dict) -> str:
    payload = {key: entry.get(key) for key in FINGERPRINT_FIELDS}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load(path: Path) -> dict:
    if not path.is_file():
        return {"schema": 1, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": 1, "entries": {}}
    payload.setdefault("entries", {})
    return payload


def record(root: Path, entry: dict, *, repeat: bool = True) -> dict:
    """Insert or update one failure, keyed by its fingerprint.

    A repeat of the same fingerprint increments ``occurrences`` and appends the
    run that saw it; it never creates a second entry.
    """
    stage = str(entry.get("stage") or "")
    error_class = str(entry.get("error_class") or "")
    if error_class not in ERROR_CLASSES:
        raise ValueError("unknown error_class: " + error_class)
    if str(entry.get("status", "open")) not in STATUSES:
        raise ValueError("unknown status: " + str(entry.get("status")))
    path = Path(root) / REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load(path)
    key = fingerprint({"stage": stage, "error_class": error_class,
                       "rule": entry.get("rule"), "expected": entry.get("expected"),
                       "actual": entry.get("actual")})
    existing = payload["entries"].get(key)
    now = time.time()
    if existing is None:
        payload["entries"][key] = {
            "failure_id": key,
            "stage": stage,
            "error_class": error_class,
            "rule": entry.get("rule"),
            "expected": entry.get("expected"),
            "actual": entry.get("actual"),
            "reproducer": entry.get("reproducer"),
            "run_ids": [entry.get("run_id")] if entry.get("run_id") else [],
            "candidate_id": entry.get("candidate_id"),
            "source_sha256": entry.get("source_sha256"),
            "contract_sha256": entry.get("contract_sha256"),
            "test_sha256": entry.get("test_sha256"),
            "agent_session": entry.get("agent_session"),
            "fix_commit": entry.get("fix_commit"),
            "regression_test": entry.get("regression_test"),
            "verified_in": entry.get("verified_in"),
            "status": str(entry.get("status", "open")),
            "occurrences": 1,
            "first_seen": now,
            "last_seen": now,
        }
    else:
        # A backfilled ledger entry is not a recurrence: re-running the seed must
        # leave the counts alone, otherwise the ledger inflates itself.
        if repeat:
            existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
        existing["last_seen"] = now
        if entry.get("run_id") and entry["run_id"] not in existing["run_ids"]:
            existing["run_ids"].append(entry["run_id"])
        for field in ("candidate_id", "fix_commit", "regression_test", "verified_in",
                      "source_sha256", "contract_sha256", "test_sha256",
                      "agent_session", "reproducer"):
            if entry.get(field):
                existing[field] = entry[field]
        if entry.get("status"):
            existing["status"] = str(entry["status"])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload["entries"][key]


def load(root: Path) -> dict:
    return _load(Path(root) / REGISTRY)


def summary(root: Path) -> dict:
    """Counts that make a run's avoidable work visible.

    ``known_failure_recurrence`` counts entries seen more than once;
    ``verified`` counts entries with a regression test and a fix commit.
    """
    entries = load(root)["entries"].values()
    total = len(list(entries))
    entries = load(root)["entries"].values()
    recurrence = sum(1 for entry in entries if int(entry.get("occurrences", 1)) > 1)
    verified = sum(1 for entry in entries if entry.get("status") == "verified")
    return {"failures": total, "known_failure_recurrence": recurrence,
            "verified": verified}


def record_preflight(root: Path, report: dict, *, run_id: str = "",
                     candidate_id: str = "", source_sha256: str = "",
                     contract_sha256: str = "") -> list:
    """Record every decisive preflight diagnostic as a registry entry."""
    recorded = []
    for entry in report.get("diagnostics", []):
        if entry.get("verdict") is None:
            continue
        recorded.append(record(root, {
            "stage": entry.get("layer"),
            "error_class": LAYER_CLASS.get(entry.get("layer"), "SCHEMA"),
            "rule": entry.get("code"),
            "expected": entry.get("expected"),
            "actual": entry.get("actual"),
            "reproducer": entry.get("location"),
            "run_id": run_id,
            "candidate_id": candidate_id,
            "source_sha256": source_sha256,
            "contract_sha256": contract_sha256,
            "status": "open",
        }))
    return recorded
