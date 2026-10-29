"""The one writer of a candidate's state.

Every transition goes through :class:`Controller`; the agents return results and
never edit the record. A record is bound to the source snapshot's digest, so a
change of source is a new version rather than an overwrite, and a finding has a
stable id with an explicit resolution instead of being inherited by luck.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..core.evidence import atomic_json
from . import states

RECORD_NAME = "v2_record.json"


def _digest_tree(tree: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(tree).rglob("*.py")):
        relative = path.relative_to(tree)
        if any(part in ("logs", "results", "dataset", "__pycache__", "ckpt")
               for part in relative.parts):
            continue
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


class Controller:
    """One candidate, one state, one writer."""

    def __init__(self, candidate_dir: Path):
        self.directory = Path(candidate_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / RECORD_NAME

    # --- the record ------------------------------------------------------
    def load(self) -> dict:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"schema": "v2", "state": None, "version": 0, "history": [],
                "findings": [], "tests": {}, "review": {}, "repairs": 0}

    def save(self, record: dict) -> None:
        atomic_json(self.path, record)

    # --- the transitions -------------------------------------------------
    def register(self, tree: Path, *, note: str = "") -> dict:
        """Freeze a source snapshot as version N and move to IMPLEMENTED."""
        record = self.load()
        digest = _digest_tree(Path(tree))
        current = record.get("state")
        if current and record.get("source_digest") == digest:
            return record                      # the same source, nothing to do
        if current:
            states.check(current, states.IMPLEMENTED)
        record["version"] = int(record.get("version", 0)) + 1
        record["source_digest"] = digest
        record["tree"] = str(Path(tree).resolve())
        record["findings"] = []                # a new version starts clean
        record["tests"] = {}
        record["review"] = {}
        record["note"] = note
        self._move(record, states.IMPLEMENTED, reason="source snapshot frozen")
        return record

    def record_tests(self, results: dict) -> dict:
        """Store the deterministic results and move according to them.

        A check that could not run is an INFRA_ERROR, never a TEST_FAILED: the
        candidate has not been judged, the verification failed to happen.
        """
        record = self.load()
        record["tests"] = results
        if not results.get("ran"):
            missing = [name for name, entry in (results.get("checks") or {}).items()
                       if entry.get("status") == "NOT_RUN"]
            self._move(record, states.INFRA_ERROR,
                       reason="a check could not run: " + ", ".join(missing))
            return record
        target = states.REVIEW_REQUIRED if results.get("passed") else states.TEST_FAILED
        self._move(record, target, reason="deterministic checks: " + (
            "all passed" if results.get("passed") else "a check failed"))
        return record

    def record_review(self, review: dict) -> dict:
        """Store one scientific review; a blocking finding moves to NEEDS_FIX."""
        record = self.load()
        record["review"] = review
        findings = []
        for index, finding in enumerate(review.get("findings") or [], 1):
            findings.append({
                "id": "F%d" % (len(record.get("findings") or []) + index),
                "severity": str(finding.get("severity") or "blocking"),
                "check": str(finding.get("check") or ""),
                "location": str(finding.get("location") or ""),
                "detail": str(finding.get("detail") or "")[:2000],
                "status": "open" if str(finding.get("severity")) == "blocking" else "noted",
            })
        record["findings"] = findings
        blocking = [finding for finding in findings if finding["status"] == "open"]
        verdict = str(review.get("verdict") or "").upper()
        # The reviewer's own verdict counts: a NEEDS_FIX with no finding is still a
        # refusal, and a PASS that did not answer every question is not a pass.
        unanswered = [check for check in (review.get("checks") or [])
                      if str(check.get("verdict") or "").upper() not in ("PASS", "NEEDS_FIX")]
        if verdict not in ("PASS", "NEEDS_FIX"):
            raise ValueError("the review carries no verdict")
        if verdict == "NEEDS_FIX" or blocking:
            target = states.NEEDS_FIX
        elif unanswered:
            raise ValueError("the review did not answer " + str(len(unanswered)) +
                             " of its questions")
        else:
            target = states.VERIFIED
        self._move(record, target, reason="scientific review: " + verdict)
        return record

    def fail(self, reason: str) -> dict:
        """An infrastructure fault: it is NOT a verdict on the candidate."""
        record = self.load()
        self._move(record, states.INFRA_ERROR, reason=str(reason)[:500])
        return record

    def reject(self, reason: str) -> dict:
        record = self.load()
        self._move(record, states.REJECTED, reason=str(reason)[:500])
        return record

    def repair(self, note: str = "") -> dict:
        """Count one repair; the budget is a hard stop for the automatic loop."""
        record = self.load()
        record["repairs"] = int(record.get("repairs", 0)) + 1
        if record["repairs"] > states.REPAIR_BUDGET:
            self.save(record)
            raise states.TransitionError(
                "the repair budget of " + str(states.REPAIR_BUDGET) +
                " is spent: the candidate needs a human decision")
        self._move(record, states.IMPLEMENTED, reason="repair: " + str(note)[:200])
        return record

    # --- internals -------------------------------------------------------
    def _move(self, record: dict, target: str, *, reason: str) -> None:
        current = record.get("state")
        if current and current != target:
            states.check(current, target)
        elif current == target:
            # Repeating a step is idempotent: the history keeps the record, and the
            # transition table stays strict for real moves.
            record.setdefault("history", []).append({
                "at": time.time(), "from": current, "to": target, "repeat": True,
                "version": record.get("version"), "reason": str(reason)[:500]})
            self.save(record)
            return
        record["state"] = target
        record.setdefault("history", []).append({
            "at": time.time(), "from": current, "to": target,
            "version": record.get("version"), "reason": str(reason)[:500]})
        self.save(record)
