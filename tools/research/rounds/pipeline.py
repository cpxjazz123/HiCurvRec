"""The round state machine of Issue #120.

One round is a directory with a state file. Every step is idempotent and
resumable: the state records what has run, what it produced and what is next, so
a round survives a restart, an agent failure or a cancelled job.

    ideation -> scoring -> selection -> implementation -> verification/repair
             -> stage2 queue -> automatic stage3

The state machine never decides anything by itself: the agents produce the
ideas, the scores, the implementations and the verification findings, and the
pure functions in `ideas` and `scoring` turn them into a selection. The machine
records every rejection, every revision and every failure signature.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..core.evidence import atomic_json, digest_file, read_json
from . import ideas as ideas_module
from . import scoring as scoring_module

#: The round lives under the issue that opened it, so #121's exploration is
#: filed with #121 and not with the framework issue that built the machinery.
DEFAULT_ISSUE = "issue120"
STATE_ROOT = Path("results/research") / DEFAULT_ISSUE / "rounds"
IDEAE_AGENTS = 3
SCORER_AGENTS = 3
VERIFIER_AGENTS = 3
#: How many repair rounds one implementation may have. The cap exists so a
#: candidate that cannot be repaired stops costing agents and is recorded with
#: its evidence instead of looping forever.
MAX_REVISIONS = 5


class RoundError(RuntimeError):
    """The round cannot advance as asked."""


def round_root(root: Path, issue: str = DEFAULT_ISSUE) -> Path:
    return Path(root) / "results" / "research" / issue / "rounds"


def find_round_directory(root: Path, round_id: str) -> Path:
    """Locate a round under whichever issue opened it."""
    base = Path(root) / "results" / "research"
    if not base.is_dir():
        raise RoundError("no research directory: " + str(base))
    for issue in sorted(base.iterdir()):
        candidate = issue / "rounds" / round_id
        if (candidate / "state.json").is_file():
            return candidate
    raise RoundError("unknown round: " + str(round_id))


def new_round(root: Path, round_id: str, *, objective: str = "",
              issue: str = DEFAULT_ISSUE) -> Path:
    """Create the round directory and its initial state."""
    directory = round_root(root, issue) / round_id
    if (directory / "state.json").is_file():
        raise RoundError("round already exists: " + round_id)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": 1,
        "round_id": round_id,
        "issue": issue,
        "objective": objective,
        "created_at": time.time(),
        "step": "ideation",
        "ideas": [],
        "ideas_rejected": [],
        "duplicates": [],
        "scores": {},
        "matrix": None,
        "selection": None,
        "implementations": [],
        "verification": {},
        "repairs": [],
        "blocked": [],
        "stage2_jobs": [],
        "stage3_jobs": [],
        "events": [{"event": "round_created", "at": time.time()}],
    }
    atomic_json(directory / "state.json", state)
    return directory


def load_state(directory: Path) -> dict:
    state = read_json(Path(directory) / "state.json")
    if not isinstance(state, dict):
        raise RoundError("round state is unreadable: " + str(directory))
    return state


def save_state(directory: Path, state: dict) -> None:
    state["updated_at"] = time.time()
    atomic_json(Path(directory) / "state.json", state)


def record_event(state: dict, event: str, **fields) -> None:
    state.setdefault("events", []).append({"event": event, "at": time.time(), **fields})


def apply_ideation(state: dict, per_agent: list) -> dict:
    """Merge the answers of the three ideation agents into the round's ideas."""
    try:
        merged = ideas_module.collect_ideas(per_agent)
    except ideas_module.IdeaError as exc:
        record_event(state, "ideation_rejected", reason=str(exc))
        raise RoundError(str(exc)) from exc
    state["ideas"] = merged
    state["ideas_rejected"] = [
        {"agent": agent, "reason": "duplicate or unusable"} for agent, _ in per_agent
        if not any(idea["origin_agent"] == str(agent) for idea in merged)]
    state["duplicates"] = [
        {"idea_id": idea["idea_id"], "also_proposed_by": idea["also_proposed_by"],
         "duplicate_count": idea["duplicate_count"]}
        for idea in merged if idea["duplicate_count"] > 1]
    state["step"] = "scoring"
    record_event(state, "ideation_complete", ideas=len(merged),
                 duplicates=len(state["duplicates"]))
    return state


def apply_scoring(state: dict, reports: list) -> dict:
    """Validate the scorers, build the matrix and select several ideas."""
    for report in reports:
        scoring_module.validate_report(report, scorer=str(report.get("scorer", "")))
    matrix = scoring_module.build_matrix(state["ideas"], reports)
    selection = scoring_module.select(matrix)
    state["scores"] = {str(report.get("scorer", "")): report for report in reports}
    state["matrix"] = matrix
    state["selection"] = selection
    state["step"] = "implementation"
    record_event(state, "scoring_complete", selected=len(selection["selected"]),
                 rejected=len(selection["rejected"]))
    return state


def implementation_record(idea: dict, report: dict, tree: Path) -> dict:
    """The provenance of one implementation, as the issue requires it."""
    tree = Path(tree)
    return {
        "idea_id": idea["idea_id"],
        "tree": str(tree),
        "tree_exists": tree.is_dir(),
        "proposal_sha256": digest_file(Path(tree) / "PROPOSAL.json")
        if (Path(tree) / "PROPOSAL.json").is_file() else None,
        "source_digest": _source_digest(tree),
        "changed_files": report.get("changed_files", []),
        "deviations": report.get("deviations", []),
        "assumptions": report.get("assumptions", []),
        "smoke": report.get("smoke", {}),
        "notes": report.get("notes", ""),
        "status": "implemented",
        "revision": 0,
        "verdicts": [],
    }


def _source_digest(tree: Path) -> str:
    """The digest of the tree's source files, for the provenance record."""
    import hashlib
    digest = hashlib.sha256()
    tree = Path(tree)
    for path in sorted(tree.rglob("*.py")):
        if "__pycache__" in path.relative_to(tree).parts:
            continue
        digest.update(str(path.relative_to(tree)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


NON_BLOCKING_SEVERITIES = frozenset({
    "minor", "nit", "style", "cosmetic", "suggestion", "advisory", "info",
    "informational", "non_blocking", "non-blocking", "optional",
})


def _is_blocking(severity) -> bool:
    """Whether a finding stops the implementation.

    Verifiers are free agents and do not share one vocabulary; a whitelist of
    "known blocking" words silently dropped the ones they did not happen to
    use, which turned three NEEDS_FIX verdicts into an approval. Everything
    that is not explicitly harmless therefore blocks.
    """
    text = str(severity or "").strip().lower().replace(" ", "_")
    if not text:
        return True
    return text not in NON_BLOCKING_SEVERITIES


def apply_verification(state: dict, implementation_id: str, reports: list) -> dict:
    """Merge the three verdicts of one implementation."""
    record = _implementation(state, implementation_id)
    if len(reports) != VERIFIER_AGENTS:
        raise RoundError("exactly three verifiers must report, got " + str(len(reports)))
    findings = []
    for report in reports:
        verdict = str(report.get("verdict", "")).upper()
        if verdict not in ("PASS", "NEEDS_FIX"):
            raise RoundError("a verifier returned an unknown verdict: " + verdict)
        reported = []
        for check in report.get("checks", []):
            for finding in check.get("findings", []):
                if _is_blocking(finding.get("severity")):
                    reported.append({"verifier": report.get("verifier", ""),
                                     "check": check.get("name", ""), **finding})
        if verdict == "NEEDS_FIX" and not reported:
            # The verifier asked for a correction without naming a blocking
            # finding. That is still a refusal to approve, so it must not be
            # recorded as an approval.
            reported.append({"verifier": report.get("verifier", ""),
                             "check": "verdict",
                             "severity": "blocking",
                             "location": "verification.json",
                             "detail": "verdict NEEDS_FIX without a stated blocking finding"})
        findings.extend(reported)
    record["verdicts"] = reports
    # The digest the reviewers saw is the tree's digest now: a stale one would make
    # the admission refuse a tree that was reviewed in exactly this state.
    record["source_digest"] = _source_digest(Path(record["tree"]))
    # A finding the framework itself raised (the output-path preflight) stays
    # in the list: the repair agent has to fix it as well.
    earlier = [finding for finding in (record.get("blocking_findings") or [])
               if finding.get("verifier") == "preflight"]
    record["blocking_findings"] = earlier + findings
    if findings:
        record["status"] = "needs_repair"
        signature = _failure_signature(findings)
        record["failure_signature"] = signature
        state.setdefault("repairs", []).append(
            {"idea_id": implementation_id, "revision": record["revision"],
             "signature": signature, "findings": findings})
        record_event(state, "verification_needs_repair", idea_id=implementation_id,
                     revision=record["revision"], signature=signature)
    else:
        record["status"] = "verified"
        record_event(state, "implementation_verified", idea_id=implementation_id,
                     revision=record["revision"])
    return state


def _failure_signature(findings: list) -> str:
    """A stable signature for a set of findings, so repeats are not new bugs."""
    import hashlib
    payload = sorted({str(finding.get("location", "")) + "|" + str(finding.get("detail", ""))[:120]
                      for finding in findings})
    return hashlib.sha256("||".join(payload).encode("utf-8")).hexdigest()[:16]


def apply_repair(state: dict, implementation_id: str, report: dict) -> dict:
    """Record a repair subagent's work and open the next review round."""
    record = _implementation(state, implementation_id)
    record["revision"] = int(record.get("revision", 0)) + 1
    record["status"] = "repaired"
    record["repairs"] = record.get("repairs", []) + [{
        "revision": record["revision"], "repairs": report.get("repairs", []),
        "remaining_issues": report.get("remaining_issues", []),
        "smoke": report.get("smoke", {})}]
    record["source_digest"] = _source_digest(Path(record["tree"]))
    record_event(state, "implementation_repaired", idea_id=implementation_id,
                 revision=record["revision"])
    return state


#: The kinds of stop a candidate can have. They are kept apart because they
#: mean different things: an agent that failed to answer says nothing about the
#: mechanism, while a defect in the code does.
AGENT_INFRA_ERROR = "AGENT_INFRA_ERROR"
IMPLEMENTATION_NEEDS_FIX = "IMPLEMENTATION_NEEDS_FIX"
RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
STAGE_FAILED = "STAGE2_3_FAILED"
BASELINE_UNAVAILABLE = "BASELINE_UNAVAILABLE"

#: How often an agent answer may be retried before the branch is recorded as
#: an infrastructure failure. A failed answer is retried, never a mechanism.
MAX_AGENT_RETRIES = 2


def mark_blocked(state: dict, implementation_id: str, reason: str,
                 *, kind: str = IMPLEMENTATION_NEEDS_FIX) -> dict:
    """Stop one implementation without stopping the round."""
    record = _implementation(state, implementation_id)
    record["status"] = "blocked"
    record["blocked_reason"] = reason
    record["blocked_kind"] = kind
    state.setdefault("blocked", []).append(
        {"idea_id": implementation_id, "reason": reason, "kind": kind,
         "revision": record.get("revision", 0),
         "signature": record.get("failure_signature")})
    record_event(state, "implementation_blocked", idea_id=implementation_id, reason=reason,
                 kind=kind)
    return state


def mark_agent_error(state: dict, implementation_id: str, error: str) -> dict:
    """Record an infrastructure failure of the agent, and keep the mechanism open.

    An agent that crashed, timed out or wrote an unusable answer says nothing
    about the idea, so the branch is retried instead of being blocked.
    """
    record = _implementation(state, implementation_id)
    attempts = int(record.get("agent_error_attempts", 0)) + 1
    record["agent_error_attempts"] = attempts
    record["agent_error"] = {"error": error, "attempts": attempts,
                              "kind": AGENT_INFRA_ERROR}
    if attempts > MAX_AGENT_RETRIES:
        record["status"] = "agent_error"
    else:
        record["status"] = "agent_error"
    state.setdefault("agent_errors", []).append(
        {"idea_id": implementation_id, "error": error, "attempts": attempts,
         "kind": AGENT_INFRA_ERROR})
    record_event(state, "agent_error", idea_id=implementation_id, error=error, attempts=attempts)
    return state


def mark_revision_capped(state: dict, implementation_id: str, reason: str) -> dict:
    """Stop a candidate that used up its repair rounds.

    The branch keeps its findings and its signature: a candidate that needs
    more than the allowed rounds is recorded, not repaired forever.
    """
    record = _implementation(state, implementation_id)
    record["status"] = "blocked"
    record["blocked_reason"] = reason
    record["blocked_kind"] = IMPLEMENTATION_NEEDS_FIX
    record["revision_capped"] = True
    state.setdefault("revision_capped", []).append(
        {"idea_id": implementation_id, "reason": reason,
         "revision": record.get("revision", 0),
         "signature": record.get("failure_signature")})
    record_event(state, "implementation_revision_capped", idea_id=implementation_id,
                 reason=reason, revision=record.get("revision", 0))
    return state


def mark_resource_blocked(state: dict, implementation_id: str, reason: str) -> dict:
    """Record a measured infeasibility, keeping the idea for an optimisation route."""
    record = _implementation(state, implementation_id)
    record["status"] = "resource_blocked"
    record["blocked_reason"] = reason
    record["blocked_kind"] = RESOURCE_BLOCKED
    state.setdefault("resource_blocked", []).append(
        {"idea_id": implementation_id, "reason": reason, "kind": RESOURCE_BLOCKED,
         "revision": record.get("revision", 0)})
    record_event(state, "implementation_resource_blocked", idea_id=implementation_id,
                 reason=reason)
    return state


def _implementation(state: dict, implementation_id: str) -> dict:
    for record in state.get("implementations", []):
        if record["idea_id"] == implementation_id:
            return record
    raise RoundError("unknown implementation: " + implementation_id)


def verified_implementations(state: dict) -> list:
    return [record for record in state.get("implementations", [])
            if record.get("status") == "verified"]


def selected_ideas(state: dict) -> list:
    selection = state.get("selection") or {}
    chosen = {entry["idea_id"] for entry in selection.get("selected", [])}
    return [idea for idea in state.get("ideas", []) if idea["idea_id"] in chosen]


def summary(state: dict) -> dict:
    """The status counters a caller polls for."""
    return {
        "round_id": state.get("round_id"),
        "step": state.get("step"),
        "ideas": len(state.get("ideas", [])),
        "duplicates": len(state.get("duplicates", [])),
        "selected": len((state.get("selection") or {}).get("selected", [])),
        "implementations": len(state.get("implementations", [])),
        "verified": len(verified_implementations(state)),
        "blocked": len(state.get("blocked", [])),
        "agent_errors": len(state.get("agent_errors", [])),
        "revision_capped": len(state.get("revision_capped", [])),
        "resource_blocked": len(state.get("resource_blocked", [])),
        "stage2_jobs": len(state.get("stage2_jobs", [])),
        "stage3_jobs": len(state.get("stage3_jobs", [])),
    }
