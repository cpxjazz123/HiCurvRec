"""The round state machine of Issue #120.

One round is a directory with a state file. Every step is idempotent and
resumable: the state records what has run, what it produced and what is next, so
a round survives a restart, an agent failure or a cancelled job.

    ideation -> scoring -> selection -> implementation -> stage2 queue
             -> automatic stage3

The state machine never decides anything by itself: the agents produce the
ideas and the scores, the main agent writes the implementation, and the pure
functions in `ideas` and `scoring` turn them into a selection. Verification is
the framework's MACHINE admission (``tools/research/stage2/checks.py``) that
runs at launch, not an agent phase.
"""
from __future__ import annotations

import re

import json
import time
from pathlib import Path

from ..agents import policy
from ..core.evidence import atomic_json, digest_file, read_json
from . import ideas as ideas_module
from . import scoring as scoring_module

#: The round lives under the issue that opened it, so #121's exploration is
#: filed with #121 and not with the framework issue that built the machinery.
DEFAULT_ISSUE = "issue120"
STATE_ROOT = Path("results/research") / DEFAULT_ISSUE / "rounds"
#: How many independent ideation agents run per round. Five, so the proposal
#: space is wider than the three-agent rounds produced and the selection is
#: less likely to hinge on a single proposal family.
IDEA_AGENTS = 5
SCORER_AGENTS = 3
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


#: The pre-score evidence states (Issue #123).
PROBE_PENDING = "PROBE_PENDING"
PROBE_RUNNING = "PROBE_RUNNING"
PROBE_COMPLETE = "PROBE_COMPLETE"
PROBE_PARTIAL = "PROBE_PARTIAL"
PROBE_FAILED = "PROBE_FAILED"
NEEDS_EVIDENCE = "NEEDS_EVIDENCE"


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
        "probe_state": PROBE_PENDING,
        "ideas": [],
        "ideas_rejected": [],
        "duplicates": [],
        "scores": {},
        "matrix": None,
        "selection": None,
        "implementations": [],
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


def _merge_ideas(previous: list, fresh: list) -> list:
    """The proposals of a round, keeping the ones an earlier pass collected.

    A second pass asks the same agents for one more idea each, and the
    proposals of the first pass are not thrown away: a round that collects two
    ideas and then two more has four, not two.
    """
    merged = list(previous or [])
    known = {str(idea.get("idea_id")) for idea in merged}
    for idea in fresh or []:
        identifier = str(idea.get("idea_id"))
        if identifier in known:
            for existing in merged:
                if str(existing.get("idea_id")) == identifier:
                    existing.setdefault("also_proposed_by", [])
                    for other in idea.get("also_proposed_by", []):
                        if other not in existing["also_proposed_by"]:
                            existing["also_proposed_by"].append(other)
                    existing["duplicate_count"] = max(
                        int(existing.get("duplicate_count", 1)),
                        int(idea.get("duplicate_count", 1)))
            continue
        known.add(identifier)
        merged.append(idea)
    return merged


def apply_ideation(state: dict, per_agent: list, require_minimum: bool = True) -> dict:
    """Merge the answers of the ideation agents into the round's ideas."""
    try:
        merged = ideas_module.collect_ideas(per_agent, require_minimum=require_minimum)
    except ideas_module.IdeaError as exc:
        record_event(state, "ideation_rejected", reason=str(exc))
        raise RoundError(str(exc)) from exc
    state["ideas"] = _merge_ideas(state.get("ideas") or [], merged)
    state["ideas_rejected"] = [
        {"agent": agent, "reason": "duplicate or unusable"} for agent, _ in per_agent
        if not any(idea["origin_agent"] == str(agent) for idea in state["ideas"])]
    state["duplicates"] = [
        {"idea_id": idea["idea_id"], "also_proposed_by": idea["also_proposed_by"],
         "duplicate_count": idea["duplicate_count"]}
        for idea in state["ideas"] if idea.get("duplicate_count", 1) > 1]
    record_event(state, "ideation_complete", ideas=len(merged),
                 duplicates=len(state["duplicates"]))
    # The evidence probe runs before the first score (Issue #123): the reviewers
    # must see the measured evidence, not only the proposal's own claims.
    if state.get("probe_enabled", True):
        state["step"] = "probe"
        state["probe_state"] = PROBE_PENDING
    else:
        state["step"] = "scoring"
    return state


def apply_scoring(state: dict, reports: list, *, comparisons: list = None,
                  bundles: dict = None, directory: Path = None,
                  source_commit: str = None) -> dict:
    """Validate the reviews, weigh them, decide, and freeze the decision.

    The decision record is written once, here, and a later recalculation is an
    additive audit that never rewrites it.
    """
    from . import decision as decision_module

    evaluation = scoring_module.evaluate(state["ideas"], reports)
    selection = evaluation["selection"]
    state["scores"] = {str(report.get("reviewer") or report.get("scorer") or ""): report
                        for report in reports}
    state["rankings"] = state["scores"]
    state["matrix"] = evaluation["matrix"]
    state["selection"] = selection
    state["rubric_version"] = evaluation["rubric_version"]
    state["rubric_sha256"] = evaluation["rubric_sha256"]
    state["review_bundles"] = {identifier: (bundle or {}).get("sha256")
                               for identifier, bundle in (bundles or {}).items()}
    if directory is not None:
        snapshot = decision_module.build_snapshot(
            round_id=state.get("round_id", ""), evaluation=evaluation, reports=reports,
            bundles=bundles or {}, proposals=state["ideas"], source_commit=source_commit)
        existing = decision_module.load_snapshot(Path(directory))
        if existing and existing.get("decision_hash") != snapshot.get("decision_hash"):
            # The decision of a completed round is never rewritten: a new review
            # (of a revised proposal, or under a new rubric) is recorded beside it
            # and linked to the decision it supersedes. The next revision number is
            # read from the highest decision file on disk, not from the original
            # one, so a second re-review never lands on an existing file.
            latest = decision_module.latest_snapshot(Path(directory)) or existing
            revision = int(latest.get("revision", 1)) + 1
            snapshot["revision"] = revision
            snapshot["supersedes_decision_hash"] = latest.get("decision_hash")
            snapshot["decision_hash"] = decision_module.decision_hash(snapshot)
            path = decision_module.write_revision(Path(directory), snapshot, revision)
            state["decision_hash"] = snapshot["decision_hash"]
            state["decision_at"] = snapshot["decision_at"]
            state["decision_revision"] = revision
            state["decision_supersedes"] = snapshot["supersedes_decision_hash"]
            state["decision_file"] = str(path)
        else:
            decision_module.write_snapshot(Path(directory), snapshot)
            state["decision_hash"] = snapshot["decision_hash"]
            state["decision_at"] = snapshot["decision_at"]
    # NO_QUALIFIED_IDEA stays a visible state: the round asks for refined
    # ideation and evidence instead of inventing a winner.
    if selection["status"] == scoring_module.STATUS_NO_QUALIFIED:
        state["step"] = "ideation"
        record_event(state, "no_qualified_idea",
                     rejected=len(selection["rejected"]))
        return state
    state["step"] = "implementation"
    record_event(state, "scoring_complete", selected=len(selection["selected"]),
                 rejected=len(selection["rejected"]),
                 decision_hash=state.get("decision_hash"))
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
        # The provenance of the copy the implementation agent worked in: which
        # baseline it came from, and the digest of that baseline and of the copy
        # before the agent touched it.
        "baseline": report.get("baseline"),
        "baseline_digest": report.get("baseline_digest"),
        "candidate_digest_before_agent": report.get("candidate_digest_before_agent"),
        "changed_files": report.get("changed_files", []),
        "deviations": report.get("deviations", []),
        "assumptions": report.get("assumptions", []),
        "smoke": report.get("smoke", {}),
        "notes": report.get("notes", ""),
        "status": "implemented",
        "revision": 0,
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


#: Findings the framework observed itself, by source. A review never erases one.
FRAMEWORK_FINDING_SOURCES = frozenset({"preflight", "smoke", "runtime"})

NON_BLOCKING_SEVERITIES = frozenset({
    "minor", "nit", "style", "cosmetic", "suggestion", "advisory", "info",
    "informational", "non_blocking", "non-blocking", "optional", "note",
    "notes", "none", "harmless", "n/a", "na", "n.a.", "nil", "no_finding",
    "not_applicable", "not applicable", "nothing", "no", "pass", "ok", "-", "--", "n.a",
})

#: The words that mean the finding stops the work.
BLOCKING_SEVERITIES = frozenset({
    "blocking", "blocker", "block", "critical", "correctness", "severe",
    "fatal", "error", "high", "major", "unsafe", "broken",
})


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


def _implementation(state: dict, implementation_id: str) -> dict:
    for record in state.get("implementations", []):
        if record["idea_id"] == implementation_id:
            return record
    raise RoundError("unknown implementation: " + implementation_id)


def register_implementation(state: dict, implementation_id: str, report: dict) -> dict:
    """Record the implementation the MAIN AGENT wrote.

    There is no implementation agent and no review agent: the main agent writes
    the mechanism, registers the changed files, the smoke it ran and the digest
    of the tree, and the framework then runs the MACHINE admission
    (``tools/research/stage2/checks.py``) against that tree before any GPU work.
    The registration is therefore the last agent step of a round.
    """
    record = _implementation(state, implementation_id)
    tree = Path(record["tree"])
    if not tree.is_dir():
        raise RoundError("the candidate tree does not exist: " + str(tree))
    digest = _source_digest(tree)
    claimed = str(report.get("source_digest") or "")
    if claimed and claimed != digest:
        raise RoundError("the registered implementation claims digest " + claimed[:16] +
                         " but the tree on disk hashes to " + digest[:16])
    # Machine check, fail closed: a tree that would write outside its own result
    # directory is refused here, not reviewed later.
    from . import baseline as baseline_module
    siblings = [other["tree"] for other in state.get("implementations", [])
                if other["idea_id"] != implementation_id]
    try:
        record["output_paths"] = baseline_module.guard_output_paths(
            tree, other_candidates=siblings)
    except baseline_module.BaselineError as exc:
        raise RoundError("the candidate writes outside its own result directory: " +
                         str(exc)) from exc
    if record.get("source_digest"):
        record["revision"] = int(record.get("revision", 0)) + 1
    record["changed_files"] = list(report.get("changed_files", []))
    record["deviations"] = list(report.get("deviations", []))
    record["assumptions"] = list(report.get("assumptions", []))
    record["smoke"] = report.get("smoke", {})
    record["source_digest"] = digest
    record["status"] = "implemented"
    record["implemented_by"] = "main_agent"
    record["blocking_findings"] = []
    state["step"] = "stage2"
    record_event(state, "implementation_registered", idea_id=implementation_id,
                 revision=int(record.get("revision", 0)),
                 changed_files=len(record["changed_files"]))
    return state


def accept_implementation(state: dict, implementation_id: str, note: str = "") -> dict:
    """The main agent accepts the implementation, so it may be trained."""
    record = _implementation(state, implementation_id)
    if str(record.get("status")) != "implemented":
        raise RoundError("only an implemented candidate can be accepted; " +
                         implementation_id + " is " + str(record.get("status")))
    record["status"] = "accepted"
    record["accepted"] = True
    record["acceptance_note"] = note
    record_event(state, "implementation_accepted", idea_id=implementation_id, note=note)
    return state


def accepted_implementations(state: dict) -> list:
    """The candidates the main agent accepted: the only ones Stage 2 may run."""
    return [record for record in state.get("implementations", [])
            if record.get("status") == "accepted" and record.get("accepted")]


def close_proposal(state: dict, implementation_id: str, *, reason: str,
                   evidence: dict = None) -> dict:
    """End a candidate because its PROPOSAL is defective, not its code.

    A proposal-level defect (a control law that saturates, a coupling that is inert
    over the operating range) cannot be repaired in the tree: it needs a new
    proposal. The record keeps the evidence and a terminal status the scheduler
    honours, so the round stops waiting for a repair that will never come and the
    next round can start.
    """
    record = _implementation(state, implementation_id)
    record["status"] = "proposal_rejected"
    record["closed_reason"] = str(reason)
    record["closed_at"] = time.time()
    record["close_evidence"] = dict(evidence or {})
    record["blocking_findings"] = []
    record["accepted"] = False
    record_event(state, "proposal_closed", idea_id=implementation_id,
                 reason=str(reason)[:200], evidence=str((evidence or {}).get("file", "")))
    state["step"] = "stage2"
    return state


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
        "accepted": len(accepted_implementations(state)),
        "stage2_jobs": len(state.get("stage2_jobs", [])),
        "stage3_jobs": len(state.get("stage3_jobs", [])),
    }
