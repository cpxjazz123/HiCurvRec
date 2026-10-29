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
#: How many independent verifiers read one implementation revision. Five, so a
#: single reviewer's subjective reading cannot decide the candidate on its own:
#: the findings carry how many reviewers raised them, and the operator (the main
#: agent) decides what to do with them.
VERIFIER_AGENTS = 5
#: How many times the verifiers may reject one implementation. The cap is a cost
#: guard, not a scientific gate: a candidate that cannot be repaired stops
#: costing agents and is recorded with its evidence instead of looping forever.
#: It counts the **review rounds**, not the source revisions: a candidate that
#: needed a performance fix, a device fix and a proposal alignment is not a
#: candidate whose mechanism was rejected nine times. The operator raised it
#: from 5 to 8 after the first round showed three candidates one or two concrete
#: defects short of a working trainer.
MAX_REVISIONS = 12

#: How many times the verifiers may reject one implementation. The cap uses
#: this counter; the revision counter only decides whether the next review runs
#: with a fresh agent.
MAX_REVIEW_ROUNDS = 8


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


def _finding_role(finding: dict) -> str:
    """The role a finding belongs to.

    New findings carry their role. A finding from before the roles existed carries
    only the reviewer's task name, whose number is the role's position in the full
    role list -- the same numbering the dispatch uses.
    """
    role = str(finding.get("role") or "")
    if role:
        return role
    verifier = str(finding.get("verifier") or "")
    for index in range(1, 32):
        if verifier.endswith("_" + str(index)):
            from . import agents as agent_contracts
            roles = agent_contracts.VERIFIER_ROLES
            if index <= len(roles):
                return roles[index - 1]["id"]
    return ""


def _is_blocking(severity) -> bool:
    """Whether a finding stops the implementation.

    Verifiers are free agents and do not share one vocabulary; a whitelist of
    "known blocking" words silently dropped the ones they did not happen to
    use, which turned three NEEDS_FIX verdicts into an approval. Everything
    that is not explicitly harmless therefore blocks.
    """
    text = str(severity or "").strip().lower()
    if not text:
        return True
    # Reviewers write sentences, not single words: "minor when it does not stop
    # the work" is a minor finding, and the first word says which it is.
    first = re.split(r"[\s,;:.()\[\]/]+", text, maxsplit=1)[0]
    first = first.replace("-", "_")
    if first in BLOCKING_SEVERITIES:
        return True
    if first in NON_BLOCKING_SEVERITIES:
        return False
    if text.replace(" ", "_") in NON_BLOCKING_SEVERITIES:
        return False
    return True


def apply_verification(state: dict, implementation_id: str, reports: list) -> dict:
    """Merge the verdicts of one implementation.

    One reviewer that finished is enough: a reviewer that ran out of its budget
    does not void the review, it only means the round has fewer opinions than it
    planned. The record says how many reported and which ones are missing, so
    the decision still shows what the round actually had.
    """
    record = _implementation(state, implementation_id)
    if len(reports) < 1:
        raise RoundError("a review needs at least one verifier's report, got none")
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
                                     "role": report.get("role", ""),
                                     "check": check.get("name", ""), **finding})
        if verdict == "NEEDS_FIX" and not reported:
            # The verifier asked for a correction without naming a blocking
            # finding. That is still a refusal to approve, so it must not be
            # recorded as an approval.
            reported.append({"verifier": report.get("verifier", ""),
                             "role": report.get("role", ""),
                             "check": "verdict",
                             "severity": "blocking",
                             "location": "verification.json",
                             "detail": "verdict NEEDS_FIX without a stated blocking finding"})
        findings.extend(reported)
    record["verdicts"] = reports
    record["reviewers_reported"] = len(reports)
    record["reviewers_expected"] = VERIFIER_AGENTS
    missing = sorted({str(name) for report in reports
                      for name in (report.get("missing_reviewers") or [])})
    record["reviewers_missing"] = missing
    # The boundary the reviewers read the candidate with. All of them must have
    # run under the same one, and the record keeps it, so an acceptance can
    # refuse a review that was produced under older rules.
    boundaries = {str(report.get("boundary_version") or "") for report in reports}
    if len(boundaries) != 1 or "" in boundaries:
        raise RoundError("the verifiers did not run under one filesystem boundary: " +
                         repr(sorted(boundaries)))
    record["boundary_version"] = boundaries.pop()
    record["boundary_sha256"] = str(reports[0].get("boundary_sha256") or "")
    # Where the stage records of this review are: the main agent reads them
    # from those files, and they are what the task actually delivered before
    # its deadline.
    record["progress"] = [
        {"verifier": str(report.get("verifier", "")),
         "file": str(report.get("progress_file", "")),
         "records": int(report.get("progress_count", 0)),
         "deadline_seconds": report.get("deadline_seconds")}
        for report in reports]
    # How many reviewers raised each finding: with five verifiers a single
    # objection and a shared one are different things, and the main agent has to
    # see the difference.
    agreement: dict[str, int] = {}
    for finding in findings:
        key = str(finding.get("location", "")) + "|" + str(finding.get("detail", ""))[:120]
        agreement[key] = agreement.get(key, 0) + 1
    record["finding_agreement"] = agreement
    # The digest the reviewers saw is the tree's digest now: a stale one would make
    # the admission refuse a tree that was reviewed in exactly this state.
    record["source_digest"] = _source_digest(Path(record["tree"]))
    # A finding the framework itself raised (the output-path preflight) stays
    # in the list: the repair agent has to fix it as well.
    # A finding the framework itself observed (the output-path preflight or the
    # Stage 2 smoke) is not a reviewer's opinion: a later review does not erase
    # it. Only a source change clears it, because then the code is different.
    earlier = [finding for finding in (record.get("blocking_findings") or [])
               if finding.get("verifier") in FRAMEWORK_FINDING_SOURCES]
    # A reviewer's finding stays open until the SAME role re-reviews the tree. A
    # PASS from another role says nothing about the area that complained, and
    # treating it as a closure is how "verified with no blocking findings" could be
    # reported while an old blocker was never re-checked.
    responding = {str(report.get("role") or "") for report in reports}
    carried = [finding for finding in (record.get("blocking_findings") or [])
               if finding.get("verifier") not in FRAMEWORK_FINDING_SOURCES
               and _finding_role(finding) not in responding]
    record["blocking_findings"] = earlier + carried + findings
    if carried:
        record["carried_findings"] = len(carried)
    # A failed output-path preflight blocks just as a reviewer's finding does.
    findings = record["blocking_findings"]
    if findings:
        record["status"] = "needs_repair"
        record["review_rounds"] = int(record.get("review_rounds", 0)) + 1
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


def apply_repair_plan(state: dict, implementation_id: str, plan: dict) -> dict:
    """Record a repair PLAN and wait for the operator to apply it.

    The repair agent does not touch the code: it reads the findings and the
    tree and writes the change that would fix each finding. The operator (the
    main agent) applies the plan to the tree and then registers the execution
    with :func:`apply_repair_execution`, which is what opens the next review
    round. Until that happens the tree is unchanged, so the revision and the
    digest stay as they were and no review may be scheduled.
    """
    record = _implementation(state, implementation_id)
    record["status"] = "repair_planned"
    record.setdefault("repair_plans", []).append({
        "revision": int(record.get("revision", 0)),
        "plans": plan.get("plans", []),
        "remaining_issues": plan.get("remaining_issues", []),
        "notes": plan.get("notes", ""),
        "source_digest_at_plan": _source_digest(Path(record["tree"])),
    })
    record_event(state, "repair_plan_ready", idea_id=implementation_id,
                 revision=int(record.get("revision", 0)),
                 plans=len(plan.get("plans", [])))
    return state


def apply_repair_execution(state: dict, implementation_id: str,
                            execution: dict) -> dict:
    """Record that the operator applied a repair plan, and reopen the review.

    ``execution`` carries what was changed and, optionally, the digest the
    operator expects. The digest on disk wins: it is what the next reviewer
    will read, and a mismatch with the tree is a closed failure rather than a
    record that claims a change nobody made.
    """
    record = _implementation(state, implementation_id)
    tree = Path(record["tree"])
    digest = _source_digest(tree)
    expected = str(execution.get("source_digest") or "")
    if expected and expected != digest:
        raise RoundError(
            "the recorded repair execution claims digest " + expected[:16] +
            " but the tree on disk hashes to " + digest[:16])
    record["revision"] = int(record.get("revision", 0)) + 1
    record["status"] = "repaired"
    record["repairs"] = record.get("repairs", []) + [{
        "revision": record["revision"],
        "changes": execution.get("changes", []),
        "notes": execution.get("notes", ""),
        "applied_by": execution.get("applied_by", "operator"),
    }]
    record["source_digest"] = digest
    # A runtime finding described the code before the repair: the repaired tree
    # is checked again by the runtime itself (the Stage 2 smoke), so the old
    # observation is archived rather than carried forward.
    archived = [finding for finding in (record.get("blocking_findings") or [])
                if finding.get("verifier") in FRAMEWORK_FINDING_SOURCES]
    if archived:
        record.setdefault("archived_findings", []).extend(
            {**finding, "archived_at_revision": record["revision"]} for finding in archived)
        record["blocking_findings"] = [
            finding for finding in record["blocking_findings"]
            if finding.get("verifier") not in FRAMEWORK_FINDING_SOURCES]
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


def mark_agent_error(state: dict, implementation_id: str, error: str,
                     details: dict | None = None) -> dict:
    """Record an infrastructure failure of the agent, and keep the mechanism open.

    An agent that crashed, timed out or wrote an unusable answer says nothing
    about the idea, so the branch is retried instead of being blocked.
    """
    record = _implementation(state, implementation_id)
    attempts = int(record.get("agent_error_attempts", 0)) + 1
    record["agent_error_attempts"] = attempts
    record["agent_error"] = {"error": error, "attempts": attempts,
                              "kind": AGENT_INFRA_ERROR}
    if details:
        # What did arrive before the fault: the verdicts of the reviewers that
        # answered, and the class of the fault, so that a missing reviewer is
        # not read as a verdict on the mechanism.
        record["agent_error"].update(details)
    if attempts > MAX_AGENT_RETRIES:
        record["status"] = "agent_error"
    else:
        record["status"] = "agent_error"
    state.setdefault("agent_errors", []).append(
        {"idea_id": implementation_id, "error": error, "attempts": attempts,
         "kind": AGENT_INFRA_ERROR})
    record_event(state, "agent_error", idea_id=implementation_id, error=error, attempts=attempts)
    return state


def record_framework_finding(state: dict, implementation_id: str, *, source: str,
                             detail: str, correction: str, check: str = "runtime") -> dict:
    """Record a defect the framework itself observed.

    The output-path preflight and the Stage 2 smoke see things a code review
    cannot: a path that only breaks after promotion, or a device mismatch that
    only appears when the trainer runs. Such a finding invalidates the review.
    """
    record = _implementation(state, implementation_id)
    findings = [finding for finding in (record.get("blocking_findings") or [])
                if finding.get("verifier") != source]
    findings.insert(0, {"verifier": source, "check": check, "severity": "blocking",
                        "location": "train_rqvae.py", "detail": detail,
                        "reproduction": detail, "correction": correction})
    record["blocking_findings"] = findings
    record["status"] = "needs_repair"
    record["failure_signature"] = _failure_signature(findings)
    record_event(state, "framework_finding", idea_id=implementation_id, source=source,
                 detail=detail[:200])
    return state


def recheck_output_paths(state: dict, implementation_id: str,
                          repo_root: Path = None, skip_if_running=()) -> dict:
    """Re-run the output-path preflight after a repair.

    A repaired tree may have moved its outputs to the right place; the earlier
    preflight finding is replaced by the current verdict so it cannot block a
    candidate that is now isolated.
    """
    from . import baseline as baseline_module

    record = _implementation(state, implementation_id)
    if implementation_id in set(skip_if_running or ()):
        # A candidate that is already training keeps its verdict: its record is
        # not rewritten under a running job.
        return state
    tree = Path(record["tree"])
    siblings = [other["tree"] for other in state.get("implementations", [])
                if other["idea_id"] != implementation_id]
    findings = [finding for finding in (record.get("blocking_findings") or [])
                if finding.get("verifier") != "preflight"]
    try:
        record["output_paths"] = baseline_module.guard_output_paths(
            tree, other_candidates=siblings, repo_root=repo_root)
        record["output_preflight"] = "passed"
    except baseline_module.BaselineError as exc:
        findings.insert(0, {"verifier": "preflight", "check": "output isolation",
                            "severity": "blocking", "location": "train_rqvae.py",
                            "detail": str(exc),
                            "reproduction": "read the trainer's output constants",
                            "correction": "write only into the candidate's own result "
                                          "directory results/stage2_RQ-VAE/TIGER_<idea>/"})
        record["output_preflight"] = "failed"
    record["blocking_findings"] = findings
    record_event(state, "output_paths_rechecked", idea_id=implementation_id,
                 verdict=record["output_preflight"])
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


def reopen_capped(state: dict, implementation_id: str, reason: str) -> dict:
    """Give a capped candidate another repair budget.

    The cap is a cost guard; the operator can raise it for candidates whose
    findings are concrete. The cap and the reason stay in the record.
    """
    record = _implementation(state, implementation_id)
    record["status"] = "needs_repair"
    record["revision_capped"] = False
    record["reopened_after_cap"] = record.get("reopened_after_cap", 0) + 1
    state["revision_capped"] = [entry for entry in state.get("revision_capped", [])
                                if entry.get("idea_id") != implementation_id]
    record_event(state, "implementation_reopened", idea_id=implementation_id, reason=reason)
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
    """The candidates the main agent has ACCEPTED.

    Five verifiers reading an implementation is a report, not a decision. The
    decision is the main agent's: it reads the verdicts and the findings, and
    only an accepted candidate may be dispatched to Stage 2.

    The blocking findings are checked here as well, so that a candidate cannot
    reach Stage 2 by a hand-set flag: the rule is that a blocking finding in
    the last review has to be fixed and reviewed again first.
    """
    return [record for record in state.get("implementations", [])
            if record.get("status") == "verified" and record.get("accepted")
            and not (record.get("blocking_findings") or [])]


def register_implementation(state: dict, implementation_id: str, report: dict) -> dict:
    """Record the implementation the MAIN AGENT wrote.

    There is no implementation agent: the main agent writes the mechanism in the
    candidate tree and registers what it did here. The record then carries the
    changed files, the smoke it ran and the digest of the tree, and the verifiers
    read exactly that tree.
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
    record["changed_files"] = list(report.get("changed_files", []))
    record["deviations"] = list(report.get("deviations", []))
    record["assumptions"] = list(report.get("assumptions", []))
    record["smoke"] = report.get("smoke", {})
    record["source_digest"] = digest
    record["status"] = "implemented"
    record["implemented_by"] = "main_agent"
    # A registration that follows a review is a new revision: the reviewers'
    # workspaces, the snapshot and the diff all key on it, so a fix must not be
    # filed under the revision number of the tree it replaces.
    if record.get("verdicts"):
        record["revision"] = int(record.get("revision", 0)) + 1
    # A fresh implementation reopens the review, whatever the previous pass left
    # behind: the tree changed, so an earlier verdict and its fingerprint do not
    # describe it. The retry counter of an earlier infrastructure failure is
    # reset with it, because the registration is a new submission and not
    # another attempt of the failed one.
    record["agent_error_attempts"] = 0
    record["fingerprint"] = ""
    state["step"] = "verification"
    # The output isolation is checked again on the tree as it now is: the
    # finding of an earlier pass described an older tree, and a candidate that
    # fixed it must not carry the stale finding into the next review.
    _refresh_output_isolation(state, record)
    record_event(state, "implementation_registered", idea_id=implementation_id,
                 revision=int(record.get("revision", 0)),
                 changed_files=len(record["changed_files"]))
    return state


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
    pending = [item for item in state.get("implementations", [])
               if item.get("status") in ("implemented", "repaired", "needs_repair",
                                         "repair_planned", "verified",
                                         "awaiting_implementation")]
    state["step"] = "verification" if pending else "stage2"
    return state


def _refresh_output_isolation(state: dict, record: dict) -> None:
    """Re-run the output-path guard on the tree as it is now.

    The guard is the framework's own finding, not a reviewer's: it says whether
    the candidate would write outside its own result directory. A tree that was
    rewritten since the last check is checked again, and a finding that no
    longer holds is dropped.
    """
    from . import baseline as baseline_module

    tree = Path(record["tree"])
    siblings = [other["tree"] for other in state.get("implementations", [])
                if other["idea_id"] != record["idea_id"]]
    try:
        record["output_paths"] = baseline_module.guard_output_paths(
            tree, other_candidates=siblings)
    except baseline_module.BaselineError as exc:
        record["blocking_findings"] = [{
            "verifier": "preflight", "check": "output isolation",
            "severity": "blocking", "location": "train_rqvae.py",
            "detail": str(exc), "reproduction": "read the trainer's output constants",
            "correction": "write only into the candidate's own result directory"}]
        record["status"] = "needs_repair"
        record["failure_signature"] = _failure_signature(record["blocking_findings"])
        return
    record["blocking_findings"] = [
        finding for finding in (record.get("blocking_findings") or [])
        if finding.get("verifier") != "preflight"]


def accept_verification(state: dict, implementation_id: str, note: str = "") -> dict:
    """The main agent accepts a candidate, and it may then be trained.

    **A candidate with a blocking finding in its last review cannot be
    accepted.** The rule is hard: the finding is fixed, the implementation is
    reviewed again, and only a review without a blocking finding opens the way
    to Stage 2. The main agent decides what a review means and how the work is
    done; it does not get to accept a known defect.

    The main agent still has the whole decision: it can order a repair plan
    (:func:`apply_repair_plan` after the planner ran), apply the plan itself
    (:func:`apply_repair_execution`) and let the verifiers read the result
    again. What it cannot do is train a candidate whose last review says there
    is a blocking defect.
    """
    record = _implementation(state, implementation_id)
    status = str(record.get("status"))
    findings = record.get("blocking_findings") or []
    if findings:
        raise RoundError(
            implementation_id + " still carries " + str(len(findings)) +
            " blocking finding(s) from its last review and cannot be accepted: " +
            "; ".join(str(finding.get("detail", ""))[:160] for finding in findings[:3]))
    if status != "verified":
        raise RoundError("only a verified candidate can be accepted; " +
                         implementation_id + " is " + status)
    # A review is evidence about the candidate under the boundary it was read
    # with. When the boundary changed, the old verdicts describe a view of the
    # files the reviewers no longer have, so they are not accepted as current.
    produced_under = str(record.get("boundary_version") or "")
    if produced_under != policy.BOUNDARY_VERSION:
        raise RoundError(
            implementation_id + " was reviewed under boundary '" +
            (produced_under or "none") + "' and the current policy is '" +
            policy.BOUNDARY_VERSION + "': re-verify it under the current "
            "boundary before accepting it")
    record["accepted"] = True
    record["acceptance_note"] = note
    record_event(state, "verification_accepted", idea_id=implementation_id,
                 note=note, boundary_version=policy.BOUNDARY_VERSION)
    return state


def decline_verification(state: dict, implementation_id: str, reason: str) -> dict:
    """The main agent refuses a candidate the reviewers reported on."""
    record = _implementation(state, implementation_id)
    record["accepted"] = False
    record["decline_reason"] = reason
    record_event(state, "verification_declined", idea_id=implementation_id,
                 reason=reason)
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
        "verified": len(verified_implementations(state)),
        "blocked": len(state.get("blocked", [])),
        "agent_errors": len(state.get("agent_errors", [])),
        "revision_capped": len(state.get("revision_capped", [])),
        "resource_blocked": len(state.get("resource_blocked", [])),
        "stage2_jobs": len(state.get("stage2_jobs", [])),
        "stage3_jobs": len(state.get("stage3_jobs", [])),
    }
