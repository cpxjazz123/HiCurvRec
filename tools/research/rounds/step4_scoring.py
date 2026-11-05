"""Step 4 - the three scoring agents (`score`), the bounded
rebuttal and the ranking that selects one mechanism.

Moved out of worker.py by the split; the bodies are unchanged.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

from .. import settings
from ..agents.runner import (READ_ONLY_ROLES, boundary_fingerprint, run_agent)
from ..agents import policy, progress_server, runner
from ..core.evidence import atomic_json, digest_file, read_json
from . import agents as agent_contracts
from . import baseline as baseline_module
from . import evals as evals_module
from . import evidence as evidence_module
from . import ideas as ideas_module
from . import literature as literature_module
from . import math_checks as math_checks_module
from . import module_differential
from . import pipeline as pipeline_module
from . import plaintext
from . import probes as probes_module
from . import scoring as scoring_module
from . import spec_tests as spec_tests_module
from . import specification as specification_module


from .agent_calls import (  # noqa: F401  (the shared call)
    SCORERS,
    AgentPhaseIncomplete, RoundIncomplete,
    _run_agent, _run_agents_concurrently, _round_roots,
    _workspace, geometry_contract, role_of_task,
)



def rank_one(root: Path, payload: str, evidence_text: str, index: int) -> dict:
    """One reviewer over the frozen evidence bundle.

    Same unit as the concurrent runner: the flow dispatches it concurrently so
    every path asks the reviewers exactly the same question.
    """
    task_id = "scoring_" + str(index)
    prompt = (agent_contracts.RANK_PROMPT.format(
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY,
        rubric_version=scoring_module.RUBRIC["version"],
        proposals=payload,
        evidence=evidence_text))
    report = _rank_with_retry(root, task_id, prompt)
    report.setdefault("reviewer", task_id)
    report["rubric_version"] = scoring_module.RUBRIC["version"]
    report["rubric_sha256"] = scoring_module.RUBRIC["sha256"]
    return report


def scoring_inputs(root: Path, ideas: list, *, repo: Path = None,
                   state: dict = None) -> tuple:
    """The frozen bundle, the proposal text and the evidence text of a review.

    One source of truth: the concurrent runner and the flow both build the
    reviewers' input here, so a reviewer sees the same question in either path.
    """
    repo = Path(repo) if repo is not None else Path(root)
    frozen = (state or {}).get("evidence_bundles") or {}
    bundle = {idea["idea_id"]: frozen.get(idea["idea_id"])
              or evidence_module.build_bundle(idea, root=repo)
              for idea in ideas}
    payload = plaintext.render_ideas(ideas)
    evidence_text = "\n\n".join(
        "### " + identifier + "\n" + evidence_module.render_bundle(item)
        for identifier, item in bundle.items())
    return bundle, payload, evidence_text


def rank_order(ideas: list, reports: list) -> dict:
    """The reviewers' aggregate order, the selection input."""
    return _rank_order(ideas, reports)


def apply_bounded_revision(directory: Path, agent_root: Path, repo: Path,
                            state: dict, reports: list, outcome: dict):
    """The one bounded revision the reviewers may buy (public entry)."""
    return _one_bounded_revision(directory, agent_root, repo, state, reports, outcome)


def run_scoring(root: Path, ideas: list, *, repo: Path = None,
                state: dict = None) -> dict:
    """Three independent reviewers over the frozen evidence bundle.

    The bundle is assembled once and handed unchanged to all three, so their
    reviews are independent of each other but not of the evidence. An objection
    that names a mathematical identity, a cancellation, leakage or a dimension
    mismatch buys the proposal ONE bounded revision, and the revision is
    reviewed afresh under a new proposal hash.
    """
    repo = Path(repo) if repo is not None else Path(root)
    bundle, payload, evidence_text = scoring_inputs(root, ideas, repo=repo, state=state)
    def rank(index):
        return rank_one(root, payload, evidence_text, index)

    try:
        reports = _run_agents_concurrently(
            [tuple([index]) for index in range(1, SCORERS + 1)], rank)
    except AgentPhaseIncomplete as exc:
        # A reviewer that ran out of its budget did not rank. The rankings that
        # did arrive are kept: the aggregate is over the reviewers who answered.
        reports = [report for report in exc.results
                   if not (report.get("partial") or report.get("timed_out"))]
        if len(reports) < MIN_SCORERS_FOR_A_MATRIX:
            raise
    # The reviewers rank; the aggregate decides. There is no score, no threshold
    # and no head-to-head phase: the proposal the reviewers place best in total
    # is the round's mechanism.
    order = _rank_order(ideas, reports)
    return {"reports": reports, "bundles": bundle, "comparisons": [],
            "finalists": order,
            "finalist_scores": {identifier: entry["total_rank"]
                                 for identifier, entry in order.items()}}


HARD_OBJECTION_TERMS = ("cancel", "identity", "leak", "dimension", "degenerat",
                        "counterexample", "equivalence", "invariant")

#: How many reviewers a matrix needs. One review that answers is enough to
#: decide; the others may be out of time, and a round is not held hostage by the
#: reviewer that did not finish.
MIN_SCORERS_FOR_A_MATRIX = 1


def _hard_objection(report: dict) -> tuple:
    """The first objection that names a mathematical or leakage defect, if any."""
    for entry in report.get("scores", []):
        objection = str(entry.get("objection", "")).lower()
        counterexample = str(entry.get("counterexample", "")).strip()
        if counterexample or any(term in objection for term in HARD_OBJECTION_TERMS):
            return entry["idea_id"], entry.get("objection", ""), counterexample
    return None


def _one_bounded_revision(directory: Path, agent_root: Path, repo: Path, state: dict,
                          reports: list, outcome: dict) -> tuple:
    """Give a proposal ONE revision when a reviewer names a hard defect.

    The revised proposal gets a new hash, and the three reviewers see it afresh;
    the revision and the original objection stay in the state.
    """
    for report in reports:
        found = _hard_objection(report)
        if not found:
            continue
        identifier, objection, counterexample = found
        idea = ideas_module.idea_index(state["ideas"]).get(identifier)
        if idea is None:
            continue
        revision = int(idea.get("proposal_revision", 0))
        if revision >= 1:
            continue                      # the bounded budget is one revision
        try:
            answer = run_rebuttal(agent_root, idea, objection, counterexample, revision)
        except Exception as exc:                            # noqa: BLE001
            pipeline_module.record_event(state, "rebuttal_failed", idea_id=identifier,
                                         error=str(exc)[:200])
            pipeline_module.save_state(directory, state)
            continue
        updated = dict(idea)
        updated["proposal_revision"] = revision + 1
        updated["rebuttal"] = answer
        if answer.get("remaining"):
            updated["unresolved_objection"] = answer["remaining"]
        pipeline_module.record_event(state, "proposal_revised", idea_id=identifier,
                                     revision=updated["proposal_revision"],
                                     objection=objection[:200])
        state["ideas"] = [updated if entry["idea_id"] == identifier else entry
                          for entry in state["ideas"]]
        pipeline_module.save_state(directory, state)
        # The revised proposal must be scored against the same frozen bundle, so
        # the state has to travel with it; dropping it made the revision path
        # crash on the first re-score.
        outcome = run_scoring(agent_root, state["ideas"], repo=repo, state=state)
        return state, outcome["reports"], outcome
    return state, reports, outcome


def _rank_with_retry(root: Path, task_id: str, prompt: str) -> dict:
    """One ranking, and one bounded retry if its schema is broken.

    The ranking schema is strict on purpose: a missing proposal, a repeated one
    or a tied position is an error, not a partial average. A reviewer that
    breaks it is told what was wrong and asked once more, with a fresh
    workspace, so a stale answer can never be reused.
    """
    attempt = 0
    last_error = None
    # The budget belongs to the task, not to the attempt: a retry does not get
    # a second five minutes.
    import time as _time
    budget = float(settings.AGENT_DEADLINE_SECONDS)
    started = _time.monotonic()
    while attempt < 2:
        remaining = budget - (_time.monotonic() - started)
        if remaining <= 1.0:
            break
        suffix = "" if attempt == 0 else (
            "\n\nYour previous answer was rejected by the schema: " + str(last_error) +
            "\nAnswer again with one complete block per proposal, and keep the five lines "
            "of a block together:\n"
            "SCORE: <idea_id> | behavior_evidence=<0-5> | geometry_necessity=<0-5> | "
            "novelty=<0-5> | downstream_mechanism=<0-5> | falsifiability=<0-5> | "
            "implementability=<0-5> | exploration_value=<0-5>\n"
            "SCOPE: <idea_id> | IN_SCOPE or OUT_OF_SCOPE | <why>\n"
            "EVIDENCE: <idea_id> | <evidence ids>\n"
            "OBJECTION: <idea_id> | <the strongest objection>\n"
            "RATIONALE: <idea_id> | <why these numbers, and the uncertainty>\n"
            "Every dimension value is a number from 0 to 5. A dimension you cannot ground "
            "takes the number you can defend, and the RATIONALE says the dimension is "
            "UNKNOWN; UNKNOWN is never a dimension value. Every proposal you score needs "
            "its own RATIONALE line, carrying its own idea_id.")
        name = "scores.txt" if attempt == 0 else "scores_retry.txt"
        run_id = task_id if attempt == 0 else task_id + "_retry1"
        try:
            report = _run_agent(root, run_id, prompt + suffix, name,
                                plaintext.parse_ranks, role="score",
                                read_paths=[],
                                allowed_roots=_round_roots(root),
                                deadline_seconds=max(1, int(remaining)))
            scoring_module.validate_report(report, reviewer=task_id)
            return report
        except (scoring_module.ScoreError, pipeline_module.RoundError) as exc:
            last_error = exc
            attempt += 1
    raise pipeline_module.RoundError("reviewer " + task_id + " broke the ranking schema "
                                     "twice: " + str(last_error))


def _rank_order(ideas: list, reports: list) -> dict:
    """The aggregate order of the proposals: the reviewers' ranks added up.

    The proposal the reviewers place first most often, and in total least far
    down, is the round's mechanism. The order is returned as a mapping from the
    idea id to its row, already sorted.
    """
    matrix = scoring_module.build_matrix(ideas, reports)
    rows = sorted(matrix["rows"],
                  key=lambda row: (row.get("total_rank", 0),
                                   -row.get("first_places", 0),
                                   row["idea_id"]))
    return {row["idea_id"]: row for row in rows}


def run_rebuttal(root: Path, idea: dict, objection: str, counterexample: str,
                 revision: int) -> dict:
    """One bounded repair of a proposal whose objection names a defect."""
    prompt = agent_contracts.REBUTTAL_PROMPT.format(objection=objection,
                                                    counterexample=counterexample or "none")
    return _run_agent(root, "rebuttal_" + idea["idea_id"] + "_r" + str(revision),
                      prompt, "rebuttal.txt", plaintext.parse_rebuttal, role="score",
                      read_paths=[],
                      allowed_roots=_round_roots(root))


def phase_scoring(root: Path, round_id: str, *, log=print) -> dict:
    """Run the three scorers over every idea and select the qualified ones."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    agent_root = directory
    if state["step"] != "scoring":
        return {"phase": "scoring", "ran": False, "step": state["step"],
                "selected": len((state.get("selection") or {}).get("selected", [])),
                "reason": "the round is no longer in the scoring phase"}
    outcome = run_scoring(agent_root, state["ideas"], repo=Path(root), state=state)
    reports = outcome["reports"]
    state, reports, outcome = _one_bounded_revision(
        directory, agent_root, Path(root), state, reports, outcome)
    state = pipeline_module.apply_scoring(
        state, reports, comparisons=outcome.get("comparisons"),
        bundles=outcome.get("bundles"), directory=directory)
    pipeline_module.save_state(directory, state)
    selection = state.get("selection") or {}
    log("selection: %s" % selection.get("status", "?"))
    return {"phase": "scoring", "ran": True, "step": state["step"],
            "rubric": state.get("rubric_version"),
            "status": selection.get("status"),
            "decision_hash": state.get("decision_hash"),
            "scorers": len(reports),
            "finalists": outcome.get("finalists"),
            "finalist_scores": outcome.get("finalist_scores"),
            "selected": [entry["idea_id"] for entry in selection.get("selected", [])],
            "backups": [entry["idea_id"] for entry in selection.get("backups", [])],
            "rejected": [(entry["idea_id"], entry.get("status"))
                         for entry in selection.get("rejected", [])]}
