"""Scoring: three reviewers in parallel, then the round's own selection."""
from __future__ import annotations

from pathlib import Path

from langgraph.types import Send

from tools.research.langgraph_workflow.adapters import scoring
from tools.research.rounds import pipeline as pipeline_module
from tools.research.rounds import worker


def dispatch_scoring(state: dict) -> list:
    """One branch per reviewer, over the frozen evidence bundle."""
    root = Path(state["root"])
    ideas = list(state.get("ideas") or [])
    round_state = state.get("round_state") or {}
    bundle, payload, evidence_text = scoring.scoring_inputs(
        root, ideas, repo=root, state=round_state)
    return [Send("scorer", {"root": str(root), "payload": payload,
                             "evidence_text": evidence_text, "index": index})
            for index in range(1, worker.SCORERS + 1)]


def scorer(task: dict) -> dict:
    """One reviewer's answer."""
    try:
        report = scoring.rank_one(Path(task["root"]), task["payload"],
                                  task["evidence_text"], task["index"])
    except Exception as exc:                                   # noqa: BLE001
        return {"errors": ["reviewer %s failed: %s" % (task["index"], exc)]}
    return {"reports": [report]}


def finalize(state: dict) -> dict:
    """Weigh the reviews and freeze the decision, exactly as the runner does."""
    root = Path(state["root"])
    directory = pipeline_module.find_round_directory(root, state["round_id"])
    round_state = state.get("round_state") or pipeline_module.load_state(directory)
    reports = list(state.get("reports") or [])
    bundle, _payload, _evidence = scoring.scoring_inputs(
        root, list(state.get("ideas") or []), repo=root, state=round_state)
    outcome = {"reports": reports, "bundles": bundle, "comparisons": [],
               "finalists": worker.rank_order(list(state.get("ideas") or []), reports),
               "finalist_scores": {}}
    round_state, reports, outcome = worker.apply_bounded_revision(
        directory, directory, root, round_state, reports, outcome)
    round_state = pipeline_module.apply_scoring(
        round_state, reports, comparisons=outcome.get("comparisons"),
        bundles=outcome.get("bundles"), directory=directory)
    pipeline_module.save_state(directory, round_state)
    selection = round_state.get("selection") or {}
    result = {"phase": "scoring", "ran": True, "step": round_state["step"],
              "status": selection.get("status"),
              "decision_hash": round_state.get("decision_hash"),
              "scorers": len(reports),
              "selected": [entry["idea_id"] for entry in selection.get("selected", [])],
              "backups": [entry["idea_id"] for entry in selection.get("backups", [])]}
    return {"selection": selection, "round_state": round_state,
            "phase_results": {"scoring": result}}
