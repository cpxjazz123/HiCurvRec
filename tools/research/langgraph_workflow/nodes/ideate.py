"""Ideation: five agents in parallel, merged by the round's own transition."""
from __future__ import annotations

from pathlib import Path

from langgraph.types import Send

from tools.research.langgraph_workflow.adapters import ideation
from tools.research.rounds import ideas as ideas_module
from tools.research.rounds import pipeline as pipeline_module


def dispatch_ideation(state: dict) -> list:
    """One branch per ideation agent."""
    root = str(state["root"])
    objective = state.get("objective") or ""
    extra = bool(state.get("extra"))
    return [Send("idea_agent", {"root": root, "objective": objective,
                                 "extra": extra, "index": index})
            for index in range(1, pipeline_module.IDEA_AGENTS + 1)]


def idea_agent(task: dict) -> dict:
    """One agent's answer; a failed agent does not cost the others theirs."""
    try:
        agent, ideas = ideation.ideate_one(Path(task["root"]), task["objective"],
                                           task["extra"], task["index"])
    except Exception as exc:                                   # noqa: BLE001
        return {"errors": ["ideation agent %s failed: %s" % (task["index"], exc)]}
    return {"per_agent": [(agent, ideas)]}


def merge_ideas(state: dict) -> dict:
    """Merge the answers with the round's own transition and persist it."""
    root = Path(state["root"])
    directory = pipeline_module.find_round_directory(root, state["round_id"])
    round_state = pipeline_module.load_state(directory)
    answers = list(state.get("per_agent") or [])
    round_state = pipeline_module.apply_ideation(round_state, answers,
                                                 require_minimum=False)
    if len(round_state.get("ideas") or []) < ideas_module.MIN_ROUND_IDEAS:
        # The agents agreed: ask the same agents for one more idea each.
        extra = [ideation.ideate_one(root, state.get("objective") or "", True, index)
                 for index in range(1, pipeline_module.IDEA_AGENTS + 1)]
        round_state = pipeline_module.apply_ideation(round_state, extra,
                                                     require_minimum=False)
    pipeline_module.save_state(directory, round_state)
    return {"ideas": list(round_state.get("ideas") or []),
            "round_state": round_state,
            "phase_results": {"ideation": {
                "phase": "ideation", "ran": True, "step": round_state["step"],
                "ideas": len(round_state["ideas"]),
                "rejected": len(round_state.get("ideas_rejected", [])),
                "duplicates": len(round_state.get("duplicates", []))}}}
