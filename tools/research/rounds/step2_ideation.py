"""Step 2 - the five ideation agents (`ideate`).

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
    _repository_root,
    AgentPhaseIncomplete, RoundIncomplete,
    _run_agent, _run_agents_concurrently, _round_roots,
    _workspace, geometry_contract, role_of_task,
)

from .agent_calls import IDEA_MAX, IDEA_MIN, SCORERS  # noqa: F401


def round_history(repo: Path, limit: int = 3) -> str:
    """What the earlier rounds measured, for the ideation agents to read.

    The proposals of a round are not revised inside the round. What the earlier
    rounds measured, what their reviews objected to and what their runs scored is
    therefore the material the next round starts from.
    """
    rounds_root = _repository_root(Path(repo)) / "results" / "research"
    if not rounds_root.is_dir():
        return ""
    blocks: list = []
    for issue in sorted(rounds_root.iterdir()):
        if not issue.is_dir():
            continue
        for directory in sorted((issue / "rounds").glob("round_*")):
            state_path = directory / "state.json"
            if not state_path.is_file():
                continue
            try:
                state = json.loads(state_path.read_text())
            except (OSError, ValueError):
                continue
            lines = [directory.name]
            selected = {entry.get("idea_id")
                        for entry in (state.get("selection") or {}).get("selected", [])}
            for idea in state.get("ideas") or []:
                if idea.get("idea_id") in selected:
                    lines.append("  selected: " + str(idea.get("idea_id")) + " | " +
                                 str(idea.get("title"))[:90])
            for identifier, outcome in (state.get("probe_results") or {}).items():
                statuses = []
                probe = (outcome.get("probe") or {}).get("status")
                math_report = (outcome.get("math") or {}).get("status")
                literature = (outcome.get("literature") or {}).get("status")
                if probe and probe != "OK":
                    statuses.append("probe=" + str(probe))
                if math_report and math_report != "OK":
                    statuses.append("math=" + str(math_report))
                if literature and literature != "OK":
                    statuses.append("literature=" + str(literature))
                if statuses:
                    lines.append("  probe of " + identifier + ": " + ", ".join(statuses))
            for job in (state.get("stage2_jobs") or []):
                lines.append("  run: " + str(job.get("idea_id")) + " -> " +
                             str(job.get("status")) + " " + str(job.get("error"))[:80])
            for job in (state.get("stage3_jobs") or []):
                result = job.get("result") or {}
                if result:
                    lines.append("  stage3 of " + str(job.get("idea_id")) + ": recall@10=" +
                                 str(result.get("test_recall@10")))
            if len(lines) > 1:
                blocks.append("\n".join(lines))
    if not blocks:
        return ""
    return "\n".join(blocks[-limit:])


def ideate_one(root: Path, objective: str, extra: bool, index: int) -> tuple:
    """One ideation agent: its own task id, prompt, workspace and answer file.

    The function is the unit the round runs concurrently and the unit the
    flow dispatches it concurrently; every caller uses it, so the workflows
    cannot drift apart.
    """
    prompt_base = agent_contracts.IDEA_PROMPT.format(
        minimum=IDEA_MIN, maximum=IDEA_MAX,
        agents=pipeline_module.IDEA_AGENTS,
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY)
    task_id = "ideation_" + str(index)
    prompt = (prompt_base + "\n\nThe round objective: " + (objective or "improve the "
                 "item semantic IDs for a generative recommender") +
                 "\nYour answer must be a different set of ideas from the other " +
                 str(pipeline_module.IDEA_AGENTS - 1) +
                 " agents, which you cannot see.")
    history = round_history(Path(root))
    if history:
        prompt += ("\n\nWHAT THE EARLIER ROUNDS MEASURED, AND WHAT THEY FAILED ON:\n"
                   + history +
                   "\nUse it to aim at something the earlier rounds did not measure. "
                   "Do not restate a proposal that was already made and rejected.")
    if extra:
        prompt += ("\n\nThe first set of ideas you gave has been collected. Give one "
                   "MORE idea, and it must be different from the one you gave before "
                   "and from what the other agents propose.")
    return (task_id, _run_agent(root, task_id, prompt,
                                "ideas_extra.txt" if extra else "ideas.txt",
                                plaintext.parse_ideas,
                                role="ideate", read_paths=[],
                                allowed_roots=_round_roots(root)))


def run_ideation(root: Path, objective: str, extra: bool = False) -> list:
    """Five independent ideation agents; each returns one to three ideas.

    With ``extra`` the agents are asked for a second, different set, written to
    another file. That pass exists because agents that agree produce one
    proposal between them, and a round needs three distinct ones to be
    compared; agreement is a result, not a reason to end the round.
    """
    prompt_base = agent_contracts.IDEA_PROMPT.format(
        minimum=IDEA_MIN, maximum=IDEA_MAX,
        agents=pipeline_module.IDEA_AGENTS,
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY)
    def ideate(index):
        return ideate_one(root, objective, extra, index)

    try:
        return _run_agents_concurrently(
            [tuple([index]) for index in range(1, pipeline_module.IDEA_AGENTS + 1)], ideate)
    except AgentPhaseIncomplete as exc:
        # An agent that ran out of its five minutes has proposed nothing. The
        # proposals that did arrive are kept: a round needs three distinct
        # proposals to compare, and a missing proposer is not a reason to throw
        # the others away.
        if not exc.results:
            raise
        return exc.results


def phase_ideation(root: Path, round_id: str, *, log=print) -> dict:
    """Run the ideation agents and merge what they proposed."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    agent_root = directory
    if state["step"] != "ideation":
        return {"phase": "ideation", "ran": False, "step": state["step"],
                "ideas": len(state.get("ideas", [])),
                "reason": "the round is no longer in the ideation phase"}
    answers = run_ideation(agent_root, state.get("objective", ""))
    state = pipeline_module.apply_ideation(state, answers, require_minimum=False)
    if not state.get("ideas"):
        # Nothing was proposed. That is a result about the round's objective, not
        # a reason to end it: ask the same agents for one more idea each.
        extra = run_ideation(agent_root, state.get("objective", ""), extra=True)
        state = pipeline_module.apply_ideation(state, extra, require_minimum=False)
        if not state.get("ideas"):
            raise pipeline_module.RoundError(
                "a round needs at least one distinct idea, got none")
    pipeline_module.save_state(directory, state)
    log("ideation: %d ideas" % len(state["ideas"]))
    return {"phase": "ideation", "ran": True, "step": state["step"],
            "ideas": len(state["ideas"]),
            "rejected": len(state.get("ideas_rejected", [])),
            "duplicates": len(state.get("duplicates", []))}
