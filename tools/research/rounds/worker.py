"""The round worker: the agent half of an Issue #120 round.

It runs the three ideation agents, the three scoring agents, and then, for every
selected idea, one implementation agent followed by three verification agents
and, when they reject, a dedicated repair agent. It never trains anything: the
Stage 2 and Stage 3 jobs are prepared here and executed by the MCP, which owns
the GPU boundary.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .. import settings
from ..agents.runner import run_agent
from ..core.evidence import atomic_json, digest_file, read_json
from . import agents as agent_contracts
from . import ideas as ideas_module
from . import pipeline as pipeline_module

IDEA_MIN = 1
IDEA_MAX = 3
SCORERS = 3
VERIFIERS = 3


def _workspace(root: Path, task_id: str) -> Path:
    return Path(root) / "agents" / task_id


def _run_agent(root: Path, task_id: str, prompt: str, output_name: str) -> dict:
    """Run one agent in its own workspace and return its parsed answer."""
    # The runner gives the agent a workspace but not a destination; without this
    # the agent answers in prose and writes no file.
    destination = Path(root) / "agents" / task_id / output_name
    prompt = (prompt + "\n\nWrite the answer to " + str(destination) +
              " (this absolute path). Write nowhere else.")
    output = run_agent(task_id, prompt, output_name, Path(root),
                        binary=settings.AGENT_BINARY, model=settings.AGENT_MODEL,
                        timeout=settings.AGENT_TIMEOUT_SECONDS)
    payload = read_json(output)
    if not isinstance(payload, dict):
        raise pipeline_module.RoundError("agent " + task_id + " returned no JSON object")
    return payload


def run_ideation(root: Path, objective: str) -> list:
    """Three independent ideation agents; each returns one to three ideas."""
    prompt_base = agent_contracts.IDEA_PROMPT.format(minimum=IDEA_MIN, maximum=IDEA_MAX)
    answers = []
    for index in range(1, pipeline_module.IDEAE_AGENTS + 1):
        task_id = "ideation_" + str(index)
        prompt = (prompt_base + "\n\nThe round objective: " + (objective or "improve the "
                     "item semantic IDs for a generative recommender") +
                     "\nYour answer must be a different set of ideas from the other two "
                     "agents, which you cannot see.")
        answers.append((task_id, _run_agent(root, task_id, prompt, "ideas.json")))
    return answers


def run_scoring(root: Path, ideas: list) -> list:
    """Three independent scorers, each judging every idea."""
    payload = json.dumps(ideas, ensure_ascii=False, indent=1)
    reports = []
    for index in range(1, SCORERS + 1):
        task_id = "scoring_" + str(index)
        prompt = (agent_contracts.SCORE_PROMPT +
                  "\n\nThe ideas to score, all of them:\n" + payload)
        report = _run_agent(root, task_id, prompt, "scores.json")
        report.setdefault("scorer", task_id)
        reports.append(report)
    return reports


def run_implementation(root: Path, idea: dict, tree: Path) -> dict:
    """One implementation agent per selected idea, in its own tree."""
    tree = Path(tree)
    tree.mkdir(parents=True, exist_ok=True)
    prompt = (agent_contracts.IMPLEMENT_PROMPT.format(
        idea=json.dumps(idea, ensure_ascii=False, indent=1), tree=str(tree)))
    report = _run_agent(root, "implementation_" + idea["idea_id"], prompt, "implementation.json")
    atomic_json(tree / "PROPOSAL.json", idea)
    return report


def run_verification(root: Path, idea: dict, tree: Path) -> list:
    """Three independent verifiers of one implementation."""
    prompt = (agent_contracts.VERIFY_PROMPT.format(
        idea=json.dumps(idea, ensure_ascii=False, indent=1), tree=str(tree)))
    reports = []
    for index in range(1, VERIFIERS + 1):
        task_id = "verify_" + idea["idea_id"] + "_" + str(index)
        report = _run_agent(root, task_id, prompt, "verification.json")
        report.setdefault("verifier", task_id)
        reports.append(report)
    return reports


def run_repair(root: Path, idea: dict, tree: Path, findings: list) -> dict:
    """The dedicated repair agent for one implementation."""
    prompt = (agent_contracts.REPAIR_PROMPT.format(
        findings=json.dumps(findings, ensure_ascii=False, indent=1), tree=str(tree)))
    return _run_agent(root, "repair_" + idea["idea_id"], prompt, "repair.json")


def advance(root: Path, round_id: str, *, log=print) -> dict:
    """Run the agent stages of a round until it waits for Stage 2."""
    directory = pipeline_module.round_root(root) / round_id
    state = pipeline_module.load_state(directory)
    # Agent workspaces belong to the round, not to the repository root.
    agent_root = directory

    if state["step"] == "ideation":
        answers = run_ideation(agent_root, state.get("objective", ""))
        state = pipeline_module.apply_ideation(state, answers)
        pipeline_module.save_state(directory, state)
        log("ideation: %d ideas" % len(state["ideas"]))

    if state["step"] == "scoring":
        reports = run_scoring(agent_root, state["ideas"])
        state = pipeline_module.apply_scoring(state, reports)
        pipeline_module.save_state(directory, state)
        log("selection: %d selected" % len(state["selection"]["selected"]))

    if state["step"] == "implementation":
        for idea in pipeline_module.selected_ideas(state):
            tree = Path(root) / "results" / "research" / "issue120" / "trees" / round_id / idea["idea_id"]
            try:
                report = run_implementation(agent_root, idea, tree)
            except Exception as exc:                       # noqa: BLE001 - recorded, not raised
                state = pipeline_module.mark_blocked(state, idea["idea_id"],
                                                     "implementation failed: " + str(exc))
                pipeline_module.save_state(directory, state)
                continue
            state.setdefault("implementations", []).append(
                pipeline_module.implementation_record(idea, report, tree))
            pipeline_module.save_state(directory, state)
            log("implemented %s" % idea["idea_id"])
        state["step"] = "verification"
        pipeline_module.save_state(directory, state)

    if state["step"] == "verification":
        for record in list(state.get("implementations", [])):
            if record.get("status") not in ("implemented", "repaired"):
                continue
            idea = ideas_module.idea_index(state["ideas"])[record["idea_id"]]
            try:
                reports = run_verification(agent_root, idea, Path(record["tree"]))
            except Exception as exc:                       # noqa: BLE001
                state = pipeline_module.mark_blocked(state, record["idea_id"],
                                                     "verification failed: " + str(exc))
                pipeline_module.save_state(directory, state)
                continue
            state = pipeline_module.apply_verification(state, record["idea_id"], reports)
            pipeline_module.save_state(directory, state)
            if record.get("status") == "needs_repair":
                try:
                    repair = run_repair(agent_root, idea, Path(record["tree"]),
                                        record.get("blocking_findings", []))
                    state = pipeline_module.apply_repair(state, record["idea_id"], repair)
                except Exception as exc:                   # noqa: BLE001
                    state = pipeline_module.mark_blocked(state, record["idea_id"],
                                                         "repair failed: " + str(exc))
                pipeline_module.save_state(directory, state)
                log("%s: %s" % (record["idea_id"], record.get("status")))
        state["step"] = "stage2"
        pipeline_module.save_state(directory, state)

    return pipeline_module.summary(state)
