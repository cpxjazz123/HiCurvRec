"""Step 6 - the candidate tree and the registration of the
mechanism the MAIN agent wrote.

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
    _run_agent, _run_agents_concurrently, _round_roots,
    _workspace, geometry_contract, role_of_task,
)



def run_implementation(root: Path, idea: dict, tree: Path) -> dict:
    """One implementation agent per selected idea, in its own tree.

    The tree is a fresh copy of the one permitted baseline, made here, so the
    agent never starts from another candidate's work and never from a mutated
    worktree.
    """
    tree = Path(tree)
    provenance = baseline_module.copy_baseline(tree)
    prompt = (agent_contracts.IMPLEMENT_PROMPT.format(
        idea=plaintext.render_idea(idea), tree=str(tree),
        objective=agent_contracts.RESEARCH_OBJECTIVE,
        scope=agent_contracts.SCOPE_CONTRACT + "\n\n" +
        agent_contracts.EXPERIMENT_POLICY))
    # No implementation agent runs in the Issue #121 flow: the framework copies
    # the baseline and the main agent writes the mechanism. If this role is
    # re-enabled, it also needs the Stage 0/1 inputs and the interpreter as read
    # paths (it must run the smoke); the tree alone is not enough.
    report = _run_agent(root, "implementation_" + idea["idea_id"], prompt,
                        "implementation.txt", plaintext.parse_implementation,
                        role="implementation")
    report["provenance"] = provenance
    report["baseline"] = provenance["baseline"]
    report["baseline_digest"] = provenance["baseline_digest"]
    report["candidate_digest_before_agent"] = provenance["candidate_digest"]
    atomic_json(tree / "PROPOSAL.json", idea)
    # Issue #124: the candidate's own path is checked, not just the generic
    # probe. A candidate whose implemented operator is insensitive to the
    # behaviour signal, or that loses the gradient, is blocked here.
    try:
        from . import candidate_checks
        check = candidate_checks.run_candidate_check(tree, repo=_repository_root(Path(root)))
    except Exception as exc:                              # noqa: BLE001
        check = {"verdict": "INCONCLUSIVE_OR_INFRA_ERROR",
                 "reason": "the candidate path check crashed: " + type(exc).__name__}
    report["candidate_path_check"] = check
    report["candidate_digest_after_agent"] = provenance["candidate_digest"]
    atomic_json(tree / "CANDIDATE_CHECK.json", check)
    return report



#: The cap on the diff a reviewer is shown. A revision that rewrites half the
#: tree is reviewed from the whole file list anyway; the patch names what moved.
DIFF_LIMIT = 60000

#: Files that belong to a run, not to the source: never part of a snapshot or a
#: diff, because they say nothing about what was implemented.
_DIFF_IGNORED = ("logs", "results", "dataset", "__pycache__", "ckpt", ".git")


def _source_files(root: Path) -> list:
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in _DIFF_IGNORED for part in relative.parts):
            continue
        files.append(relative)
    return files


def scope_file(root: Path) -> Path:
    """Where the round's scope contract is written for a reviewer to read."""
    target = Path(root) / "SCOPE_CONTRACT.md"
    if not target.is_file():
        target.write_text(agent_contracts.SCOPE_CONTRACT + "\n\n" +
                          agent_contracts.EXPERIMENT_POLICY + "\n", encoding="utf-8")
    return target


def objective_summary(objective: str) -> str:
    """The objective in one line: the prompt carries the file, not the text."""
    for line in str(objective).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:240]
    return str(objective)[:240]


def phase_implementation(root: Path, round_id: str, *, log=print) -> dict:
    """Build one isolated candidate per selected idea, each from the baseline."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    agent_root = directory
    if state["step"] != "implementation":
        return {"phase": "implementation", "ran": False, "step": state["step"],
                "implementations": len(state.get("implementations", [])),
                "reason": "the round is no longer in the implementation phase"}
    # The mathematics is stated, reviewed and frozen first, and three
    # independent implementations of it must agree before the candidate tree is
    # touched: the implementation may only follow the frozen specification.
    if state.get("step") == "formulation":
        formulation = phase_formulation(root, round_id, log=log)
        if formulation.get("ran") is False:
            return formulation
        state = pipeline_module.load_state(directory)
    if state.get("step") == "coding":
        coding = phase_coding(root, round_id, log=log)
        if coding.get("ran") is False:
            return coding
        state = pipeline_module.load_state(directory)

    # The three independent derivations of the selected mechanism come first:
    # registration is refused until they agree.
    derivation = phase_derivation(root, round_id, log=log)
    # The gate is a CONTRADICTION, not an incomparability: two derivations of
    # the same instance that disagree stop the candidate; two that could not be
    # compared are a limitation of the specification, recorded with it.
    state_after = pipeline_module.load_state(directory)
    differential = state_after.get("differential") or {}
    contradictions = [entry for entry in differential.get("pairs") or []
                      if entry.get("kind") == "contradiction"]
    if contradictions:
        raise pipeline_module.RoundError(
            "the independent derivations contradict each other: " +
            str(contradictions[0].get("reason"))[:300])
    trees_root = (Path(root) / "results" / "research" / state.get("issue", "issue120")
                  / "trees" / round_id)
    for idea in pipeline_module.selected_ideas(state):
        tree = trees_root / idea["idea_id"]
        if any(record["idea_id"] == idea["idea_id"]
               for record in state.get("implementations", [])):
            continue
        # There is no implementation agent: the main agent writes the mechanism.
        # The candidate tree is prepared here as a fresh copy of the one
        # permitted baseline, and the record waits for the main agent to write
        # the code and register it (research_round_register_implementation).
        try:
            tree.parent.mkdir(parents=True, exist_ok=True)
            # One copy path for every candidate: it renames the framework-owned
            # ARM_NAME to this tree, records the baseline path and both digests,
            # and refuses a copy that changed anything else. A plain copytree left
            # the candidate pointing at the baseline's result directory.
            provenance = baseline_module.copy_baseline(tree)
            record = pipeline_module.implementation_record(idea, {}, tree)
            record["baseline"] = provenance["baseline"]
            record["baseline_digest"] = provenance["baseline_digest"]
            record["arm_renamed"] = provenance["arm_renamed"]
            record["status"] = "awaiting_implementation"
            record["implemented_by"] = "main_agent"
            state.setdefault("implementations", []).append(record)
            pipeline_module.save_state(directory, state)
            log("%s: candidate tree prepared for the main agent; it must declare "
                "its equations in reference_contract.json - registration is "
                "refused until the framework rebuilds them and they agree"
                % idea["idea_id"])
        except baseline_module.BaselineError as exc:
            # The baseline is the one permitted source: if it is missing or
            # unreadable the candidate is not built from anything else.
            state = pipeline_module.mark_blocked(
                state, idea["idea_id"], "baseline unavailable: " + str(exc),
                kind=pipeline_module.BASELINE_UNAVAILABLE)
            pipeline_module.save_state(directory, state)
            continue
        record = [item for item in state["implementations"]
                  if item["idea_id"] == idea["idea_id"]][0]
        # The candidate must write only into its own result directory: a
        # candidate that would overwrite the baseline or a sibling is
        # refused before it can train.
        siblings = [other["tree"] for other in state.get("implementations", [])
                    if other["idea_id"] != idea["idea_id"]]
        try:
            record["output_paths"] = baseline_module.guard_output_paths(
                tree, other_candidates=siblings)
        except baseline_module.BaselineError as exc:
            raise pipeline_module.RoundError(
                idea["idea_id"] + " writes outside its own result directory: " +
                str(exc)) from exc
        pipeline_module.save_state(directory, state)
        log("implemented %s" % idea["idea_id"])
    state["step"] = "stage2"
    pipeline_module.save_state(directory, state)
    return {"phase": "implementation", "ran": True, "step": state["step"],
            "implementations": [{"idea_id": record["idea_id"], "status": record["status"],
                                 "revision": record.get("revision", 0)}
                                for record in state.get("implementations", [])]}






#: The phases of a round, in the order they run. Verification is not a phase:
#: it is the framework's machine admission at launch.
#: The phases the worker can run, in the order the flow runs them.
PHASES = ("ideation", "probe", "scoring", "implementation")


def advance(root: Path, round_id: str, *, log=print) -> dict:
    """Run the agent stages of a round until it waits for Stage 2."""
    from . import lock as lock_module
    from . import pipeline as pipeline_module

    directory = pipeline_module.find_round_directory(root, round_id)
    lock = lock_module.RoundLock(directory)
    # One runner per round: the lock keeps a detached worker and a synchronous
    # phase call from paying for the same agents twice.
    lock.acquire(owner="worker:" + str(round_id))
    try:
        for phase in (phase_ideation, phase_probe, phase_scoring, phase_implementation):
            phase(root, round_id, log=log)
    finally:
        lock.release()
    return pipeline_module.summary(
        pipeline_module.load_state(pipeline_module.find_round_directory(root, round_id)))
