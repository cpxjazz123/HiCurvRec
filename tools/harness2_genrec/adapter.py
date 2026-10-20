"""The GeneRec benchmark adapter: the seam between Harness² and this repository.

Issue #101 §D/P0. The adapter answers three questions for the framework:

* what is a task (a frozen correctness task with a served spec and a private held-out
  set), and what is a domain (a fixture family);
* what does a rollout produce (a verifier run over the frozen fixtures) and is it a
  deliverable (a truthful one);
* how is a candidate harness judged (correctness, mutation detection, false successes,
  reproducibility, cost) -- with **any false success disqualifying the candidate**,
  whatever the weighted numbers say.

The adapter never imports candidate code and never lets a candidate reach the oracle:
candidate trees are copied to an isolated scratch directory before they run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from harness2.core import tree as harness_tree
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec

from . import prompts_ext, tasks
from .substrate import VERIFIER_TIMEOUT_S, LocalSubstrate, run_verifier
from .surface import GENREC_SURFACE

# The framework's prompt builder resolves per-surface text from a module-level dict; this
# repository's surface is not one of the two it ships text for, so it registers its own.
prompts_ext.register()

#: The frozen Stage-2 hard-L1 ER threshold and the historical R4-08 value. A candidate may
#: inherit the scripts that carry them; it may not change them.
FROZEN_THRESHOLDS = ("16.38953942232631", "16.402627511591962")

#: The launch and gate tools the always-on protocol must keep naming.
REQUIRED_TOOLS = ("stage2_start", "cgv_check")

_SCRIPT_NAME = re.compile(r"[a-z0-9_][a-z0-9_]*\.py")


def repo_surface_tree(repo: Path) -> dict[str, str]:
    """The editable surface of the repository.

    Read by explicit path rather than by walking the tree: the repository holds gigabytes
    of run artifacts, and only the three editable kinds of file may ever reach a
    candidate.
    """
    repo = Path(repo)
    tree: dict[str, str] = {}
    claude = repo / "CLAUDE.md"
    if claude.is_file():
        tree["CLAUDE.md"] = claude.read_text(encoding="utf-8")
    skills = repo / ".agent" / "skills"
    if skills.is_dir():
        for skill in sorted(skills.glob("*/SKILL.md")):
            tree[skill.relative_to(repo).as_posix()] = skill.read_text(encoding="utf-8")
    scripts = repo / "scripts"
    if scripts.is_dir():
        for script in sorted(scripts.glob("*.py")):
            if not _SCRIPT_NAME.fullmatch(script.name):
                continue
            tree[f"scripts/{script.name}"] = script.read_text(encoding="utf-8")
    return tree


class GenrecAdapter:
    """The benchmark seam for the GeneRec iteration harness."""

    def __init__(
        self,
        repo: Path,
        runs_root: Path,
        task_ids: Sequence[str] | None = None,
        channel: ChannelSpec | None = None,
    ) -> None:
        self.repo = Path(repo)
        self.runs_root = Path(runs_root)
        self.substrate = LocalSubstrate()
        self._task_ids = list(task_ids) if task_ids else tasks.all_tasks()
        self._base_tree = repo_surface_tree(self.repo)
        channel = channel or ChannelSpec("opencode", "local/deterministic")
        self.spec = BenchSpec(
            name="genrec",
            runs_root=self.runs_root,
            install_mode="workspace_agents_md",
            prompts=BenchPrompts(
                work=(
                    "ship a verifier that computes the masked weighted sum of a frozen "
                    "fixture from its input and reports whether it reproduces the declared "
                    "value"
                ),
                domain_word="fixture family",
                inv_levers=(
                    "name the invariant that the verifier must COMPUTE the value from the "
                    "input instead of restating the declared value"
                ),
                dom_levers=(
                    "name the fixture shape and the masked weighted sum rule of this fixture "
                    "family"
                ),
                yardstick=(
                    "the verifier tells the truth on held-out fixtures it has never seen: it "
                    "catches every fixture whose value changed and never claims success on one"
                ),
                deliverable_word="verifier",
                domain_guidance=(
                    "the fixture is the frozen specification: its input and its declared value "
                    "are both part of it, and the held-out fixtures are not yours to read"
                ),
                selection_checks=(
                    "the candidate's verifier reproduces the served fixture, catches every "
                    "held-out fixture whose value changed, and never claims success on one"
                ),
                selection_lead=(
                    "choose the candidate whose verifier computes the value itself and stays "
                    "silent about the fixtures it cannot reproduce"
                ),
                selection_conformance="full",
                prompt_pack="adapt",
            ),
            channels={
                "proposer": channel,
                "selector": channel,
            },
            # The held-out fixtures must never enter a model context.
            leak_globs=("heldout", "mutant_"),
            allow_subagents=False,
            reserved_scripts=frozenset(),
            proposer_deliverables="contents",
            state_deliverable=False,
            obs_trunc=50_000,
        )

    # --- the benchmark seam --------------------------------------------
    def tasks(self, split: str = "all") -> list[str]:
        return list(self._task_ids)

    def domain_of(self, task_id: str) -> str:
        return tasks.task(task_id).domain

    def task_prompt(self, task_id: str) -> str:
        return tasks.task(task_id).brief

    def task_files(self, task_id: str) -> Path | None:
        served = tasks.served_dir(task_id)
        return served if served.is_dir() else None

    def base_harness(self) -> Path:
        dest = self.runs_root / "base_harness"
        harness_tree.write_tree(GENREC_SURFACE, self._base_tree, dest, stubs=False)
        return dest

    # --- the gate a candidate tree must pass ---------------------------
    def validate_tree(self, tree: dict[str, str]) -> None:
        """Reject a candidate tree before it is ever rolled.

        The message is fed back verbatim to the proposer as retry feedback, so it says
        what to do, not only what is wrong.
        """
        problems: list[str] = []

        script = tree.get(tasks.VERIFIER_SCRIPT)
        if script is None:
            problems.append(
                f"the harness has no {tasks.VERIFIER_SCRIPT}: this task needs a verifier that "
                "takes one fixture path and prints a STATUS line and a RESULT line"
            )
        else:
            for literal in self._forbidden_literals():
                if re.search(rf"(?<![\d.]){re.escape(literal)}(?![\d.])", script):
                    problems.append(
                        f"{tasks.VERIFIER_SCRIPT} contains the literal {literal}, a declared "
                        "fixture value: the verifier must COMPUTE the value from the input, "
                        "not restate it"
                    )

        claude = tree.get("CLAUDE.md", "")
        for tool in REQUIRED_TOOLS:
            if tool not in claude:
                problems.append(
                    f"CLAUDE.md no longer names `{tool}`: the always-on protocol must keep "
                    "naming the launch and gate tools"
                )

        for rel, body in sorted(tree.items()):
            base_body = self._base_tree.get(rel)
            if base_body is None or body == base_body:
                continue
            if any(threshold in base_body for threshold in FROZEN_THRESHOLDS):
                problems.append(
                    f"{rel} carries a frozen threshold: that file is not candidate-editable"
                )
            for threshold in FROZEN_THRESHOLDS:
                if threshold in body and threshold not in base_body:
                    problems.append(
                        f"{rel} introduces the frozen threshold {threshold}: thresholds are "
                        "not candidate-editable"
                    )

        if problems:
            raise ValueError(" | ".join(problems))

    def _forbidden_literals(self) -> list[str]:
        return sorted({f"{value:.1f}" for value in tasks.expected_values().values()})

    # --- one rollout ---------------------------------------------------
    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        harness = Path(harness_dir) if harness_dir else self.base_harness()
        run_dir = self.runs_root / "executions" / run_id
        run_verifier(task_id, harness, run_dir, timeout=timeout or VERIFIER_TIMEOUT_S)
        return run_dir

    def delivered(self, run_dir: Path) -> bool:
        """A rollout delivers only if its verifier was truthful on every fixture."""
        verdict = Path(run_dir) / "verdict.json"
        if not verdict.is_file():
            return False
        doc = json.loads(verdict.read_text(encoding="utf-8"))
        return bool(doc["correct_pass"]) and int(doc["false_pass_count"]) == 0

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        path = Path(run_dir) / "trajectory.md"
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[:obs_trunc]

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        path = Path(run_dir) / "report.txt"
        if not path.is_file():
            return {}
        return {"report.txt": path.read_text(encoding="utf-8", errors="replace")}

    def deliverable_text(self, run_dir: Path) -> str:
        return "".join(self.deliverable_files(run_dir).values())

    # --- the judgment --------------------------------------------------
    def score(self, run_dir: Path, task_id: str) -> dict | None:
        """Judge one run. ``None`` means the judge failed, which is not a zero."""
        verdict = Path(run_dir) / "verdict.json"
        if not verdict.is_file():
            return None
        doc = json.loads(verdict.read_text(encoding="utf-8"))
        total = int(doc["mutation_total"])
        detected = int(doc["mutation_detected"])
        return {
            "task": task_id,
            "correctness_pass_rate": 1.0 if doc["correct_pass"] else 0.0,
            "mutation_detection": (detected / total) if total else 0.0,
            "false_pass_count": int(doc["false_pass_count"]),
            "false_fail_count": int(doc["false_fail_count"]),
            "repair_success": bool(doc["correct_pass"]),
            "reproducible": True,
            "cost_usd": 0.0,
        }


__all__ = ["FROZEN_THRESHOLDS", "GenrecAdapter", "REQUIRED_TOOLS", "repo_surface_tree"]
