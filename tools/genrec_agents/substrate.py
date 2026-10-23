"""The local substrate: runs a candidate harness's verifier against the frozen fixtures.

`LocalSubstrate` implements the framework's ``Substrate`` protocol (``surface`` /
``materialise`` / ``execute`` / ``read_trajectory`` / ``preflight``). The executor is
the candidate's own verifier script, run from an isolated copy of the candidate tree
with a scrubbed environment; the substrate never imports candidate code and a
candidate can never import the oracle.

Two verdicts matter and are computed here, not by the candidate:

* ``false_pass`` -- the verifier claimed the fixture reproduces, but the independent
  reference disagrees (or the fixture's declared value is not the true one at all);
* ``mutation_detected`` -- how many held-out fixtures the verifier caught.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

from .surface import SurfaceSpec

from . import opencode, oracle, tasks
from .surface import GENREC_SURFACE

VERIFIER_TIMEOUT_S = 60

_STATUS_PREFIX = "STATUS:"
_RESULT_PREFIX = "RESULT:"


@dataclass(frozen=True)
class FixtureRow:
    fixture: str
    role: str  # "served" | "heldout"
    truth: float
    declared: float
    reported: str  # "PASS" | "FAIL" | "ERROR"
    computed: float | None
    false_pass: bool
    false_fail: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "fixture": self.fixture,
            "role": self.role,
            "truth": self.truth,
            "declared": self.declared,
            "reported": self.reported,
            "computed": self.computed,
            "false_pass": self.false_pass,
            "false_fail": self.false_fail,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RunReport:
    task_id: str
    verifier_present: bool
    rows: tuple[FixtureRow, ...]
    correct_pass: bool
    mutation_detected: int
    mutation_total: int
    false_pass_count: int
    false_fail_count: int
    detail: str = ""

    @property
    def truthful(self) -> bool:
        """A deliverable must reproduce the served fixture and never claim a false success."""
        return self.correct_pass and self.false_pass_count == 0

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "verifier_present": self.verifier_present,
            "correct_pass": self.correct_pass,
            "mutation_detected": self.mutation_detected,
            "mutation_total": self.mutation_total,
            "false_pass_count": self.false_pass_count,
            "false_fail_count": self.false_fail_count,
            "detail": self.detail,
            "rows": [r.as_dict() for r in self.rows],
        }

    def report_text(self) -> str:
        lines = [
            f"task: {self.task_id}",
            f"verifier present: {'yes' if self.verifier_present else 'no'}",
            f"reproduces the served fixture: {'yes' if self.correct_pass else 'no'}",
            f"held-out fixtures caught: {self.mutation_detected}/{self.mutation_total}",
            f"false successes: {self.false_pass_count}",
            f"false failures: {self.false_fail_count}",
            "",
        ]
        for r in self.rows:
            lines.append(
                f"[{r.role}] {r.fixture}: reported {r.reported}, computed {r.computed}, "
                f"reference {r.truth}, declared {r.declared}"
                + ("  <-- FALSE SUCCESS" if r.false_pass else "")
                + ("  <-- FALSE FAILURE" if r.false_fail else "")
            )
            if r.detail:
                lines.append(f"        {r.detail}")
        return "\n".join(lines) + "\n"


def _clean_env() -> dict[str, str]:
    """A scrubbed environment for a candidate script: no tokens, no inherited config."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _parse_output(stdout: str) -> tuple[str, float | None, str]:
    status = "ERROR"
    computed: float | None = None
    detail = ""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(_STATUS_PREFIX):
            value = line[len(_STATUS_PREFIX) :].strip().upper()
            status = value if value in ("PASS", "FAIL") else "ERROR"
        elif line.startswith(_RESULT_PREFIX):
            raw = line[len(_RESULT_PREFIX) :].strip()
            try:
                computed = float(raw)
            except ValueError:
                computed = None
                detail = f"RESULT was not a number: {raw!r}"
    if status == "ERROR" and not detail:
        detail = "no STATUS line in the verifier output"
    return status, computed, detail


def run_verifier(
    task_id: str,
    harness_dir: Path,
    run_dir: Path,
    *,
    timeout: int = VERIFIER_TIMEOUT_S,
) -> RunReport:
    """Run the candidate's verifier over the served and the held-out fixtures."""
    harness_dir = Path(harness_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    checker = harness_dir / tasks.VERIFIER_SCRIPT
    if not checker.is_file():
        report = RunReport(
            task_id=task_id,
            verifier_present=False,
            rows=(),
            correct_pass=False,
            mutation_detected=0,
            mutation_total=len(tasks.heldout_files(task_id)),
            false_pass_count=0,
            false_fail_count=0,
            detail=f"no verifier at {tasks.VERIFIER_SCRIPT}",
        )
        _write_report(run_dir, report)
        return report

    fixtures: list[tuple[Path, str]] = [
        (f, "served") for f in tasks.served_fixtures(task_id)
    ] + [(f, "heldout") for f in tasks.heldout_files(task_id)]

    rows: list[FixtureRow] = []
    for path, role in fixtures:
        doc = json.loads(path.read_text(encoding="utf-8"))
        truth = oracle.reference_value(doc)
        declared = float(doc["expected"])
        try:
            proc = subprocess.run(
                [sys.executable, str(checker), str(path)],
                cwd=str(harness_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_clean_env(),
            )
            status, computed, detail = _parse_output(proc.stdout)
            if proc.returncode != 0 and status == "ERROR":
                detail = (detail + " | " + (proc.stderr or "").strip()[-200:]).strip(" |")
        except subprocess.TimeoutExpired:
            status, computed, detail = "ERROR", None, f"verifier timed out after {timeout}s"
        except OSError as exc:
            status, computed, detail = "ERROR", None, f"{type(exc).__name__}: {exc}"

        fixture_is_valid = abs(truth - declared) <= oracle.TOLERANCE
        claim_ok = (
            fixture_is_valid
            and computed is not None
            and abs(computed - truth) <= oracle.TOLERANCE
        )
        false_pass = status == "PASS" and not claim_ok
        false_fail = status == "FAIL" and claim_ok
        rows.append(
            FixtureRow(
                fixture=path.name,
                role=role,
                truth=truth,
                declared=declared,
                reported=status,
                computed=computed,
                false_pass=false_pass,
                false_fail=false_fail,
                detail=detail,
            )
        )

    served = [r for r in rows if r.role == "served"]
    heldout = [r for r in rows if r.role == "heldout"]
    correct_pass = bool(served) and all(
        r.reported == "PASS" and not r.false_pass and not r.false_fail for r in served
    )
    report = RunReport(
        task_id=task_id,
        verifier_present=True,
        rows=tuple(rows),
        correct_pass=correct_pass,
        mutation_detected=sum(1 for r in heldout if r.reported == "FAIL" and not r.false_fail),
        mutation_total=len(heldout),
        false_pass_count=sum(1 for r in rows if r.false_pass),
        false_fail_count=sum(1 for r in rows if r.false_fail),
    )
    _write_report(run_dir, report)
    return report


def _write_report(run_dir: Path, report: RunReport) -> None:
    (run_dir / "verdict.json").write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "report.txt").write_text(report.report_text(), encoding="utf-8")
    lines = [
        f"[ACTION] run the harness verifier on the frozen fixtures of {report.task_id}",
    ]
    for r in report.rows:
        lines.append(
            f"[OBSERVATION] {r.fixture} -> STATUS {r.reported} (reference {r.truth}, "
            f"declared {r.declared})"
        )
        if r.detail:
            lines.append(f"[ERROR] {r.fixture}: {r.detail}")
    if not report.verifier_present:
        lines.append(f"[ERROR] {report.detail}")
    lines.append(
        "[OBSERVATION] held-out fixtures caught "
        f"{report.mutation_detected}/{report.mutation_total}, false successes "
        f"{report.false_pass_count}"
    )
    (run_dir / "trajectory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Install:
    """A materialised harness: where it was staged and what it was staged from."""

    staging_dir: Path
    fingerprint: str
    extras: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRequest:
    """One request to run a task's verifier over an installed harness."""

    task_ref: str
    run_ids: Tuple[str, ...] = ()
    timeout: int = 600
    name: str = ""

class LocalSubstrate:
    """Runs a candidate harness locally; the only substrate the P0 pipeline needs."""

    name = "local"

    def surface(self) -> SurfaceSpec:
        return GENREC_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        import shutil
        import tempfile

        tree_dir = Path(tree_dir)
        staged = Path(tempfile.mkdtemp(prefix="genrec-harness-"))
        if tree_dir.is_dir():
            shutil.copytree(tree_dir, staged, dirs_exist_ok=True)
        for key, body in (ctx or {}).items():
            target = staged / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        return Install(staging_dir=staged, fingerprint=str(tree_dir))

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        out: list[Path] = []
        for run_id in run_ids:
            run_dir = install.staging_dir / "runs" / run_id
            run_verifier(task_ref, install.staging_dir, run_dir, timeout=timeout)
            out.append(run_dir)
        return out

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400
    ) -> str:
        path = Path(stream_path)
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[TRUNCATED]\n"
        return text

    def preflight(self) -> None:
        for task_id in tasks.all_tasks():
            if not tasks.served_fixtures(task_id):
                raise RuntimeError(f"task {task_id} has no served fixture")
            if not tasks.heldout_files(task_id):
                raise RuntimeError(f"task {task_id} has no held-out fixture")


class OpenCodeSubstrate:
    """Executes a task with OpenCode, in an isolated copy of the work tree.

    The agent works in a scratch copy of the tree, never in the repository: the copy is
    what it edits, and the copy is what the judges are run against. The repository is
    reachable only through the genrec MCP server, whose tools decide what they touch, and
    the files that must not be edited -- the frozen specifications, the judges, the
    historical artefacts -- are outside the copy in the first place.

    Every run leaves its evidence behind: the event stream, the tool calls with their
    inputs and outputs, the cost, the tokens, the exit status and the wall time.
    """

    name = "opencode"

    def __init__(
        self,
        *,
        model: str = "",
        timeout: int = 1800,
        repo_root: Path | None = None,
        use_mcp: bool = True,
    ) -> None:
        self.model = model or opencode.model_name()
        self.timeout = int(timeout)
        self.repo_root = Path(repo_root) if repo_root else None
        self.use_mcp = bool(use_mcp)
        self._staging: List[Path] = []

    # -- the substrate protocol -----------------------------------------
    def surface(self) -> SurfaceSpec:
        return GENREC_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        staging = Path(tempfile.mkdtemp(prefix="genrec-oc-stage-"))
        self._staging.append(staging)
        if Path(tree_dir).is_dir():
            shutil.copytree(tree_dir, staging, dirs_exist_ok=True)
        for key, body in (ctx or {}).items():
            target = staging / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        return Install(staging_dir=staging, fingerprint=str(tree_dir))

    def execute(
        self, task_ref: str, install: Install, run_ids: List[str], timeout: int
    ) -> List[Path]:
        out: List[Path] = []
        for run_id in run_ids:
            out.append(
                self.run_task(
                    task_ref,
                    install.staging_dir,
                    install.staging_dir / "runs" / run_id,
                    timeout=timeout or self.timeout,
                )
            )
        return out

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400
    ) -> str:
        path = Path(stream_path)
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[TRUNCATED]\n"
        return text

    def preflight(self) -> None:
        report = opencode.preflight(self.model)
        if not report.ok:
            raise RuntimeError(
                "the OpenCode backend is not usable here: " + "; ".join(report.problems)
            )

    # -- the agent runner ------------------------------------------------
    def run_task(
        self,
        task_ref: str,
        tree: Path,
        run_dir: Path,
        *,
        prompt: str = "",
        timeout: int | None = None,
        title: str = "",
    ) -> Path:
        """Run the agent on the tree and write the evidence into ``run_dir``."""
        tree = Path(tree)
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        mcp = None
        if self.use_mcp and self.repo_root is not None:
            mcp = opencode.mcp_server_block(self.repo_root)
        result = opencode.run_agent(
            tree,
            prompt or f"Work on the task {task_ref}.",
            model=self.model,
            timeout=timeout or self.timeout,
            mcp=mcp,
            title=title or f"genrec {task_ref}",
            deny_paths=[Path(self.repo_root)] if self.repo_root else (),
        )
        (run_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(event, default=str) for event in result.get("events", [])) + "\n",
            encoding="utf-8",
        )
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "task": task_ref,
                    "model": result.get("model"),
                    "argv": result.get("argv"),
                    "exit_code": result.get("exit_code"),
                    "timed_out": result.get("timed_out"),
                    "elapsed_s": result.get("elapsed_s"),
                    "cost": result.get("cost"),
                    "tokens": result.get("tokens"),
                    "tool_calls": result.get("tool_calls"),
                    "errors": result.get("errors"),
                    "text": result.get("text"),
                    "stdout": result.get("stdout"),
                    "stderr": result.get("stderr"),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        lines = [f"[ACTION] opencode run on the task {task_ref}"]
        for call in result.get("tool_calls", []):
            lines.append(
                f"[ACTION] {call.get('tool')}: {json.dumps(call.get('input'), default=str)[:300]}"
            )
            lines.append(f"[OBSERVATION] {str(call.get('output', ''))[:600]}")
        for error in result.get("errors", []):
            lines.append(f"[ERROR] {error.get('message', error)}")
        if result.get("text"):
            lines.append("[OBSERVATION] " + str(result["text"])[:4000])
        (run_dir / "trajectory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return run_dir


__all__ = [
    "FixtureRow",
    "LocalSubstrate",
    "OpenCodeSubstrate",
    "RunReport",
    "VERIFIER_TIMEOUT_S",
    "run_verifier",
]
