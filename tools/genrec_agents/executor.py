"""The native production executor: a DAG task becomes a real OpenCode CLI subprocess.

The chain this module implements, end to end and without any third-party framework:

    DAG READY task
      -> Executor.execute(task, spec)
         -> opencode.run_agent(workspace, prompt, ...)      (opencode.py)
            -> subprocess.Popen([opencode, "run", "--standalone", "--format", "json", ...])
               -> the OpenCode CLI's own JSONL event stream on stdout
         -> the session's JSONL and its parsed summary are written to the evidence directory
         -> the task's verifier runs over the agent's workspace
      -> an ExecutionRecord: role, immutable prompt, model, input hash, task id, the real argv,
         the workspace, the pid and the real lifetime, the tool JSONL, the exit code, the elapsed
         time, the tokens and the cost, the outputs and the verification verdict

Two rules the module enforces, because the campaign's record shows what goes wrong without them:

* **An idle tick is not an execution.** The record is written only after a real subprocess ran; a
  task that was merely marked DONE by a coordinator has no record and cannot be counted.
* **One workspace per task.** Two agents never share a directory, so their outputs cannot be
  confused, and the verifier always reads the workspace the agent actually wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import opencode

__all__ = ["AgentSpec", "ExecutionRecord", "OpenCodeAgentExecutor", "ROLE_LITERATURE",
           "ROLE_ANALYSIS", "ROLE_PROPOSAL", "ROLE_IMPLEMENTATION", "ROLES"]

#: The four research roles the campaign's DAG binds to real sessions.
ROLE_LITERATURE = "L"
ROLE_ANALYSIS = "A"
ROLE_PROPOSAL = "M"
ROLE_IMPLEMENTATION = "I"
ROLES = (ROLE_LITERATURE, ROLE_ANALYSIS, ROLE_PROPOSAL, ROLE_IMPLEMENTATION)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AgentSpec:
    """Everything the executor needs to run one agent, and nothing it may change.

    The prompt is immutable: it is hashed into the record, so a run can always be compared with
    the declaration it claims to have executed.
    """

    role: str
    task_id: str
    prompt: str
    model: str = ""
    timeout: int = 900
    #: The hash of the proposal or input the prompt was built from. It ties the session to the
    #: frozen material it read.
    inputs_hash: str = ""
    #: Files that must exist in the workspace when the agent claims to be done.
    expected_outputs: Tuple[str, ...] = ()
    #: A callable over the workspace returning a verdict mapping. ``ok`` decides the outcome.
    verifier: Optional[Callable[[Path], Dict[str, Any]]] = None
    #: A bound on the agent's wall time, independent of the subprocess timeout.
    max_seconds: int = 900
    #: The files copied into the workspace before the agent starts.
    input_files: Tuple[Tuple[str, Path], ...] = ()

    @property
    def prompt_sha256(self) -> str:
        return _sha256_text(self.prompt)


@dataclass
class ExecutionRecord:
    """What one agent session actually did, written down whether it succeeded or not."""

    role: str
    task_id: str
    ok: bool
    started_utc: str = ""
    finished_utc: str = ""
    elapsed_s: float = 0.0
    pid: Optional[int] = None
    model: str = ""
    prompt_sha256: str = ""
    inputs_hash: str = ""
    argv: List[str] = field(default_factory=list)
    workspace: str = ""
    cwd: str = ""
    exit_code: Optional[int] = None
    timed_out: bool = False
    n_tool_calls: int = 0
    tools: List[str] = field(default_factory=list)
    tokens: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    session_path: str = ""
    tool_jsonl_path: str = ""
    outputs: List[str] = field(default_factory=list)
    missing_outputs: List[str] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OpenCodeAgentExecutor:
    """Runs DAG tasks as isolated OpenCode CLI sessions and records what happened.

    ``root`` is the run directory: ``root/workspaces/<task_id>`` is the agent's private
    workspace and ``root/evidence/<task_id>/`` holds the session's JSONL, the raw streams and
    this run's record.
    """

    def __init__(self, root: Path | str, *, model: str = "", mcp: Optional[Dict[str, Any]] = None,
                 binary: str = "", keep_workspaces: bool = True) -> None:
        self.root = Path(root)
        self.model = model or opencode.model_name()
        self.mcp = mcp
        self.binary = binary
        self.keep_workspaces = keep_workspaces

    # -- paths ------------------------------------------------------------
    def workspace_for(self, task_id: str) -> Path:
        return self.root / "workspaces" / task_id

    def evidence_dir(self, task_id: str) -> Path:
        return self.root / "evidence" / task_id

    # -- execution --------------------------------------------------------
    def execute(self, task_id: str, spec: AgentSpec) -> ExecutionRecord:
        """Run one agent to completion and write the evidence. Never raises for a failed run."""
        started = time.time()
        workspace = self.workspace_for(task_id)
        evidence = self.evidence_dir(task_id)
        evidence.mkdir(parents=True, exist_ok=True)
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True, exist_ok=True)

        record = ExecutionRecord(
            role=spec.role,
            task_id=task_id,
            ok=False,
            model=spec.model or self.model,
            prompt_sha256=spec.prompt_sha256,
            inputs_hash=spec.inputs_hash,
            workspace=str(workspace),
            cwd=str(workspace),
        )

        # the prompt is written down before it runs: the session can then be compared with it
        prompt_path = evidence / "prompt.txt"
        prompt_path.write_text(spec.prompt, encoding="utf-8")
        record.argv = []
        try:
            for name, source in spec.input_files:
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        except OSError as exc:
            record.error = f"could not stage the agent's inputs: {exc}"
            self._write(evidence, record)
            return record

        try:
            summary = opencode.run_agent(
                workspace,
                spec.prompt,
                model=record.model,
                timeout=spec.timeout,
                mcp=self.mcp,
                title=f"{spec.role}:{task_id}",
            )
        except Exception as exc:  # noqa: BLE001 - a failed agent is state, not a crash
            record.error = f"{type(exc).__name__}: {exc}"
            record.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            record.elapsed_s = round(time.time() - started, 3)
            self._write(evidence, record)
            return record

        # the session's own JSONL, so the tool stream is evidence and not a summary of one
        raw_jsonl = evidence / "session.jsonl"
        raw_jsonl.write_text(summary.get("stdout") or "", encoding="utf-8")
        session_path = evidence / "session.json"
        session_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

        record.argv = list(summary.get("argv") or [])
        record.pid = summary.get("pid")
        record.exit_code = summary.get("exit_code")
        record.timed_out = bool(summary.get("timed_out"))
        record.elapsed_s = float(summary.get("elapsed_s") or 0.0)
        record.started_utc = str(summary.get("started_utc") or "")
        record.finished_utc = str(summary.get("finished_utc") or "")
        record.n_tool_calls = int(summary.get("n_tool_calls") or 0)
        record.tools = list(summary.get("tools") or [])
        record.tokens = dict(summary.get("tokens") or {})
        record.cost = float(summary.get("cost") or 0.0)
        record.session_path = str(session_path)
        record.tool_jsonl_path = str(raw_jsonl)

        # the outputs the agent promised, and the verifier's verdict
        for name in spec.expected_outputs:
            candidate = workspace / name
            if candidate.exists():
                record.outputs.append(name)
            else:
                record.missing_outputs.append(name)
        if spec.verifier is not None:
            try:
                record.verification = dict(spec.verifier(workspace) or {})
            except Exception as exc:  # noqa: BLE001
                record.verification = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        problems: List[str] = []
        if record.exit_code != 0:
            problems.append(f"the agent exited with {record.exit_code}")
        if record.timed_out:
            problems.append("the agent timed out")
        if record.missing_outputs:
            problems.append("missing outputs: " + ", ".join(record.missing_outputs))
        if spec.verifier is not None and not record.verification.get("ok"):
            problems.append("the verifier refused the output")
        if record.elapsed_s > spec.max_seconds:
            problems.append(f"the agent ran {record.elapsed_s:.1f}s, over its {spec.max_seconds}s bound")
        if problems:
            record.error = "; ".join(problems)
        record.ok = not problems
        self._write(evidence, record)
        if not self.keep_workspaces:
            shutil.rmtree(workspace, ignore_errors=True)
        return record

    # -- evidence ---------------------------------------------------------
    def _write(self, evidence: Path, record: ExecutionRecord) -> None:
        (evidence / "execution.json").write_text(
            json.dumps(record.as_dict(), indent=2, default=str) + "\n", encoding="utf-8")

    def read_record(self, task_id: str) -> Optional[Dict[str, Any]]:
        path = self.evidence_dir(task_id) / "execution.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
