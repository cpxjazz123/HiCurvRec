"""The LangChain backend for the research agents.

The OpenCode backend starts a CLI in a bubblewrap mount namespace. This backend
creates the agent with LangChain instead: ``langchain.agents.create_agent`` over a
chat model, with the project's own boundary expressed as the agent's TOOLS. The
task contract does not change: the agent runs in its own workspace under
``root/agents/<task_id>``, its answer is a file the caller parses, its progress
records travel through the same progress channel, and the run leaves the same
receipt.

The boundary is enforced by the tools, not by a mount namespace: a read outside
the workspace and the policy-approved roots is refused, and a write outside the
workspace is refused. That is a weaker isolation than bubblewrap, so the backend
is opt-in (``settings.AGENT_BACKEND = "langchain"``) and it refuses to run
without a configured chat model.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from tools.research import settings
from tools.research.agents import policy, progress_server, runner

SYSTEM_PROMPT = (
    "You are a research agent of a scientific programme. Work inside your own "
    "workspace, read only what your task allows, and answer in the exact format "
    "the task asks for. Record every meaningful step with the append_progress "
    "tool: each record must stand on its own, and the complete answer goes in "
    "the last record."
)


class AgentBackendUnavailable(RuntimeError):
    """The LangChain backend has no chat model to create an agent with."""


def available() -> bool:
    """Whether a chat model is configured for this backend."""
    return bool((settings.AGENT_CHAT_MODEL or "").strip())


def build_model(model_id: str | None = None):
    """The chat model the agents are created with."""
    identifier = (model_id or settings.AGENT_CHAT_MODEL or "").strip()
    if not identifier:
        raise AgentBackendUnavailable(
            "the LangChain backend needs a chat model: set AGENT_CHAT_MODEL "
            "(for example 'anthropic:claude-sonnet-4-5' or 'openai:gpt-4o-mini'). "
            "The key and the base URL come from the environment "
            "(ANTHROPIC_API_KEY / OPENAI_API_KEY, and their base URLs).")
    from langchain.chat_models import init_chat_model
    return init_chat_model(identifier)


def workspace_tools(workspace: Path, read_paths, *, allow_write: bool = False,
                    progress=None):
    """The tools a LangChain agent may use, each with the project's boundary."""
    from langchain_core.tools import tool

    workspace = Path(workspace).resolve()
    allowed = [Path(entry).resolve() for entry in (read_paths or [])]

    def _readable(candidate: Path) -> bool:
        resolved = candidate.resolve()
        if resolved == workspace or workspace in resolved.parents:
            return True
        return any(resolved == entry or entry in resolved.parents
                   for entry in allowed)

    def _writable(candidate: Path) -> bool:
        resolved = candidate.resolve()
        return resolved == workspace or workspace in resolved.parents

    @tool
    def read_file(path: str) -> str:
        """Read a file inside the task workspace or an allowed read root."""
        target = Path(path)
        if not target.is_absolute():
            target = workspace / target
        if not _readable(target):
            raise ValueError("refused: " + str(target) + " is outside the allowed roots")
        return target.read_text(encoding="utf-8", errors="replace")

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a file inside the task workspace."""
        if not allow_write:
            raise ValueError("this task has no write tool")
        target = Path(path)
        if not target.is_absolute():
            target = workspace / target
        if not _writable(target):
            raise ValueError("refused: " + str(target) + " is outside the workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return "wrote " + str(target)

    @tool
    def list_directory(path: str = ".") -> str:
        """List a directory inside the task workspace or an allowed read root."""
        target = Path(path)
        if not target.is_absolute():
            target = workspace / target
        if not _readable(target):
            raise ValueError("refused: " + str(target) + " is outside the allowed roots")
        if not target.is_dir():
            raise ValueError("not a directory: " + str(target))
        return "\n".join(sorted(entry.name + ("/" if entry.is_dir() else "")
                                for entry in target.iterdir()))

    @tool
    def append_progress(content: str) -> str:
        """Append one stage record to this task's own progress file."""
        if progress is None:
            raise ValueError("this task has no progress channel")
        saved = progress.append(content)
        return "saved as record " + str(saved["seq"]) + " of this task's progress file"

    return [read_file, write_file, list_directory, append_progress]


def run_agent(task_id: str, prompt: str, output_name: str, root: Path, *,
              model=None, timeout=None, role: str = "verify", read_paths=None,
              allowed_roots=None, candidate_tree=None, allow_shell: bool = False,
              allow_write: bool = False, deadline_seconds: int | None = None,
              source_digest: str = "") -> Path:
    """Run one agent with a LangChain agent and return its receipt path."""
    if (not output_name.endswith((".json", ".txt")) or "/" in output_name
            or "\\" in output_name
            or not task_id.replace("-", "").replace("_", "").isalnum()):
        raise ValueError("unsafe task id or output filename")
    root = Path(root)
    workspace = root / "agents" / task_id
    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / output_name
    if output.exists():
        raise RuntimeError("refuse to overwrite previous role output")
    contract = runner._geometry_contract()
    read_paths = list(read_paths or []) + (
        [] if contract["path"] in (read_paths or []) else [contract["path"]])
    if allowed_roots is not None:
        allowed_roots = list(allowed_roots) + [str(Path(contract["path"]).parent)]
    allowed = policy.check_read_paths(read_paths, roots=allowed_roots,
                                      task_root=root, candidate_tree=candidate_tree)
    deadline = (time.time() + float(deadline_seconds)) if deadline_seconds else None
    progress = progress_server.ProgressFile(workspace / runner.PROGRESS_FILE_NAME,
                                             task_id, deadline=deadline)
    tools = workspace_tools(workspace, allowed, allow_write=allow_write,
                            progress=progress)
    chat_model = build_model(model)
    from langchain.agents import create_agent
    agent = create_agent(model=chat_model, tools=tools, system_prompt=SYSTEM_PROMPT)
    started = time.time()
    answer_text = ""
    failure = ""
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                              config={"recursion_limit": 60})
        messages = result.get("messages") or []
        for message in reversed(messages):
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                answer_text = content.strip()
                break
    except Exception as exc:                                   # noqa: BLE001
        failure = type(exc).__name__ + ": " + str(exc)[:300]
    if answer_text and not output.is_file():
        # A read-only role has no write tool: its reply IS the answer.
        output.write_text(answer_text + "\n", encoding="utf-8")
    if answer_text:
        try:
            progress.append(answer_text)
        except Exception:                                      # noqa: BLE001
            pass
    boundary = runner.boundary_fingerprint(
        workspace, role, runner.effective_read_paths(read_paths), allow_shell,
        allow_write, prompt, model or settings.AGENT_CHAT_MODEL,
        source_digest=source_digest)
    receipt = {
        "task_id": task_id, "backend": "langchain", "verified": bool(output.is_file()
                                                                     and not failure),
        "finished_utc_epoch": time.time(), "elapsed_s": round(time.time() - started, 3),
        "model": model or settings.AGENT_CHAT_MODEL, "role": role,
        "boundary_sha256": boundary, "read_allowlist": [str(entry) for entry in allowed],
        "geometry_contract": contract, "failure": failure,
        "progress_file": str(workspace / runner.PROGRESS_FILE_NAME),
        "answer": str(output),
    }
    receipt_path = workspace / "execution.json"
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    if failure and not output.is_file():
        raise runner.AgentFailure("AGENT_CRASHED", str(receipt_path), failure)
    return receipt_path
