"""The agent runtime: OpenAI Agents SDK agents with a tool-enforced boundary.

The agents are created and run by the OpenAI Agents SDK (``agents.Agent`` and
``agents.Runner``) and they work in their own workspace with the project's
boundary as their TOOLS: a read outside the workspace and the policy-approved
roots is refused, a write outside the workspace is refused, and the progress
channel is the only place an agent records what it did. The task contract is the one the framework has always
read: the workspace is ``root/agents/<task_id>``, the answer is the file the
caller parses, and the run leaves ``execution.json`` with the boundary
fingerprint that says whether the answer may be reused.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import settings
from ..core.evidence import digest_bytes
from . import policy, progress_server

#: The roles that may not write anything. Their answer arrives through the
#: progress channel instead of a file they wrote.
READ_ONLY_ROLES = ("ideate", "score", "probe")

#: The append-only progress channel: the framework reads the answer from it, and
#: an agent records every stage there.
PROGRESS_FILE_NAME = "progress.jsonl"


def tool_permissions(workspace: Path, role: str = "verify", read_paths=None,
                     allow_shell: bool = False,
                     allow_write: bool = False) -> dict:
    """The boundary the tools enforce, as the framework records it.

    Reads are allowed inside the workspace and inside the declared read roots;
    writes are allowed inside the workspace, and only for a role that may write.
    The shell is not a tool, so a task that asks for it gets no shell.
    """
    roots = [str(Path(entry).resolve()) for entry in (read_paths or [])]
    return {
        "read_roots": roots,
        "write_root": str(Path(workspace).resolve()),
        "shell": bool(allow_shell),
        "write": bool(allow_write),
        "role": str(role),
    }


def effective_model() -> str:
    """The model the agents are created with."""
    return settings.AGENT_CHAT_MODEL


def _read_patterns(path: Path) -> list:
    """The permission patterns that name this path, under both spellings.

    The repository is reachable through a symlinked home, and an agent may be
    given either spelling; both spellings name the same file.
    """
    raw = Path(path)
    resolved = raw.resolve()
    suffix = "" if resolved.is_file() else "/**"
    patterns = [str(resolved) + suffix]
    if raw.is_absolute() and str(raw) != str(resolved):
        patterns.append(str(raw) + suffix)
    return list(dict.fromkeys(patterns))


def effective_read_paths(read_paths) -> list:
    """The read set an agent really runs with: the declared one plus the contract.

    The executor and the caller must agree on it: the fingerprint is computed
    from the allowlist the agent actually gets, so a cached answer is reused only
    when the same sources and the same contract were in front of the agent.
    """
    contract = _geometry_contract()
    paths = list(read_paths or [])
    if contract["path"] not in paths:
        paths.append(contract["path"])
    return paths


def boundary_fingerprint(workspace: Path, role: str, read_paths,
                         allow_shell: bool = False, allow_write: bool = False,
                         prompt: str = "", model: str = "",
                         source_digest: str = "") -> str:
    """The identity of one run: the boundary, the task and the model.

    Two runs with the same fingerprint had the same view of the filesystem and
    were given the same task; a different fingerprint means the evidence was
    produced under different rules and may not be reused as if it were current.
    The prompt and the model are part of it, so a re-asked task with a changed
    question or a changed model is not answered by the old answer.
    """
    payload = {
        "version": policy.BOUNDARY_VERSION,
        "role": str(role),
        "workspace": str(Path(workspace).resolve()),
        "allow_shell": bool(allow_shell),
        "allow_write": bool(allow_write),
        "read_paths": sorted(str(Path(entry).resolve()) for entry in (read_paths or [])),
        "permissions": tool_permissions(workspace, role, read_paths,
                                        allow_shell, allow_write),
        "prompt_sha256": digest_bytes(str(prompt).encode()),
        "model": str(model),
        # The shared mathematics is part of the identity of a run: a review made
        # under one convention cannot answer a question under another.
        "geometry_contract": _geometry_contract()["sha256"],
        "source_digest": str(source_digest),
    }
    return digest_bytes(json.dumps(payload, sort_keys=True).encode())


def saved_answer(workspace: Path, parser=None, assemble: bool = False) -> str:
    """The last saved record that carries a complete answer.

    A task that ran out of time still saved its records. The last one that
    parses as an answer is what the agent delivered; the caller decides whether
    a partial delivery counts (an idea does, a verdict does not).

    With ``assemble`` the records are joined in order and the parser is tried
    on the whole of them: a task that delivered its answer one block at a time
    (one proposal per record) did deliver the answer. The flag is not used for
    a verdict, where a stage record must never become the final judgement.
    """
    path = Path(workspace) / PROGRESS_FILE_NAME
    records = progress_server.read_records(path)
    if assemble and parser is not None and records:
        # A task that delivered its answer one block at a time did deliver the
        # answer: the blocks are joined in the order they were saved.
        joined = "\n\n".join(str(record.get("content", "")) for record in records
                             if str(record.get("content", "")).strip())
        if joined:
            try:
                parser(joined)
            except Exception:                              # noqa: BLE001
                pass
            else:
                return joined
    for record in reversed(records):
        content = str(record.get("content", "")).strip()
        if not content:
            continue
        if parser is None:
            return content
        try:
            parser(content)
        except Exception:                                  # noqa: BLE001
            continue
        return content
    if assemble and parser is not None and records:
        joined = "\n\n".join(str(record.get("content", "")) for record in records
                             if str(record.get("content", "")).strip())
        if joined:
            try:
                parser(joined)
            except Exception:                              # noqa: BLE001
                return ""
            return joined
    return ""


def _geometry_contract() -> dict:
    """The shared mathematical contract every agent reads, with its digest."""
    import hashlib
    path = Path(__file__).resolve().parents[1] / "contracts" / "HYPERBOLIC_GEOMETRY_CONTRACT.md"
    text = path.read_text(encoding="utf-8")
    version = "poincare-contract/v1"
    for line in text.splitlines():
        if line.startswith("Status:") and "`" in line:
            version = line.split("`")[1]
            break
    return {"path": str(path), "version": version,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class AgentFailure(RuntimeError):
    """An agent run that produced no usable answer, with the class of failure.

    ``auth`` means the provider refused the credentials: retrying the same call
    cannot help and the caller pauses. ``timeout``, ``cancel``, ``crash`` and
    ``empty`` are execution faults of one agent and are retried in place.
    """

    def __init__(self, failure_class: str, receipt: str, message: str):
        super().__init__(message)
        self.failure_class = failure_class
        self.receipt = receipt


def class_of_failure(receipt: Path, *, timed_out: bool, cancelled: bool,
                     exit_code: int, wrote_answer: bool) -> str:
    """Why an agent run produced no usable answer."""
    text = ""
    if receipt.is_file():
        try:
            text = receipt.read_text(encoding="utf-8", errors="replace")[-200000:]
        except OSError:
            text = ""
    lowered = text.lower()
    if "authentication" in lowered or "invalid api key" in lowered \
            or "401" in lowered:
        return "auth"
    if "quota" in lowered or "rate limit" in lowered:
        return "quota"
    if timed_out:
        return "timeout"
    if cancelled:
        return "cancel"
    if exit_code != 0:
        return "crash"
    return "empty" if not wrote_answer else "unusable"


def run_agent(task_id: str, prompt: str, output_name: str, root: Path, *,
              model=None, timeout=None, role: str = "verify", read_paths=None,
              allowed_roots=None, candidate_tree=None, allow_shell: bool = False,
              allow_write: bool = False, deadline_seconds: int | None = None,
              source_digest: str = "") -> Path:
    """Run one agent and return the path of its execution receipt.

    The agent is created and run by the OpenAI Agents SDK over the configured
    chat model; the boundary is enforced by the tools it is given. The receipt says whether the
    run is verified, and carries the boundary fingerprint that decides whether
    the answer may be reused.
    """
    from . import runtime
    return runtime.run_agent(
        task_id, prompt, output_name, Path(root), model=model, timeout=timeout,
        role=role, read_paths=read_paths, allowed_roots=allowed_roots,
        candidate_tree=candidate_tree, allow_shell=allow_shell,
        allow_write=allow_write, deadline_seconds=deadline_seconds,
        source_digest=source_digest)
