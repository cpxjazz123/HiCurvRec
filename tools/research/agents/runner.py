"""Real OpenCode CLI with cooperative cancellation and process receipts."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from .. import settings
from ..core.evidence import atomic_json, digest_bytes, digest_file


def isolation_permissions(workspace: Path, role: str = "verify") -> dict:
    """The opencode permission object that confines an agent to its workspace.

    opencode v2 reads a single ``permission`` object whose keys are the tool
    names and whose values are ``ask``/``allow``/``deny``. An earlier version of
    this function wrote a ``permissions`` array of action/resource/effect
    rules; opencode ignores that shape, so every agent kept a full shell and the
    boundary was never in force. The roles are now stated in the schema the
    binary actually reads.

    A verifier reviews code and does not run it, so it has no shell. The
    verifier and the repair planner are **read-only**: they get neither a shell
    nor an edit tool, so they cannot change the candidate tree or anything
    else. Their answers are taken from the session stream, where the agent
    printed them, and the framework writes the file.

    An implementer must write the tree and run the smoke test, so it keeps the
    shell and the edit tool inside the workspace, bounded by prompt and by the
    framework's output-path preflight, because this opencode version has no
    per-path edit rule.
    """
    runs_code = str(role) == "implementation"
    read_only = str(role) in ("verify", "repair", "ideate", "score")
    return {
        "bash": "allow" if runs_code else "deny",
        "edit": "deny" if read_only else "allow",
        "write": "deny" if read_only else "allow",
        "webfetch": "deny",
        "external_directory": "deny",
    }


#: The roles that may not write anything. Their answer arrives through the
#: session stream instead of a file they wrote.
READ_ONLY_ROLES = ("verify", "repair", "ideate", "score")


def session_answer(path: Path) -> str:
    """The text an agent printed, read back from the opencode session stream.

    A read-only role cannot create its answer file, so the framework takes the
    answer from the session and writes the file itself.
    """
    chunks: list[str] = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") != "text":
            continue
        part = event.get("part") or {}
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
    return "\n".join(chunks).strip()


def write_isolation_config(workspace: Path, role: str = "verify") -> Path:
    """Write the agent's private permission boundary next to its session."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "opencode.json"
    config = {"$schema": "https://opencode.ai/config.json",
              "permission": isolation_permissions(workspace, role)}
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def mark_workspace_project(workspace: Path) -> Path:
    """Make the workspace opencode's OWN project root.

    opencode resolves its project by walking up to the enclosing git root, so a
    workspace inside the repository makes the REPOSITORY the project: the task's
    own opencode.json is then ignored, the permission boundary never applies and
    the Agent writes into the baseline trees. An empty .git/ directory stops the
    walk at the workspace, so the boundary in opencode.json is the one that
    applies and the shell stays denied. The marker is empty on purpose: it is a
    project marker, not a repository.
    """
    marker = Path(workspace) / ".git"
    marker.mkdir(parents=True, exist_ok=True)
    return marker


#: How long a role may take. The implementer writes a mechanism and runs its
#: smoke test, which does not fit the review budget; the reviewers only read.
#: The reviewer budget follows the model: DeepSeek V4.1 Flash needs longer on
#: the long review prompts than the previous model did, and a reviewer that is
#: cancelled for time leaves the round without a verdict.
ROLE_TIMEOUT_SECONDS = {
    "implementation": 3600,
    "repair": 2400,
    "verify": 2400,
    "ideate": 2400,
    "scoring": 2400,
}


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


def class_of_failure(session: Path, *, timed_out: bool, cancelled: bool,
                     exit_code: int, wrote_answer: bool) -> str:
    """Why an agent run produced no usable answer."""
    text = ""
    if session.is_file():
        try:
            text = session.read_text(encoding="utf-8", errors="replace")[-200000:]
        except OSError:
            text = ""
    if "provider.auth" in text or "authentication token has expired" in text:
        return "auth"
    if "provider.quota" in text or "rate limit" in text.lower():
        return "quota"
    if timed_out:
        return "timeout"
    if cancelled:
        return "cancel"
    if exit_code != 0:
        return "crash"
    return "empty" if not wrote_answer else "unusable"


def run_agent(task_id: str, prompt: str, output_name: str, root: Path,
              binary=None, model=None, timeout=None,
              role: str = "verify") -> Path:
    # The binary and the model come from the one source of configuration.
    binary = binary or settings.AGENT_BINARY
    model = model or settings.AGENT_MODEL
    # The role budget is the DEFAULT, and an explicit timeout wins: a caller that
    # asks for two seconds (a test, or a deliberate short probe) must not be
    # silently given the reviewer's forty minutes.
    if timeout is None:
        timeout = ROLE_TIMEOUT_SECONDS.get(str(role)) or settings.AGENT_TIMEOUT_SECONDS
    if (not output_name.endswith((".json", ".txt")) or "/" in output_name
            or "\\" in output_name or not task_id.replace("-", "").replace("_", "").isalnum()):
        raise ValueError("unsafe task id or output filename")
    root = Path(root)
    workspace = root / "agents" / task_id
    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / output_name
    if output.exists():
        raise RuntimeError("refuse to overwrite previous role output")
    isolation = write_isolation_config(workspace, role)
    marker = mark_workspace_project(workspace)
    cmd = [binary, "run", "--standalone", "--format", "json", "--auto",
           "--model", model, "--title", task_id, prompt]
    started = time.monotonic()
    cancelled, timed_out, stderr = False, False, ""
    try:
        with (workspace / "session.jsonl").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=workspace, stdout=log,
                                       stderr=subprocess.PIPE, text=True,
                                       start_new_session=True)
            atomic_json(workspace / "running.json", {"pid": process.pid,
                        "started_at": time.time(), "task_id": task_id})
            while True:
                try:
                    _, stderr = process.communicate(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if (root / "cancel.json").is_file():
                        cancelled = True
                    if time.monotonic() - started > timeout:
                        timed_out = True
                    if cancelled or timed_out:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            _, stderr = process.communicate(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            _, stderr = process.communicate()
                        break
    except OSError as exc:
        atomic_json(workspace / "execution.json",
                    {"task_id": task_id, "error_type": type(exc).__name__,
                     "verified": False})
        raise
    finally:
        (workspace / "running.json").unlink(missing_ok=True)
    # The Agent is told the absolute destination, but a model can still write the
    # answer into a nested or slightly misnamed directory of its own workspace.
    # The artifact is what matters and the caller validates its content
    # (candidate identity and source digest), so accept a bounded relocation and
    # record where it landed instead of discarding a real session.
    relocated = None
    if not output.is_file() and str(role) in READ_ONLY_ROLES:
        # A read-only role printed its answer instead of writing it.
        printed = session_answer(workspace / "session.jsonl")
        if printed:
            output.write_text(printed, encoding="utf-8")
    if not output.is_file():
        matches = sorted(workspace.rglob(output_name))
        # A repaired run archives older answers under attempt-N/. Never
        # "recover" an archived candidate/review as though this subprocess
        # had produced it: that would make a failed retry look successful.
        matches = [match for match in matches
                   if match.is_file()
                   and ".." not in match.relative_to(workspace).parts
                   and not any(part.startswith("attempt-")
                               for part in match.relative_to(workspace).parts)]
        if matches:
            relocated = matches[0]
            output = relocated
    valid = False
    if output.is_file() and process.returncode == 0 and not timed_out and not cancelled:
        try:
            text = output.read_text(encoding="utf-8")
            if output_name.endswith(".json"):
                valid = isinstance(json.loads(text), dict)
            else:
                # A plain-text answer only has to exist and say something: the
                # caller reads it and decides whether it is usable.
                valid = bool(text.strip())
        except (OSError, ValueError):
            valid = False
    record = {
        "task_id": task_id, "pid": process.pid,
        "finished_utc_epoch": time.time(), "exit_code": process.returncode,
        "timed_out": timed_out, "cancelled": cancelled, "model": model,
        "argv_shape": ["opencode", "run", "--standalone", "--format", "json",
                       "--auto", "--model", "<model>", "--title", task_id, "<prompt>"],
        "prompt_sha256": digest_bytes(prompt.encode()),
        "session_sha256": digest_file(workspace / "session.jsonl"),
        "isolation_config": isolation.name,
        "isolation_config_sha256": digest_file(isolation),
        "project_marker": marker.name,
        "isolation_policy": "opencode v2 permission object: bash " +
                            ("allowed (this role runs the smoke test)" if role == "implementation"
                             else "denied") +
                            ", edit and write " +
                            ("denied (this role is read-only and its answer is read back from "
                             "the session stream)" if role in READ_ONLY_ROLES else "allowed") +
                            ", webfetch and external_directory denied; the workspace is its "
                            "own opencode project so its opencode.json applies",
        "stderr_present": bool(stderr), "output_name": output_name,
        "output_sha256": digest_file(output) if output.is_file() else None,
        "verified": valid,
        "limitations": "A cwd is NOT a sandbox; deployment must confine actual agent tools.",
    }
    record["failure_class"] = "none" if valid else class_of_failure(
        workspace / "session.jsonl", timed_out=timed_out, cancelled=cancelled,
        exit_code=process.returncode, wrote_answer=output.is_file())
    receipt = workspace / "execution.json"
    atomic_json(receipt, record)
    if not valid:
        # An answer is published only when the run was valid. A timed-out run can
        # leave half a report behind, and a half report read as an answer is how a
        # cancelled reviewer became a verdict.
        if output.is_file():
            partial = workspace / "partial"
            partial.mkdir(parents=True, exist_ok=True)
            output.replace(partial / output_name)
        raise AgentFailure(record["failure_class"], str(receipt),
                           "OpenCode did not produce a usable answer (" +
                           record["failure_class"] + "): " + str(receipt))
    return receipt
