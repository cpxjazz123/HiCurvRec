"""Real OpenCode CLI with cooperative cancellation and process receipts."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from .. import settings
from ..core.evidence import atomic_json, digest_bytes, digest_file


def isolation_permissions(workspace: Path) -> list:
    """The opencode v2 rules that confine an agent to its own workspace.

    opencode v2 matches permission rules in order and the LAST match wins, so
    the order here is the policy: the broad edit allow covers the agent's own
    Location (a relative path inside the workspace), the narrower deny then
    blocks an absolute write into the repository, and the final allow re-opens
    the agent's own workspace for absolute paths. external_directory blocks
    every path outside the repository, and shell plus execute are denied
    outright, so the agent cannot reach a shell through a tool or through
    Code Mode.
    """
    workspace = Path(workspace).resolve()
    return [
        {"action": "shell", "resource": "*", "effect": "deny"},
        {"action": "execute", "resource": "*", "effect": "deny"},
        {"action": "external_directory", "resource": "*", "effect": "deny"},
        {"action": "edit", "resource": "*", "effect": "allow"},
        {"action": "edit", "resource": str(settings.ROOT) + "/*", "effect": "deny"},
        {"action": "edit", "resource": str(workspace) + "/*", "effect": "allow"},
        # A relative path that climbs out of the workspace escapes every rule
        # above, because the matcher sees the raw relative string. Deny any
        # resource containing a parent traversal, and deny it LAST so that it
        # wins over the workspace allow.
        {"action": "edit", "resource": "../*", "effect": "deny"},
        {"action": "edit", "resource": "*/../*", "effect": "deny"},
        {"action": "edit", "resource": "*..*", "effect": "deny"},
    ]


def write_isolation_config(workspace: Path) -> Path:
    """Write the agent's private permission boundary next to its session."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "opencode.json"
    config = {"$schema": "https://opencode.ai/config.json",
              "permissions": isolation_permissions(workspace)}
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


def run_agent(task_id: str, prompt: str, output_name: str, root: Path,
              binary="opencode", model="openai/gpt-6-luna", timeout=900) -> Path:
    if (not output_name.endswith(".json") or "/" in output_name
            or "\\" in output_name or not task_id.replace("-", "").replace("_", "").isalnum()):
        raise ValueError("unsafe task id or output filename")
    root = Path(root)
    workspace = root / "agents" / task_id
    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / output_name
    if output.exists():
        raise RuntimeError("refuse to overwrite previous role output")
    isolation = write_isolation_config(workspace)
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
    if not output.is_file():
        matches = sorted(workspace.rglob(output_name))
        matches = [match for match in matches
                   if match.is_file() and ".." not in match.relative_to(workspace).parts]
        if matches:
            relocated = matches[0]
            output = relocated
    valid = False
    if output.is_file() and process.returncode == 0 and not timed_out and not cancelled:
        try:
            valid = isinstance(json.loads(output.read_text(encoding="utf-8")), dict)
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
        "isolation_policy": "v2 permissions: shell/execute denied, external_directory denied, "
                            "edit limited to the agent workspace; the workspace is its own "
                            "opencode project so the boundary is the one that applies",
        "stderr_present": bool(stderr), "output_name": output_name,
        "output_sha256": digest_file(output) if output.is_file() else None,
        "verified": valid,
        "limitations": "A cwd is NOT a sandbox; deployment must confine actual agent tools.",
    }
    receipt = workspace / "execution.json"
    atomic_json(receipt, record)
    if not valid:
        raise RuntimeError("OpenCode failed, cancelled, timed out or returned invalid JSON: " + str(receipt))
    return receipt
