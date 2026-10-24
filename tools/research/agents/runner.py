"""Actual OpenCode subprocess, isolated working directory, immutable execution receipt.

Deployment MUST independently restrict the OpenCode process to its workspace;
a cwd alone is not an OS sandbox. Live mode defaults off until checked.
"""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from ..core.evidence import atomic_json, digest_bytes, digest_file


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
    cmd = [binary, "run", "--standalone", "--format", "json", "--auto",
           "--model", model, "--title", task_id, prompt]
    started = time.time()
    try:
        with (workspace / "session.jsonl").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=workspace, stdout=log,
                                       stderr=subprocess.PIPE, text=True,
                                       start_new_session=True)
            timed_out = False
            try:
                _, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                _, stderr = process.communicate()
                timed_out = True
    except OSError as exc:
        atomic_json(workspace / "execution.json",
                    {"task_id": task_id, "started": started, "error_type": type(exc).__name__,
                     "verified": False})
        raise
    valid = False
    if output.is_file() and process.returncode == 0 and not timed_out:
        try:
            valid = isinstance(json.loads(output.read_text(encoding="utf-8")), dict)
        except (OSError, ValueError):
            valid = False
    record = {
        "task_id": task_id, "pid": process.pid,
        "started_utc_epoch": started, "finished_utc_epoch": time.time(),
        "exit_code": process.returncode, "timed_out": timed_out,
        "model": model,
        "argv_shape": ["opencode", "run", "--standalone", "--format", "json",
                       "--auto", "--model", "<model>", "--title", task_id, "<prompt>"],
        "prompt_sha256": digest_bytes(prompt.encode()),
        "session_sha256": digest_file(workspace / "session.jsonl"),
        "stderr_present": bool(stderr), "output_name": output_name,
        "output_sha256": digest_file(output) if output.is_file() else None,
        "verified": valid,
        "limitations": "No OS-level sandbox guarantee; live deployment requires filesystem tool permissions.",
    }
    receipt = workspace / "execution.json"
    atomic_json(receipt, record)
    if not valid:
        raise RuntimeError("OpenCode failed, timed out or wrote invalid JSON: " + str(receipt))
    return receipt
