"""MCP-facing control API for a persistent, real, behavior-first research DAG.

No long work is performed in the MCP request. A worker is a real independent
Python process and OpenCode agents are genuine child processes of that worker.
"""
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .. import settings
from ..core.evidence import atomic_json, digest_file, read_json
from ..workflows import pipeline

RUN_ID = re.compile(r"^run_[0-9a-f]{16}$")
ROLES = ("behavior", "literature", "geometry", "review")
MAX_RESULT_CHARS = 4000


@contextmanager
def _launch_lock():
    path = settings.STATE_ROOT / "mcp" / "launch.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def _root(run_id):
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run_id")
    path = pipeline.run_dir(run_id)
    if not (path / "run.json").is_file():
        raise ValueError("unknown research run")
    return path


def _worker_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"tools.research.mcp.worker" in cmd


def _verify_worker_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"tools.research.mcp.verify_worker" in cmd


def _workers(root):
    out = []
    for path in sorted((root / "workers").glob("*.json")) if (root / "workers").is_dir() else ():
        try:
            row = read_json(path)
            out.append({"pid": row["pid"], "alive": _worker_alive(row["pid"]),
                        "started_at": row.get("started_at")})
        except (ValueError, KeyError, OSError):
            continue
    return out


def _spawn_worker(run_id, root):
    (root / "workers").mkdir(parents=True, exist_ok=True)
    target = root / "workers" / ("launch_" + uuid.uuid4().hex + ".log")
    with target.open("wb") as log:
        proc = subprocess.Popen(
            [settings.RESEARCH_PYTHON, "-m", "tools.research.mcp.worker"],
            cwd=str(settings.ROOT), stdin=subprocess.PIPE, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            proc.stdin.write((run_id + "\n").encode("ascii"))
            proc.stdin.close()
        except OSError:
            proc.terminate()
            raise
    atomic_json(target.with_suffix(".json"), {
        "pid": proc.pid, "started_at": time.time(),
        "run_id": run_id, "log": target.name,
    })
    return proc.pid


def _launch_if_possible(run_id, root):
    if (root / "cancel.json").is_file():
        return {"workers_started": [], "reason": "cancel_requested"}
    state = pipeline.store().snapshot()["tasks"]
    prefix = run_id + "__"
    tasks = {name: state.get(prefix + name, {}) for name in ROLES}
    running = [task for task in tasks.values() if task.get("status") in ("running", "cancelling")]
    if running and any(worker["alive"] for worker in _workers(root)):
        return {"workers_started": [], "reason": "already_running"}
    # The source-defined safety switch cannot be bypassed by MCP arguments.
    roles = ROLES if settings.LIVE_AGENTS_ALLOWED else ("behavior",)
    eligible = [name for name in roles if tasks[name].get("status") == "pending"
                and all(tasks[dep].get("status") == "done"
                        for dep in dict((n, d) for n, _, d in pipeline.TASKS)[name])]
    expired = any(task.get("status") == "running" and task.get("lease_until", 0) < time.time()
                  for task in tasks.values())
    if not eligible and not expired:
        return {"workers_started": [], "reason": (
            "agent_execution_disabled" if not settings.LIVE_AGENTS_ALLOWED
            else "no_ready_tasks")}
    # Allow DATA/LITERATURE parallelism; each worker claims from the SAME root DAG.
    count = min(2, len(eligible)) if eligible and settings.LIVE_AGENTS_ALLOWED else 1
    return {"workers_started": [_spawn_worker(run_id, root) for _ in range(count)],
            "reason": "dispatched"}


def research_start():
    with _launch_lock():
        run_id = "run_" + uuid.uuid4().hex[:16]
        root = settings.STATE_ROOT / "runs" / run_id
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "run.json", {
            "schema": 1, "run_id": run_id, "created_at": time.time(),
            "objective": "Behavior-grounded hyperbolic semantic IDs for next-item recommendation",
            "frozen_protocol": "Issue116 / CLAUDE.md", "autorun_gpu": False,
        })
        pipeline.bootstrap_run(run_id)
        dispatch = _launch_if_possible(run_id, root)
    return {"run_id": run_id, **dispatch, "gpu_launched": False,
            "next": "research_status"}


def research_resume(run_id):
    with _launch_lock():
        root = _root(run_id)
        dispatch = _launch_if_possible(run_id, root)
    return {"run_id": run_id, **dispatch, "gpu_launched": False}


def _task_view(run_id):
    tasks = pipeline.store().snapshot()["tasks"]
    prefix = run_id + "__"
    return {name: {"status": tasks[prefix + name]["status"],
                   "attempts": tasks[prefix + name]["attempts"],
                   "failure": tasks[prefix + name].get("failure"),
                   "evidence_sha256": (tasks[prefix + name].get("evidence") or {}).get("sha256")}
            for name in ROLES if prefix + name in tasks}


def research_status(run_id):
    root = _root(run_id)
    tasks = _task_view(run_id)
    statuses = [item["status"] for item in tasks.values()]
    workers = _workers(root)
    active = any(w["alive"] for w in workers)
    if all(x == "done" for x in statuses) and len(statuses) == len(ROLES):
        status = "completed"
    elif (root / "cancel.json").is_file():
        status = "cancelling" if "cancelling" in statuses or active else "cancelled"
    elif any(x == "blocked" for x in statuses):
        status = "blocked"
    elif any(x == "running" for x in statuses):
        status = "running" if active else "worker_down"
    elif all(x == "pending" for x in statuses):
        status = "queued"
    else:
        status = "waiting_external" if not settings.LIVE_AGENTS_ALLOWED else "ready_to_resume"
    return {"run_id": run_id, "status": status, "tasks": tasks,
            "workers": workers, "real_agent_enabled": settings.LIVE_AGENTS_ALLOWED,
            "gpu_launched_by_research": False}


def research_results(run_id):
    root = _root(run_id)
    tasks = _task_view(run_id)
    results = {}
    locations = {
        "behavior": root / "behavior.json",
        "literature": root / "agents" / "literature" / "literature.json",
        "geometry": root / "agents" / "geometry" / "candidate.json",
        "review": root / "agents" / "review" / "review.json",
    }
    dag = pipeline.store().snapshot()["tasks"]
    for name, path in locations.items():
        evidence = (dag.get(run_id + "__" + name) or {}).get("evidence") or {}
        if tasks.get(name, {}).get("status") != "done" or not path.is_file():
            continue
        if name == "behavior":
            expected = evidence.get("sha256")
        else:
            receipt = root / "agents" / name / "execution.json"
            if not receipt.is_file() or digest_file(receipt) != evidence.get("sha256"):
                results[name] = {"verified": False, "refused": "agent execution receipt hash drift"}
                continue
            expected = read_json(receipt).get("output_sha256")
        try:
            if expected != digest_file(path):
                results[name] = {"verified": False, "refused": "evidence hash drift"}
                continue
            text = path.read_text(encoding="utf-8")
            results[name] = {"verified": True, "sha256": expected,
                             "preview": text[:MAX_RESULT_CHARS],
                             "truncated": len(text) > MAX_RESULT_CHARS}
        except (OSError, ValueError):
            results[name] = {"verified": False, "refused": "unreadable evidence"}
    return {"run_id": run_id, "status": research_status(run_id)["status"],
            "artifacts": results}


def research_cancel(run_id):
    with _launch_lock():
        root = _root(run_id)
        tasks = _task_view(run_id)
        if len(tasks) == len(ROLES) and all(x["status"] == "done" for x in tasks.values()):
            return {"run_id": run_id, "cancel_requested": False,
                    "reason": "already_completed"}
        atomic_json(root / "cancel.json", {"requested_at": time.time(),
                                          "reason": "MCP user cancellation"})
        changed = pipeline.store().cancel_prefix(run_id + "__")
    return {"run_id": run_id, "cancel_requested": True,
            "tasks_changed": changed,
            "note": "Running OpenCode processes are terminated cooperatively; status may briefly be cancelling."}
