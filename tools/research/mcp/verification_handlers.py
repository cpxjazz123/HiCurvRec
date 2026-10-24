"""Asynchronous MCP endpoints for independent verification Agents."""
import fcntl
import json
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .. import settings
from ..core.evidence import atomic_json, read_json
from . import handlers
from ..verification.admission import source_digest

SOURCE_DIR = settings.ROOT / "stage2_RQ-VAE"
SAFE_ID = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{2,63}$")


def _source(source_dir):
    if not isinstance(source_dir, str) or not source_dir:
        raise ValueError("source_dir required")
    tree = Path(source_dir)
    if not tree.is_absolute():
        tree = settings.ROOT / tree
    tree = tree.resolve(strict=True)
    if SOURCE_DIR.resolve() not in tree.parents or tree.name in (
            "TIGER_RQ-VAE", "TIGER_BEHAVIOR_SAMPLE_RQ-VAE"):
        raise ValueError("requires a distinct Stage2 candidate source tree")
    if not (tree / "train_rqvae.py").is_file():
        raise ValueError("missing candidate trainer")
    return tree


@contextmanager
def _lock(root):
    path = root / "verification_requests" / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def _candidate(run_id):
    root = handlers._root(run_id)
    if (root / "cancel.json").is_file():
        raise ValueError("research run cancelled")
    if handlers.research_status(run_id)["tasks"].get("review", {}).get("status") != "done":
        raise ValueError("scientific REVIEW must pass before verifier Agents")
    candidate = read_json(root / "agents" / "geometry" / "candidate.json")
    cid = candidate.get("candidate_id")
    if not isinstance(cid, str) or not SAFE_ID.fullmatch(cid):
        raise ValueError("candidate ID unsafe or missing")
    return cid


def _start(run_id, source_dir, action):
    if not settings.LIVE_AGENTS_ALLOWED:
        return {"started": False, "reason": "live_opencode_disabled_in_settings"}
    cid = _candidate(run_id)
    tree = _source(source_dir)
    root = handlers._root(run_id)
    path = root / "verification_requests" / (action + ".json")
    with _lock(root):
        source_sha = source_digest(tree)
        if path.is_file():
            existing = read_json(path)
            stale = existing.get("source_sha256") not in (None, source_sha)
            if (existing.get("status") not in ("failed", "cancelled", "superseded")
                    and not stale):
                return {"started": False, "reason": "request_already_recorded",
                        "status": existing.get("status"), "request_id": existing.get("request_id")}
            # A failed attempt must not block the next one, and a request made
            # against a source that has since changed is stale evidence. The old
            # record is kept under its own request id and superseded with a
            # fresh one, so earlier evidence is never overwritten.
            keep = path.with_name(path.stem + "." + str(existing.get("request_id", "old"))[:8] + ".json")
            path.replace(keep)
        if action == "review":
            draft = root / "verification_requests" / "draft.json"
            if not draft.is_file() or read_json(draft).get("status") != "drafts_ready":
                return {"started": False, "reason": "drafts_not_ready"}
        record = {"schema": 1, "request_id": uuid.uuid4().hex,
                  "run_id": run_id, "candidate_id": cid, "source_dir": str(tree),
                  "source_sha256": source_sha,
                  "action": action, "created_at": time.time(), "status": "queued"}
        atomic_json(path, record)
        try:
            with path.with_suffix(".log").open("wb") as log:
                proc = subprocess.Popen(
                    [settings.RESEARCH_PYTHON, "-m", "tools.research.mcp.verify_worker"],
                    cwd=str(settings.ROOT), stdin=subprocess.PIPE, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                record.update(pid=proc.pid, status="running")
                atomic_json(path, record)
                proc.stdin.write((json.dumps(record) + "\n").encode("utf-8"))
                proc.stdin.close()
        except OSError as exc:
            record.update(status="failed", error=type(exc).__name__)
            atomic_json(path, record)
            raise
    return {"started": True, "run_id": run_id, "candidate_id": cid,
            "request_id": record["request_id"], "action": action,
            "pid": proc.pid, "gpu_launched": False}


def research_verify_start(run_id, source_dir):
    return _start(run_id, source_dir, "draft")


def research_verify_review(run_id, source_dir):
    return _start(run_id, source_dir, "review")


def research_verify_status(run_id):
    root = handlers._root(run_id)
    actions = {}
    for action in ("draft", "review"):
        path = root / "verification_requests" / (action + ".json")
        if not path.is_file():
            continue
        row = read_json(path)
        status = row.get("status")
        if status == "running" and not handlers._verify_worker_alive(row.get("pid")):
            status = "worker_down"
        actions[action] = {
            "status": status, "request_id": row.get("request_id"),
            "candidate_id": row.get("candidate_id"), "pid": row.get("pid"),
            "error": row.get("error"), "manifest": row.get("manifest")}
    return {"run_id": run_id, "actions": actions}
