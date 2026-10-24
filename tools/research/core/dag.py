"""One atomic root DAG across all Issue116 research runs."""
import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path
from .evidence import atomic_json, digest_file, read_json


class StateError(RuntimeError):
    pass


class TaskStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "dag" / "state.json"
        self.lock_path = self.root / "dag" / ".lock"
        self.events = self.root / "dag" / "events.jsonl"

    @contextmanager
    def locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)

    def _load(self):
        return read_json(self.path) if self.path.is_file() else {"schema": 2, "tasks": {}}

    def snapshot(self):
        with self.locked():
            return self._load()

    def _save(self, state, event):
        atomic_json(self.path, state)
        with self.events.open("a", encoding="utf-8") as fd:
            fd.write(json.dumps({"at": time.time(), **event}, sort_keys=True) + "\n")
            fd.flush()

    def add(self, task_id, role, dependencies=()):
        with self.locked():
            state = self._load()
            tasks = state["tasks"]
            deps = list(dependencies)
            if task_id in deps or any(dep not in tasks for dep in deps):
                raise StateError("unknown or circular dependency")
            if task_id in tasks:
                if tasks[task_id]["role"] != role or tasks[task_id]["dependencies"] != deps:
                    raise StateError("cannot redefine task")
                return
            tasks[task_id] = {"role": role, "dependencies": deps, "status": "pending",
                              "attempts": 0, "owner": None, "lease_until": 0,
                              "evidence": None}
            self._save(state, {"event": "add", "task": task_id})

    def claim(self, owner, lease_seconds=1200, max_attempts=2, roles=None, prefix=None):
        if not owner or lease_seconds <= 0 or max_attempts <= 0:
            raise ValueError("owner, lease and attempts must be positive")
        with self.locked():
            state = self._load()
            tasks = state["tasks"]
            now = time.time()
            for task_id, task in tasks.items():
                if prefix and not task_id.startswith(prefix):
                    continue
                if task["status"] == "running" and task["lease_until"] < now:
                    task["status"], task["owner"] = "pending", None
                    self._save(state, {"event": "lease_expired", "task": task_id})
            for task_id, task in tasks.items():
                if (prefix and not task_id.startswith(prefix)
                        or task["status"] != "pending" or task["attempts"] >= max_attempts
                        or (roles is not None and task["role"] not in roles)
                        or not all(tasks[dep]["status"] == "done"
                                   for dep in task["dependencies"])):
                    continue
                task.update(status="running", owner=owner, attempts=task["attempts"] + 1,
                            lease_until=now + lease_seconds)
                self._save(state, {"event": "claim", "task": task_id, "owner": owner})
                return task_id, task["role"]
            return None

    def complete(self, task_id, owner, artifact: Path, metadata=None):
        artifact = Path(artifact).resolve()
        if not artifact.is_file():
            raise StateError("a DONE state requires a real evidence file")
        evidence = dict(metadata or {})
        evidence.update(path=str(artifact), sha256=digest_file(artifact))
        with self.locked():
            state = self._load()
            task = state["tasks"].get(task_id)
            if not task or task["status"] != "running" or task["owner"] != owner:
                raise StateError("task ownership mismatch or task was cancelled")
            task.update(status="done", owner=None, lease_until=0,
                        evidence=evidence, finished_at=time.time())
            self._save(state, {"event": "done", "task": task_id,
                               "sha256": evidence["sha256"]})
        return evidence

    def fail(self, task_id, owner, reason, retry=False):
        with self.locked():
            state = self._load()
            task = state["tasks"].get(task_id)
            if not task or task["status"] != "running" or task["owner"] != owner:
                raise StateError("task ownership mismatch or task was cancelled")
            task.update(status="pending" if retry else "blocked",
                        owner=None, lease_until=0, failure=str(reason)[:300])
            self._save(state, {"event": "retry" if retry else "blocked",
                               "task": task_id, "reason": str(reason)[:300]})

    def cancel_prefix(self, prefix):
        """Pending jobs are cancelled; active workers must cooperatively stop."""
        with self.locked():
            state = self._load()
            changed = []
            for key, task in state["tasks"].items():
                if not key.startswith(prefix) or task["status"] in ("done", "cancelled"):
                    continue
                if task["status"] == "running":
                    task["status"] = "cancelling"
                else:
                    task["status"] = "cancelled"
                changed.append(key)
            if changed:
                self._save(state, {"event": "cancel_requested", "tasks": changed})
            return changed

    def finish_cancel(self, task_id, owner):
        with self.locked():
            state = self._load()
            task = state["tasks"].get(task_id)
            if not task or task["status"] != "cancelling" or task["owner"] != owner:
                return False
            task.update(status="cancelled", owner=None, lease_until=0)
            self._save(state, {"event": "cancelled", "task": task_id})
            return True
