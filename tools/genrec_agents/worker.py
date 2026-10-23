"""The supervised production worker: the DAG's only production executor.

The campaign's audit found that the DAG had no production executor at all: its tasks were
dispatched by an interactive coordinator, which meant a "dispatched" event never implied that
anything ran, and a task could be marked DONE by hand. This module is the missing half.

    DAG READY task
      -> AgentWorker.run_once()
         -> AgentWorker._execute(task)
            -> opencode.run_agent()   (a real subprocess; see executor.py)
            -> the verifier over the agent's workspace
         -> DAG.complete(...) with the session's artifacts, or DAG.fail(...)

What the worker guarantees, because each of them was a way for the record to lie before:

* **Nothing is marked DONE without a process.** A task's completion carries the execution
  record; a task with no record cannot be completed by the worker.
* **Concurrency is bounded per role.** L, A and M are independent and may overlap; I reads what
  M produced, so only one I runs at a time.
* **A failure backs off, it does not spin.** The delay grows with the attempt count and is
  capped; the DAG's own attempt budget is the outer bound.
* **Liveness is observable.** The worker writes its pid and a heartbeat; a stale heartbeat means
  no worker is running, and the status says so instead of claiming continuous work.
* **The gates stay fail-closed.** The worker completes a task through the DAG, so a task whose
  gate measurement is missing or failing is refused by the DAG itself.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dag import DAG, Task
from .executor import AgentSpec, OpenCodeAgentExecutor
from .roles import BINDINGS, build_spec, role_limits_for

__all__ = ["AgentWorker", "ROLE_BY_STAGE", "role_for_task"]

#: How a task's stage names its role. The role is what decides the prompt and the verifier.
ROLE_BY_STAGE = {
    "literature": "L",
    "analysis": "A",
    "retrospective": "A",
    "proposal": "M",
    "implementation": "I",
    "contract": "I",
}


def role_for_task(task: Task) -> Optional[str]:
    """The role a task belongs to, from its stage or from its id.

    A task that names no role is not an agent task: the worker leaves it to whoever owns it,
    rather than guessing.
    """
    stage = (task.stage or "").strip().lower()
    if stage in ROLE_BY_STAGE:
        return ROLE_BY_STAGE[stage]
    task_id = (task.task_id or "").lower()
    for prefix, role in ((":literature", "L"), (":analysis", "A"), (":retrospective", "A"),
                         (":proposal", "M"), (":implementation", "I"), (":contract", "I")):
        if task_id.endswith(prefix) or prefix in task_id:
            return role
    for role in BINDINGS:
        if task_id.startswith(role + ":") or f"/{role}/" in task_id:
            return role
    return None


class AgentWorker:
    """Runs the DAG's agent tasks as real OpenCode sessions, with a bounded pool."""

    def __init__(self, root: Path | str, *, executor: OpenCodeAgentExecutor,
                 dag: Optional[DAG] = None, concurrency: int = 4, owner: str = "worker",
                 role_limits: Optional[Dict[str, int]] = None, backoff_base_s: float = 5.0,
                 backoff_max_s: float = 300.0, spec_factory: Optional[Any] = None) -> None:
        self.root = Path(root)
        self.executor = executor
        self.dag = dag or DAG(self.root / "dag")
        self.concurrency = max(1, int(concurrency))
        self.owner = owner
        self.role_limits = role_limits or role_limits_for(self.concurrency)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_max_s = float(backoff_max_s)
        self.status_path = self.root / "worker_status.json"
        self._lock = threading.Lock()
        self._in_flight: Dict[str, int] = {}
        self._not_before: Dict[str, float] = {}
        #: How a task becomes a role's spec. The default is the L/A/M/I binding; a caller may
        #: bind its own prompts (and the tests bind a mock).
        self.spec_factory = spec_factory or build_spec
        self._stop = threading.Event()
        self._counters: Dict[str, int] = {"dispatched": 0, "completed": 0, "failed": 0,
                                          "skipped": 0, "idle": 0}
        self._started_utc = ""
        self._heartbeat = 0.0

    # -- liveness ---------------------------------------------------------
    def _write_status(self) -> None:
        payload = {
            "pid": os.getpid(),
            "owner": self.owner,
            "started_utc": self._started_utc,
            "heartbeat_epoch": time.time(),
            "heartbeat_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "concurrency": self.concurrency,
            "role_limits": dict(self.role_limits),
            "counters": dict(self._counters),
            "in_flight": dict(self._in_flight),
            "alive": True,
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.status_path)

    def status(self, *, stale_after_s: float = 120.0) -> Dict[str, Any]:
        """The worker's own state, and whether a worker is actually alive right now."""
        if not self.status_path.is_file():
            return {"alive": False, "reason": "no worker has ever run here",
                    "counters": dict(self._counters)}
        payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        age = time.time() - float(payload.get("heartbeat_epoch") or 0)
        payload["heartbeat_age_s"] = round(age, 2)
        payload["alive"] = age <= stale_after_s
        if not payload["alive"]:
            payload["reason"] = (f"the last heartbeat is {age:.1f}s old, so no worker is running; "
                                 "the DAG is idle, not working")
        return payload

    def stop(self) -> None:
        self._stop.set()

    # -- dispatch ---------------------------------------------------------
    def _role_free(self, role: str) -> bool:
        with self._lock:
            return self._in_flight.get(role, 0) < int(self.role_limits.get(role, 1))

    def _enter(self, role: str) -> None:
        with self._lock:
            self._in_flight[role] = self._in_flight.get(role, 0) + 1

    def _leave(self, role: str) -> None:
        with self._lock:
            self._in_flight[role] = max(0, self._in_flight.get(role, 0) - 1)

    def run_once(self) -> Dict[str, Any]:
        """One pass: take at most one ready task and run it as an agent session."""
        with self._lock:
            self._heartbeat = time.time()
        result = self.dag.tick(owner=self.owner, limit=1)
        dispatched = result.get("dispatched") or []
        if not dispatched:
            with self._lock:
                self._counters["idle"] += 1
            self._write_status()
            return {"dispatched": [], "idle": True, "deferred": result.get("deferred", {})}
        task_id = dispatched[0]
        task = self.dag.task(task_id)
        role = role_for_task(task)
        if role is None:
            # not an agent task: the worker does not own it, so it releases the lease honestly
            self.dag.fail(task_id, self.owner,
                          "the worker owns agent tasks only: this task names no role")
            with self._lock:
                self._counters["skipped"] += 1
            self._write_status()
            return {"dispatched": [task_id], "idle": False, "skipped": True}
        if not self._role_free(role):
            # the role is at its concurrency limit: wait for a slot rather than exceed the bound
            self.dag.release(task_id, self.owner, "the role is at its concurrency limit")
            with self._lock:
                self._counters["skipped"] += 1
            self._write_status()
            return {"dispatched": [task_id], "idle": False, "deferred_for_limit": role}
        wait = self._not_before.get(task_id, 0.0) - time.time()
        if wait > 0:
            self.dag.release(task_id, self.owner, f"backing off for {wait:.1f}s")
            with self._lock:
                self._counters["skipped"] += 1
            self._write_status()
            return {"dispatched": [task_id], "idle": False, "backoff_s": round(wait, 2)}
        outcome = self._execute(task, role)
        with self._lock:
            self._counters["dispatched"] += 1
            self._counters["completed" if outcome["ok"] else "failed"] += 1
        self._write_status()
        return {"dispatched": [task_id], "idle": False, **outcome}

    def _execute(self, task: Task, role: str) -> Dict[str, Any]:
        """Run the agent, then let the DAG record the outcome."""
        self._enter(role)
        try:
            spec = self._spec_for(task, role)
            record = self.executor.execute(task.task_id, spec)
            artifacts = [record.session_path, record.tool_jsonl_path] if record.session_path else []
            if record.ok:
                self.dag.complete(task.task_id, self.owner, artifacts=artifacts,
                                  detail=record.error or f"the {role} agent finished its work",
                                  metrics={"role": role, "pid": record.pid,
                                           "elapsed_s": record.elapsed_s,
                                           "exit_code": record.exit_code,
                                           "tool_calls": record.n_tool_calls,
                                           "tokens": record.tokens, "cost": record.cost})
                return {"ok": True, "role": role, "record": record.as_dict()}
            self._not_before[task.task_id] = time.time() + self._backoff_for(task)
            self.dag.fail(task.task_id, self.owner, record.error or "the agent produced no usable output")
            return {"ok": False, "role": role, "record": record.as_dict()}
        finally:
            self._leave(role)

    def _backoff_for(self, task: Task) -> float:
        delay = self.backoff_base_s * (2 ** max(0, int(task.attempts) - 1))
        return min(delay, self.backoff_max_s)

    def _spec_for(self, task: Task, role: str) -> AgentSpec:
        """The role's spec, with the task's frozen inputs when it declares any."""
        inputs: Dict[str, Any] = {}
        files: Tuple[Tuple[str, Path], ...] = ()
        detail = (task.detail or "").strip()
        if detail.startswith("{"):
            try:
                payload = json.loads(detail)
            except json.JSONDecodeError:
                payload = {}
            inputs = payload.get("inputs") or {}
            files = tuple((str(name), Path(path)) for name, path in (payload.get("files") or []))
        return self.spec_factory(role, task.task_id, model=self.executor.model,
                                 inputs=inputs, files=files)

    # -- supervision ------------------------------------------------------
    def serve(self, *, max_ticks: Optional[int] = None, idle_sleep: float = 2.0) -> Dict[str, Any]:
        """Run the worker pool until it is stopped or the tick budget is spent."""
        self._started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._write_status()
        ticks = 0
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = []
            while max_ticks is None or ticks < max_ticks:
                if self._stop.is_set():
                    break
                ticks += 1
                futures = [f for f in futures if not f.done()]
                if len(futures) >= self.concurrency:
                    time.sleep(idle_sleep)
                    continue
                futures.append(pool.submit(self.run_once))
                time.sleep(0.05)
            for future in futures:
                future.result()
        self._write_status()
        return {"ticks": ticks, "counters": dict(self._counters)}
