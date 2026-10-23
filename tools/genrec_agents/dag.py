"""A durable dependency scheduler for the research campaign (issue #112, P0).

The campaign is a graph of tasks per candidate: literature (L), retrospective analysis (A) and
mechanism sketches (M) run concurrently; a JOIN R reviews them before the proposal hash is
frozen; then implementation (I), independent oracle fixtures and the evaluation plan run
concurrently; JOIN E requires complete H1-H7 AND CGV G1-G6 on identical frozen inputs before the
single authorised GPU critical path is even READY.

What this module guarantees, and why each guarantee exists:

* **Durable state.** Every mutation writes the whole graph atomically (temporary file, then
  ``os.replace``) and appends an event to ``events.jsonl``. A process that dies mid-tick leaves a
  readable state, not a half-written one.
* **Atomic leases.** A claim takes an exclusive ``flock``, re-reads the state, and only then
  writes the lease. Two workers cannot both hold the same task, and a worker crash is detected by
  the lease expiring -- not by a heartbeat thread that may also have died.
* **Idempotent reservations.** An official experiment is reserved exactly once per
  ``(candidate, stage, seed)``. The reservation is the only thing that counts as a research
  round; a tick that dispatches nothing increments the idle counter and nothing else.
* **Selective invalidation.** Invalidating a task re-queues it and its transitive dependents, and
  leaves the rest of the graph alone. A changed proposal does not force the whole portfolio to
  re-run, and a changed artefact does not silently keep a stale dependent "done".
* **Bounded retries.** A task that fails more than ``max_attempts`` times becomes BLOCKED with
  its error recorded; it is never retried forever, and it never counts as progress.
* **Disjoint write scopes.** Two running tasks may not declare overlapping write scopes. The
  scheduler refuses the second one, which is the mechanical half of "no concurrent file
  corruption".
* **Authorisation is a state, not a comment.** A GPU task sits in WAITING_AUTHORISATION until an
  operator token is recorded. The scheduler will not dispatch it, and an idle tick says so
  instead of pretending to work.

The scheduler is NOT a scientific run. A task's ``kind`` distinguishes ``engineering`` work
(dry runs, fixtures, scheduler tests) from ``experiment`` work (the official Stage2/Stage3). Only
``experiment`` reservations are rounds, so a dry-run gallery can never be reported as science.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA_VERSION = 1

#: The lifecycle. ``waiting_*`` and ``blocked_*`` are NOT dispatchable; only ``ready`` is.
READY = "ready"
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
BLOCKED = "blocked"
WAITING_EXTERNAL = "waiting_external"
WAITING_AUTHORISATION = "waiting_authorisation"
BLOCKED_NO_NOVEL_EVIDENCE = "blocked_no_novel_evidence"
USER_STOPPED = "user_stopped"

#: States from which no dispatch may happen until something explicit changes them.
NOT_DISPATCHABLE = (WAITING_EXTERNAL, WAITING_AUTHORISATION, BLOCKED_NO_NOVEL_EVIDENCE,
                    USER_STOPPED, BLOCKED, DONE, RUNNING, PENDING)

RESOURCE_CLASSES = ("none", "cpu", "gpu")
KINDS = ("engineering", "experiment")

#: The four research roles and the two joins of the #112 DAG, as dependency names.
ROLE_STAGES = ("L", "A", "M")
JOIN_R = "join_R"
WORKSTREAM_STAGES = ("implement", "oracle", "eval_plan")
JOIN_E = "join_E"
OFFICIAL_STAGES = ("stage2", "er_d2", "g_h1")


def _now() -> float:
    return time.time()


@dataclass
class Task:
    """One node of the campaign graph."""

    task_id: str
    candidate: str
    stage: str
    deps: tuple[str, ...] = ()
    resource: str = "cpu"
    kind: str = "engineering"
    write_scope: tuple[str, ...] = ()
    verify: tuple[str, ...] = ()
    max_attempts: int = 2
    requires: dict | None = None
    description: str = ""
    status: str = PENDING
    lease_owner: str = ""
    lease_expires: float = 0.0
    attempts: int = 0
    error: str = ""
    artifacts: list[dict[str, str]] = field(default_factory=list)
    gate_measurement: dict | None = None
    started_at: float | None = None
    finished_at: float | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "candidate": self.candidate, "stage": self.stage,
            "deps": list(self.deps), "resource": self.resource, "kind": self.kind,
            "write_scope": list(self.write_scope), "verify": list(self.verify),
            "max_attempts": self.max_attempts, "requires": self.requires,
            "description": self.description,
            "status": self.status, "lease_owner": self.lease_owner,
            "lease_expires": self.lease_expires, "attempts": self.attempts,
            "error": self.error, "artifacts": list(self.artifacts),
            "gate_measurement": self.gate_measurement,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        known["deps"] = tuple(known.get("deps") or ())
        known["write_scope"] = tuple(known.get("write_scope") or ())
        known["verify"] = tuple(known.get("verify") or ())
        known["artifacts"] = list(known.get("artifacts") or [])
        known["gate_measurement"] = known.get("gate_measurement")
        return cls(**known)


def _satisfies(measured: float, op: str, threshold: float) -> bool:
    if op == ">":
        return measured > threshold
    if op == ">=":
        return measured >= threshold
    if op == "<":
        return measured < threshold
    if op == "<=":
        return measured <= threshold
    if op == "==":
        return measured == threshold
    raise SchedulerError(f"unknown comparison {op!r}")


def validate_task(task: Task) -> list[str]:
    problems: list[str] = []
    if not task.task_id or not task.task_id.strip():
        problems.append("task_id: missing")
    if not task.candidate or not task.candidate.strip():
        problems.append(f"{task.task_id}: candidate missing")
    if not task.stage or not task.stage.strip():
        problems.append(f"{task.task_id}: stage missing")
    if task.resource not in RESOURCE_CLASSES:
        problems.append(f"{task.task_id}: resource {task.resource!r} not in {list(RESOURCE_CLASSES)}")
    if task.kind not in KINDS:
        problems.append(f"{task.task_id}: kind {task.kind!r} not in {list(KINDS)}")
    if task.max_attempts < 1:
        problems.append(f"{task.task_id}: max_attempts must be at least 1")
    if task.task_id in task.deps:
        problems.append(f"{task.task_id}: depends on itself")
    if task.requires is not None:
        requirement = task.requires
        if not isinstance(requirement, dict) or "metric" not in requirement:
            problems.append(f"{task.task_id}: requires needs a metric name")
        elif requirement.get("op", ">") not in (">", ">=", "<", "<=", "=="):
            problems.append(f"{task.task_id}: requires.op {requirement.get('op')!r} is unknown")
        elif "threshold" not in requirement:
            problems.append(f"{task.task_id}: requires needs a threshold")
    return problems


class SchedulerError(RuntimeError):
    """A request the scheduler refuses, always with the reason."""


class DAG:
    """The campaign's durable graph. One directory, one lock, one state file."""

    STATE = "state.json"
    EVENTS = "events.jsonl"
    RESERVATIONS = "reservations.json"
    LOCK = ".lock"
    ARTIFACTS = "artifacts"

    def __init__(self, root: Path | str, *, resource_limits: dict[str, int] | None = None,
                 clock: Callable[[], float] | None = None,
                 lease_ttl: float = 300.0) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self.ARTIFACTS).mkdir(exist_ok=True)
        self.lock_path = self.root / self.LOCK
        self.clock = clock or _now
        self.lease_ttl = float(lease_ttl)
        self.resource_limits = dict(resource_limits or {"cpu": 4, "gpu": 1, "none": 64})
        if not (self.root / self.STATE).is_file():
            with self._Lock(self.lock_path):
                if not (self.root / self.STATE).is_file():
                    self._write_state(self._empty_state())

    # ------------------------------------------------------------------ durable plumbing
    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "tasks": {},
            "rounds": 0,
            "idle_ticks": 0,
            "dispatched": 0,
            "authorisations": {},
            "created_utc": self.clock(),
        }

    def _write_state(self, state: dict[str, Any]) -> None:
        temporary = self.root / (self.STATE + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, self.root / self.STATE)

    def _read_state(self) -> dict[str, Any]:
        path = self.root / self.STATE
        if not path.is_file():
            return self._empty_state()
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchedulerError(f"the scheduler state at {path} is unreadable: {exc}") from exc

    def _event(self, kind: str, **fields: Any) -> None:
        record = {"at": self.clock(), "event": kind, **fields}
        with (self.root / self.EVENTS).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    class _Lock:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.handle = None

        def __enter__(self):
            self.handle = self.path.open("a+")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
            return self

        def __exit__(self, *exc_info):
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            return False

    # ------------------------------------------------------------------ graph construction
    def add(self, task: Task | Iterable[Task]) -> list[str]:
        if isinstance(task, Task):
            tasks = [task]
        else:
            tasks = list(task)
        added: list[str] = []
        with self._Lock(self.lock_path):
            state = self._read_state()
            known = state["tasks"]
            for item in tasks:
                problems = validate_task(item)
                if problems:
                    raise SchedulerError("; ".join(problems))
                if item.task_id in known:
                    raise SchedulerError(f"task {item.task_id!r} already exists; tasks are immutable")
                missing = [dep for dep in item.deps if dep not in known]
                if missing:
                    raise SchedulerError(
                        f"task {item.task_id!r} names unknown dependencies {missing}")
                known[item.task_id] = item.as_dict()
                added.append(item.task_id)
            for item in tasks:
                self._recompute(known[item.task_id], known)
            self._write_state(state)
        for task_id in added:
            self._event("added", task=task_id)
        return added

    def _recompute(self, task: dict[str, Any], known: dict[str, Any]) -> None:
        """Promote PENDING to READY when every dependency is DONE."""
        if task["status"] != PENDING:
            return
        if all(known[dep]["status"] == DONE for dep in task["deps"]):
            task["status"] = READY
            self._event("ready", task=task["task_id"])

    # ------------------------------------------------------------------ observation
    def state(self) -> dict[str, Any]:
        return self._read_state()

    def task(self, task_id: str) -> Task:
        state = self._read_state()
        if task_id not in state["tasks"]:
            raise SchedulerError(f"no task {task_id!r}")
        return Task.from_dict(state["tasks"][task_id])

    def tasks(self) -> list[Task]:
        state = self._read_state()
        return [Task.from_dict(value) for value in state["tasks"].values()]

    def events(self) -> list[dict[str, Any]]:
        path = self.root / self.EVENTS
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def rounds(self) -> int:
        return int(self._read_state().get("rounds", 0))

    def idle_ticks(self) -> int:
        return int(self._read_state().get("idle_ticks", 0))

    def ready(self) -> list[Task]:
        return [task for task in self.tasks() if task.status == READY]

    def running(self) -> list[Task]:
        return [task for task in self.tasks() if task.status == RUNNING]

    # ------------------------------------------------------------------ authorisation
    def authorise(self, resource: str, token: str, note: str = "") -> None:
        """Record an operator authorisation for a resource class (``gpu`` today)."""
        with self._Lock(self.lock_path):
            state = self._read_state()
            state.setdefault("authorisations", {})[resource] = {
                "token": token, "note": note, "at": self.clock()}
            self._write_state(state)
        self._event("authorised", resource=resource, token=token)

    def authorised(self, resource: str) -> bool:
        if resource != "gpu":
            return True
        return bool((self._read_state().get("authorisations") or {}).get("gpu"))

    # ------------------------------------------------------------------ dispatch
    def _scopes_overlap(self, left: Sequence[str], right: Sequence[str]) -> bool:
        for a in left:
            for b in right:
                if a == b or a.startswith(b.rstrip("/") + "/") or b.startswith(a.rstrip("/") + "/"):
                    return True
        return False

    def _running_by_class(self, tasks: dict[str, Any]) -> dict[str, int]:
        counts = {name: 0 for name in RESOURCE_CLASSES}
        for value in tasks.values():
            if value["status"] == RUNNING:
                counts[value["resource"]] = counts.get(value["resource"], 0) + 1
        return counts

    def tick(self, owner: str = "coordinator", *, resource: str | None = None,
             limit: int | None = None) -> dict[str, Any]:
        """Reconcile, then dispatch READY tasks within the resource limits.

        A tick that dispatches nothing is an IDLE tick: it increments ``idle_ticks`` and nothing
        else. ``rounds`` only moves when an official experiment is actually reserved, which is
        what makes 72 idle ticks produce zero false rounds.
        """
        now = self.clock()
        reclaimed: list[str] = []
        dispatched: list[str] = []
        deferred: dict[str, str] = {}
        with self._Lock(self.lock_path):
            state = self._read_state()
            tasks = state["tasks"]
            # 1. reconcile expired leases: a dead worker must not hold a task forever
            for value in tasks.values():
                if value["status"] == RUNNING and float(value["lease_expires"]) <= now:
                    value["lease_owner"] = ""
                    value["lease_expires"] = 0.0
                    reclaimed.append(value["task_id"])
                    if value["attempts"] >= value["max_attempts"]:
                        value["status"] = BLOCKED
                        value["error"] = value["error"] or "the lease expired and the attempts are spent"
                        self._event("blocked", task=value["task_id"],
                                    reason="lease expired, attempts spent")
                    else:
                        value["status"] = READY
                        self._event("lease_expired", task=value["task_id"])
            # 2. promote pending tasks whose dependencies are now done, and release the tasks
            #    that were only waiting for an authorisation that has since been recorded
            for value in tasks.values():
                if value["status"] == WAITING_AUTHORISATION and self.authorised(value["resource"]):
                    value["status"] = READY
                    value["detail"] = ""
                    self._event("authorisation_released", task=value["task_id"])
                self._recompute(value, tasks)
            # 2b. a READY task whose dependencies are not all DONE is not dispatchable.
            #     A dependency that was later closed (blocked, failed, or refuted) closes
            #     the task with it; a dependency that is merely unfinished demotes it back
            #     to PENDING so it can be promoted again when the dependency completes.
            #     Without this, a task promoted before its dependency was closed stays
            #     READY and a tick dispatches a candidate whose family was refuted.
            for value in tasks.values():
                if value["status"] != READY:
                    continue
                unmet = [dep for dep in value["deps"] if tasks[dep]["status"] != DONE]
                if not unmet:
                    continue
                closed = [dep for dep in unmet
                          if tasks[dep]["status"] in (BLOCKED, FAILED, BLOCKED_NO_NOVEL_EVIDENCE,
                                                     USER_STOPPED)]
                value["status"] = BLOCKED if closed else PENDING
                value["detail"] = (
                    "a dependency is closed: " + ", ".join(unmet) if closed
                    else "waiting for: " + ", ".join(unmet))
                self._event("demoted", task=value["task_id"], reason=value["detail"])
            # 3. dispatch
            counts = self._running_by_class(tasks)
            running_scopes = [tuple(value["write_scope"]) for value in tasks.values()
                              if value["status"] == RUNNING]
            budget = limit
            for value in sorted(tasks.values(), key=lambda item: item["task_id"]):
                if budget is not None and budget <= 0:
                    break
                if value["status"] != READY:
                    continue
                if resource is not None and value["resource"] != resource:
                    continue
                if value["resource"] == "gpu" and not self.authorised("gpu"):
                    value["status"] = WAITING_AUTHORISATION
                    deferred[value["task_id"]] = "no operator authorisation for gpu"
                    self._event("deferred", task=value["task_id"], reason="waiting_authorisation")
                    continue
                capacity = self.resource_limits.get(value["resource"], 0)
                if counts.get(value["resource"], 0) >= capacity:
                    deferred[value["task_id"]] = f"resource {value['resource']!r} at capacity"
                    continue
                if any(self._scopes_overlap(tuple(value["write_scope"]), scope)
                       for scope in running_scopes):
                    deferred[value["task_id"]] = "write scope overlaps a running task"
                    self._event("deferred", task=value["task_id"], reason="write scope overlap")
                    continue
                value["status"] = RUNNING
                value["lease_owner"] = owner
                value["lease_expires"] = now + self.lease_ttl
                value["attempts"] = int(value["attempts"]) + 1
                value["started_at"] = now
                counts[value["resource"]] = counts.get(value["resource"], 0) + 1
                running_scopes.append(tuple(value["write_scope"]))
                dispatched.append(value["task_id"])
                if budget is not None:
                    budget -= 1
                self._event("dispatched", task=value["task_id"], owner=owner,
                            attempt=value["attempts"], resource=value["resource"])
            if dispatched:
                state["dispatched"] = int(state.get("dispatched", 0)) + len(dispatched)
            else:
                state["idle_ticks"] = int(state.get("idle_ticks", 0)) + 1
                self._event("idle_tick", idle=state["idle_ticks"], deferred=deferred)
            self._write_state(state)
        return {"dispatched": dispatched, "reclaimed": reclaimed, "deferred": deferred,
                "idle": not dispatched, "rounds": state.get("rounds", 0),
                "idle_ticks": state.get("idle_ticks", 0)}

    # ------------------------------------------------------------------ lease lifecycle
    def heartbeat(self, task_id: str, owner: str) -> bool:
        now = self.clock()
        with self._Lock(self.lock_path):
            state = self._read_state()
            value = state["tasks"].get(task_id)
            if not value or value["status"] != RUNNING or value["lease_owner"] != owner:
                return False
            value["lease_expires"] = now + self.lease_ttl
            self._write_state(state)
        self._event("heartbeat", task=task_id, owner=owner)
        return True

    def complete(self, task_id: str, owner: str, *, artifacts: Iterable[str] = (),
                 detail: str = "", metrics: dict[str, Any] | None = None) -> None:
        with self._Lock(self.lock_path):
            state = self._read_state()
            value = state["tasks"].get(task_id)
            if not value:
                raise SchedulerError(f"no task {task_id!r}")
            if value["status"] != RUNNING or (owner and value["lease_owner"] != owner):
                raise SchedulerError(
                    f"task {task_id!r} is {value['status']}, owner {value['lease_owner']!r}; "
                    f"only the lease holder can complete it")
            # A GATE. The dependency graph itself refuses to release what follows a gate whose
            # measurement does not clear the threshold -- the ER gate before Stage-3 is the case
            # that matters, because a coordinator that ignored the evaluation plan would
            # otherwise be free to start Stage-3 on a failing ER.
            requirement = value.get("requires")
            if requirement:
                metric = requirement["metric"]
                measured = (metrics or {}).get(metric)
                if measured is None:
                    raise SchedulerError(
                        f"task {task_id!r} is gated on {metric!r}, and no measured value was given")
                if not _satisfies(float(measured), requirement.get("op", ">"),
                                  float(requirement["threshold"])):
                    raise SchedulerError(
                        f"task {task_id!r} is gated on {metric!r} {requirement.get('op', '>')} "
                        f"{requirement['threshold']}, and the measurement is {measured}")
                value["gate_measurement"] = {"metric": metric, "value": float(measured),
                                             "op": requirement.get("op", ">"),
                                             "threshold": float(requirement["threshold"])}
            value["status"] = DONE
            value["lease_owner"] = ""
            value["lease_expires"] = 0.0
            value["finished_at"] = self.clock()
            value["detail"] = detail
            for item in artifacts:
                value["artifacts"].append({"path": str(item),
                                           "sha256": self._hash_path(Path(item))})
            for other in state["tasks"].values():
                self._recompute(other, state["tasks"])
            self._write_state(state)
        self._event("done", task=task_id, detail=detail)

    def fail(self, task_id: str, owner: str, error: str) -> None:
        with self._Lock(self.lock_path):
            state = self._read_state()
            value = state["tasks"].get(task_id)
            if not value:
                raise SchedulerError(f"no task {task_id!r}")
            if value["status"] != RUNNING or (owner and value["lease_owner"] != owner):
                raise SchedulerError(
                    f"task {task_id!r} is {value['status']}; only the lease holder can fail it")
            value["error"] = str(error)[:2000]
            value["lease_owner"] = ""
            value["lease_expires"] = 0.0
            if int(value["attempts"]) >= int(value["max_attempts"]):
                value["status"] = BLOCKED
                value["finished_at"] = self.clock()
            else:
                value["status"] = READY
            self._write_state(state)
        self._event("failed", task=task_id, error=str(error)[:400],
                    blocked=value["status"] == BLOCKED, attempts=value["attempts"])

    def set_status(self, task_id: str, status: str, reason: str = "") -> None:
        if status not in (READY, PENDING, WAITING_EXTERNAL, WAITING_AUTHORISATION,
                          BLOCKED_NO_NOVEL_EVIDENCE, USER_STOPPED, BLOCKED):
            raise SchedulerError(f"{status!r} is not a status a coordinator may set directly")
        with self._Lock(self.lock_path):
            state = self._read_state()
            value = state["tasks"].get(task_id)
            if not value:
                raise SchedulerError(f"no task {task_id!r}")
            if value["status"] == RUNNING:
                raise SchedulerError(
                    f"task {task_id!r} is RUNNING; complete or fail it, or let its lease expire")
            value["status"] = status
            value["detail"] = reason
            self._write_state(state)
        self._event("status", task=task_id, status=status, reason=reason)

    # ------------------------------------------------------------------ the worker loop
    def serve(self, run: Callable[[Task], None], *, owner: str = "worker",
              resource: str | None = None, idle_sleep: float = 0.0,
              max_ticks: int | None = None) -> dict[str, int]:
        """A supervised worker: tick, run ONE dispatched task, heartbeat while it runs.

        The heartbeat is what makes a lease meaningful: a task that runs longer than the TTL
        keeps its lease, so a concurrent tick does not treat a live worker as dead. A task that
        dies takes its process with it, its heartbeats stop, and the lease expires into a
        reclaim -- which is the crash recovery path, not an exception handler.
        """
        served = 0
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            ticks += 1
            result = self.tick(owner=owner, resource=resource, limit=1)
            if not result["dispatched"]:
                if idle_sleep:
                    time.sleep(idle_sleep)
                continue
            task_id = result["dispatched"][0]
            task = self.task(task_id)
            stop = threading.Event()

            def beat() -> None:
                while not stop.wait(max(self.lease_ttl / 3.0, 0.01)):
                    self.heartbeat(task_id, owner)

            thread = threading.Thread(target=beat, daemon=True)
            thread.start()
            try:
                run(task)
            except Exception as exc:  # noqa: BLE001 - a failing task is state, not a crash
                self.fail(task_id, owner, f"{type(exc).__name__}: {exc}")
            else:
                self.complete(task_id, owner)
            finally:
                stop.set()
                thread.join(timeout=1.0)
            served += 1
        return {"served": served, "ticks": ticks}

    # ------------------------------------------------------------------ experiments
    def reserve(self, candidate: str, stage: str, seed: int = 42, *,
                protocol: str = "") -> bool:
        """Reserve an official experiment exactly once. True when this caller got it.

        The reservation is the only thing that counts as a research round, and it is written
        under the same lock as the task leases, so two coordinators racing for the same
        ``(candidate, stage, seed)`` produce one reservation and one round -- never two runs.
        """
        key = f"{candidate}|{stage}|{int(seed)}|{protocol}"
        with self._Lock(self.lock_path):
            state = self._read_state()
            path = self.root / self.RESERVATIONS
            reservations = {}
            if path.is_file():
                reservations = json.loads(path.read_text(encoding="utf-8"))
            if key in reservations:
                return False
            reservations[key] = {"candidate": candidate, "stage": stage, "seed": int(seed),
                                 "protocol": protocol, "at": self.clock()}
            temporary = self.root / (self.RESERVATIONS + ".tmp")
            temporary.write_text(json.dumps(reservations, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, path)
            state["rounds"] = int(state.get("rounds", 0)) + 1
            self._write_state(state)
        self._event("reserved", key=key, rounds=state.get("rounds", 0))
        return True

    def reservations(self) -> dict[str, Any]:
        path = self.root / self.RESERVATIONS
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ invalidation
    def invalidate(self, task_id: str, reason: str = "") -> list[str]:
        """Re-queue a task and its transitive dependents; leave the rest of the graph alone."""
        with self._Lock(self.lock_path):
            state = self._read_state()
            tasks = state["tasks"]
            if task_id not in tasks:
                raise SchedulerError(f"no task {task_id!r}")
            affected = [task_id]
            frontier = [task_id]
            while frontier:
                current = frontier.pop()
                for name, value in tasks.items():
                    if current in value["deps"] and name not in affected:
                        affected.append(name)
                        frontier.append(name)
            for name in affected:
                value = tasks[name]
                value["status"] = PENDING
                value["lease_owner"] = ""
                value["lease_expires"] = 0.0
                value["finished_at"] = None
                value["artifacts"] = []
                value["detail"] = f"invalidated by {task_id}: {reason}" if reason else \
                    f"invalidated by {task_id}"
            for name in affected:
                self._recompute(tasks[name], tasks)
            self._write_state(state)
        self._event("invalidated", task=task_id, affected=affected, reason=reason)
        return affected

    # ------------------------------------------------------------------ artifacts
    def _hash_path(self, path: Path) -> str:
        if not path.is_file():
            return ""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def store_artifact(self, data: bytes | str | Path, *, suffix: str = "") -> dict[str, str]:
        """Content-addressed storage: the same bytes are stored once, under their digest."""
        if isinstance(data, Path):
            payload = data.read_bytes()
            suffix = suffix or data.suffix
        elif isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = bytes(data)
        digest = hashlib.sha256(payload).hexdigest()
        folder = self.root / self.ARTIFACTS / digest[:2]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (digest + suffix)
        if not path.is_file():
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
        return {"sha256": digest, "path": str(path)}

    # ------------------------------------------------------------------ reporting
    def report(self) -> dict[str, Any]:
        state = self._read_state()
        counts: dict[str, int] = {}
        for value in state["tasks"].values():
            counts[value["status"]] = counts.get(value["status"], 0) + 1
        experiments = [value for value in state["tasks"].values()
                       if value["kind"] == "experiment"]
        return {
            "rounds": state.get("rounds", 0),
            "idle_ticks": state.get("idle_ticks", 0),
            "dispatched_total": state.get("dispatched", 0),
            "status_counts": dict(sorted(counts.items())),
            "experiment_tasks": len(experiments),
            "experiment_done": sum(1 for value in experiments if value["status"] == DONE),
            "authorisations": state.get("authorisations", {}),
            "tasks": [Task.from_dict(value) for value in state["tasks"].values()],
        }


__all__ = [
    "BLOCKED",
    "BLOCKED_NO_NOVEL_EVIDENCE",
    "DAG",
    "DONE",
    "FAILED",
    "JOIN_E",
    "JOIN_R",
    "KINDS",
    "OFFICIAL_STAGES",
    "PENDING",
    "READY",
    "RESOURCE_CLASSES",
    "ROLE_STAGES",
    "RUNNING",
    "SchedulerError",
    "Task",
    "USER_STOPPED",
    "WAITING_AUTHORISATION",
    "WAITING_EXTERNAL",
    "WORKSTREAM_STAGES",
    "validate_task",
]
