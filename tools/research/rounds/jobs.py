"""The Stage 2 / Stage 3 job queue of Issue #120.

Every verified implementation becomes a Stage 2 job. The queue is
resource-aware (it never starts more concurrent trainings than the configured
budget), deterministic (jobs start in a fixed order), and isolated (each job owns
its tree, its result directory and its artifacts). A Stage 2 job that finishes
with a technically valid SID hands over to Stage 3 automatically - no human
approval, no quality floor, no single-winner restriction. A job that fails
records the failure with its evidence and lets the other jobs proceed.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

from ..core.evidence import atomic_json, digest_file, read_json

JOB_FILE = "jobs.json"
GATE_PATH = Path("tools/genrec-mcp/stage2_quality_gate.py")


class JobError(RuntimeError):
    """The queue cannot do what was asked."""


def _gate_module():
    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("genrec_stage2_validity", repo / GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sid_is_technically_valid(sid_path: Path, *, repo: Path = None) -> dict:
    """The admission decision: can this SID be used, whatever its quality is?"""
    repo = Path(repo or Path(__file__).resolve().parents[3])
    module = _gate_module()
    result = module.evaluate(
        Path(sid_path),
        repo / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json",
        repo / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz",
        repo / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.meta.json")
    return {"valid": bool(result["passed"]), "refused": result.get("refused"),
            "diagnostics": result.get("candidate", {}),
            "reference_status": result.get("reference_status", {})}


def plan_gpu_allocation(running_jobs: list, usable: list, *, jobs_per_gpu: int = 2) -> dict:
    """Plan where the next single-card jobs run.

    A card may host several of these trainers, so the capacity is counted per
    card: the result is the list of free slots, one entry per job that fits, and
    the cards that hold no job at all, which is what a four-rank trainer needs.
    """
    running_per_card = {}
    for job in running_jobs:
        for card in str(job.get("gpu", "")).split(","):
            card = card.strip()
            if not card:
                continue
            running_per_card[card] = running_per_card.get(card, 0) + 1
    free_slots = []
    idle_cards = []
    for card in usable:
        card = str(card)
        used = running_per_card.get(card, 0)
        if used == 0:
            idle_cards.append(card)
        free_slots.extend([card] * max(0, int(jobs_per_gpu) - used))
    return {"free_slots": free_slots, "idle_cards": idle_cards,
            "running_per_card": running_per_card,
            "capacity": len(free_slots) + sum(running_per_card.values())}


class JobQueue:
    """A durable, resource-aware queue of Stage 2 and Stage 3 jobs."""

    def __init__(self, root: Path, *, max_parallel: int = 1):
        self.root = Path(root)
        self.max_parallel = max(1, int(max_parallel))
        self.path = self.root / JOB_FILE
        if self.path.is_file():
            self.jobs = read_json(self.path) or {"schema": 1, "jobs": []}
        else:
            self.jobs = {"schema": 1, "jobs": []}
        self.jobs.setdefault("jobs", [])

    # ------------------------------------------------------------- persistence
    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, self.jobs)

    def _job(self, job_id: str) -> dict:
        for job in self.jobs["jobs"]:
            if job["job_id"] == job_id:
                return job
        raise JobError("unknown job: " + job_id)

    # ----------------------------------------------------------------- enqueue
    def enqueue_stage2(self, *, idea_id: str, tree: Path, result_dir: Path,
                       round_id: str = "") -> dict:
        job = {
            "job_id": "stage2:" + idea_id,
            "kind": "stage2",
            "idea_id": idea_id,
            "round_id": round_id,
            "tree": str(tree),
            "result_dir": str(result_dir),
            "status": "pending",
            "attempts": 0,
            "queued_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
        }
        self.jobs["jobs"].append(job)
        self.save()
        return job

    def enqueue_stage3(self, *, idea_id: str, tree: Path, sid_path: Path,
                       round_id: str = "", stage2_job_id: str = "") -> dict:
        job = {
            "job_id": "stage3:" + idea_id,
            "kind": "stage3",
            "idea_id": idea_id,
            "round_id": round_id,
            "tree": str(tree),
            "sid_path": str(sid_path),
            "stage2_job_id": stage2_job_id,
            "status": "pending",
            "attempts": 0,
            "queued_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
        }
        self.jobs["jobs"].append(job)
        self.save()
        return job

    # ------------------------------------------------------------------ status
    def running(self) -> list:
        return [job for job in self.jobs["jobs"] if job["status"] == "running"]

    def pending(self, kind: str = None) -> list:
        jobs = [job for job in self.jobs["jobs"] if job["status"] == "pending"]
        if kind:
            jobs = [job for job in jobs if job["kind"] == kind]
        return sorted(jobs, key=lambda job: (job["kind"] != "stage3", job["idea_id"]))

    def next_batch(self, kind: str = "stage2") -> list:
        """The jobs that may start now, in a deterministic order."""
        free = self.max_parallel - len(self.running())
        if free <= 0:
            return []
        return self.pending(kind)[:free]

    # ------------------------------------------------------------------ results
    def complete_stage2(self, job_id: str, *, sid_path: Path, metrics: dict = None) -> dict:
        """Record a finished Stage 2 and hand it to Stage 3 when the SID is usable."""
        job = self._job(job_id)
        job["status"] = "done"
        job["finished_at"] = time.time()
        validity = sid_is_technically_valid(sid_path)
        job["result"] = {"sid_path": str(sid_path), "validity": validity,
                         "metrics": metrics or {},
                         "sid_sha256": digest_file(Path(sid_path))
                         if Path(sid_path).is_file() else None}
        handoff = None
        if validity["valid"]:
            handoff = self.enqueue_stage3(
                idea_id=job["idea_id"], tree=Path(job["tree"]), sid_path=Path(sid_path),
                round_id=job.get("round_id", ""), stage2_job_id=job_id)
            job["stage3_job_id"] = handoff["job_id"]
        self.save()
        return {"job": job, "stage3": handoff}

    def fail(self, job_id: str, error: str, *, evidence: dict = None) -> dict:
        """Record a failed job; the other jobs keep running."""
        job = self._job(job_id)
        job["status"] = "failed"
        job["finished_at"] = time.time()
        job["error"] = str(error)[:2000]
        job["failure_evidence"] = evidence or {}
        self.save()
        return job

    def resource_block(self, job_id: str, reason: str, *, evidence: dict = None) -> dict:
        """Record a measured infeasibility.

        The mechanism is not rejected: the run was too expensive on this host,
        so the idea is kept for an optimisation route.
        """
        job = self._job(job_id)
        job["status"] = "resource_blocked"
        job["finished_at"] = time.time()
        job["error"] = str(reason)[:2000]
        job["failure_evidence"] = evidence or {}
        job["failure_kind"] = "RESOURCE_BLOCKED"
        self.save()
        return job

    def complete_stage3(self, job_id: str, result: dict) -> dict:
        job = self._job(job_id)
        job["status"] = "done"
        job["finished_at"] = time.time()
        job["result"] = result
        self.save()
        return job

    def start(self, job_id: str) -> dict:
        job = self._job(job_id)
        if job["status"] != "pending":
            raise JobError("job is not pending: " + job_id)
        job["status"] = "running"
        job["attempts"] = int(job.get("attempts", 0)) + 1
        job["started_at"] = time.time()
        self.save()
        return job

    # ----------------------------------------------------------------- reports
    def report(self) -> dict:
        counts = {}
        for job in self.jobs["jobs"]:
            key = job["kind"] + ":" + job["status"]
            counts[key] = counts.get(key, 0) + 1
        return {"max_parallel": self.max_parallel, "counts": counts,
                "jobs": [{"job_id": job["job_id"], "kind": job["kind"],
                          "idea_id": job["idea_id"], "status": job["status"],
                          "attempts": job.get("attempts", 0),
                          "stage3_job_id": job.get("stage3_job_id"),
                          "failure_kind": job.get("failure_kind"),
                          "error": job.get("error")} for job in self.jobs["jobs"]]}

    def stage3_results(self) -> list:
        return [{"idea_id": job["idea_id"], "result": job.get("result"),
                 "status": job["status"]}
                for job in self.jobs["jobs"] if job["kind"] == "stage3"]
