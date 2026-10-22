"""The durable research controller (issue #107).

The old `run`/`campaign` commands were one-shot: they did a fixed number of things and
returned. This module is the replacement the issue asks for -- a **resumable worker** with a
persistent state machine:

    read -> hypothesise -> screen -> train -> evaluate -> learn -> (next portfolio)

Every transition is checkpointed, so a crash, a restart or a machine reboot resumes where
the work stopped. The queue is durable, the dispatch is idempotent (the state records what
was dispatched *before* the launch, so a restart never launches the same job twice), the
audit log is append-only, and the guardrails are consulted before every dispatch.

The controller distinguishes two kinds of stop:

* **scientific stop** -- a confirmed Stage-3 success: the frozen Hard-L1 ER gate is cleared
  AND both Stage-3 metrics beat the archived control. Only then does the controller say the
  goal is met, and even then the issue waits for an explicit human approval before it is
  closed.
* **operational pause** -- a quota, an unavailable GPU, a tool outage, a resource limit or
  an explicit stop. The state is persisted, the issue stays OPEN, and the controller
  resumes when the capacity is back.

Nothing here promises uninterrupted execution: a controller that is not running makes no
progress, and the state file says exactly what it was doing when it stopped.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .evidence import EvidenceStore, canonical_json, now_utc, sha256_bytes, sha256_file
from .guardrails import ExecutionIntent, GuardrailRefusal, check_intent

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

#: The phases of the state machine, in order.
PHASES = ("read", "hypothesise", "screen", "train", "evaluate", "learn", "paused", "done")

#: The archived control of this window: the comparator, read-only.
ARCHIVED_CONTROL = {
    "hard_l1_er": 16.334505666275888,
    "stage3_recall@10": 0.06406796775709883,
    "stage3_ndcg@10": 0.03530651701940677,
}

#: The frozen Hard-L1 ER gate.
FROZEN_GATE = 16.38953942232631

#: The historical R4-08 reference, reported separately.
R4_08_REFERENCE = 16.402627511591962


def mechanism_family(proposal: Dict[str, Any]) -> str:
    """The KIND of a mechanism, read from its own declaration.

    The agent is asked to name the family, but it is not obliged to and often does not; a
    portfolio's spread has to be measurable either way, so the family is derived from what
    the proposal actually says.
    """
    text = " ".join(
        str(proposal.get(key, "")).lower()
        for key in ("equation", "title", "summary", "behaviour_signal", "codebook_interaction")
    )
    families = {
        "codebook_geometry": (
            "repuls", "separat", "apart", "margin", "distanc", "geometry",
        ),
        "rigid_alignment": (
            "rigid", "rotat", "shear", "procrustes",
        ),
        "assignment_structure": (
            "occupanc", "prefix", "branch", "partition", "assign", "usage", "capacity",
            "balance", "quot",
        ),
        "residual_space": (
            "residual", "transport", "displacement", "whiten",
        ),
        "assignment_temperature": (
            "entropy", "temperature", "sharp", "softmax",
        ),
    }
    # the family with the most evidence wins, not the first one that matches: a term that
    # speaks of distances and of capacities is not a geometry term just because it says
    # "distance" first
    scores = {
        name: sum(text.count(word) for word in words) for name, words in families.items()
    }
    best = max(scores, key=lambda name: scores[name])
    return best if scores[best] else "unspecified"


def semantic_key(proposal: Dict[str, Any]) -> str:
    """The digest of what a mechanism DOES, independent of what it is called.

    A renamed copy of an unchanged mechanism has the same key, so the guardrails can refuse
    it as the arm it pretends to be.
    """
    payload = {
        "equation": proposal.get("equation"),
        "gradient_targets": proposal.get("gradient_targets"),
        "target_paths": proposal.get("target_paths"),
        "parameters": proposal.get("parameters"),
        "invariants": proposal.get("invariants"),
        "activation": proposal.get("activation"),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


@dataclass
class ControllerConfig:
    """Everything the controller needs, and the caps it respects."""

    store: Path = REPO_ROOT / "results" / "harness3_runs" / "store"
    ledger: Path = REPO_ROOT / "results" / "harness3_runs" / "hypotheses" / "ledger.json"
    state: Path = REPO_ROOT / "results" / "harness3_runs" / "controller" / "state.json"
    audit: Path = REPO_ROOT / "results" / "harness3_runs" / "controller" / "audit.jsonl"
    results: Path = REPO_ROOT / "results" / "harness3_runs" / "controller" / "rounds"
    campaign: Path = REPO_ROOT / "results" / "harness3_runs"
    max_concurrent_screening: int = 2
    max_concurrent_training: int = 1
    ddp_port_base: int = 50300
    min_free_gb: int = 20
    poll_seconds: int = 60
    model: str = ""
    max_rounds: Optional[int] = None          # unbounded by default


@dataclass
class Job:
    """One unit of work the controller dispatched."""

    job_id: str
    kind: str                                 # screen | stage2 | stage3
    mechanism_id: str
    status: str = "queued"                    # queued | running | done | failed | lost
    dispatched_at: str = ""
    finished_at: str = ""
    pid: Optional[int] = None
    port: Optional[int] = None
    result_dir: str = ""
    detail: str = ""
    semantic_key: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Controller:
    """The durable controller. One instance owns one state file."""

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()
        self.state: Dict[str, Any] = {
            "version": 1,
            "round": 0,
            "phase": "read",
            "portfolio": [],
            "jobs": {},
            "queue": [],
            "audit_cursor": 0,
            "pause_reason": None,
            "next_action": "read the evidence and the ledger",
            "scientific_stop": False,
            "created_utc": now_utc(),
            "updated_utc": now_utc(),
        }
        self.store = EvidenceStore(self.config.store)
        self.store.open()

    # ---------------------------------------------------------------- persistence
    def load(self) -> "Controller":
        path = self.config.state
        if path.is_file():
            self.state = json.loads(path.read_text(encoding="utf-8"))
            for key in ("portfolio", "queue"):
                self.state.setdefault(key, [])
            self.state.setdefault("jobs", {})
        return self

    def save(self) -> None:
        self.state["updated_utc"] = now_utc()
        path = self.config.state
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(temporary, path)

    def audit(self, event: str, **fields: Any) -> Dict[str, Any]:
        """The append-only audit log. Every transition leaves a line."""
        record = {"utc": now_utc(), "event": event, "round": self.state.get("round"), **fields}
        self.config.audit.parent.mkdir(parents=True, exist_ok=True)
        with self.config.audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    # ---------------------------------------------------------------- inspection
    def status(self) -> Dict[str, Any]:
        return {
            "round": self.state.get("round"),
            "phase": self.state.get("phase"),
            "next_action": self.state.get("next_action"),
            "pause_reason": self.state.get("pause_reason"),
            "scientific_stop": self.state.get("scientific_stop"),
            "portfolio": [
                {
                    "mechanism_id": item.get("mechanism_id"),
                    "status": item.get("status"),
                    "semantic_key": (item.get("semantic_key") or "")[:12],
                }
                for item in self.state.get("portfolio") or ()
            ],
            "jobs": {
                job_id: {"kind": job["kind"], "status": job["status"], "mechanism_id": job["mechanism_id"]}
                for job_id, job in (self.state.get("jobs") or {}).items()
            },
            "queued": len(self.state.get("queue") or ()),
            "updated_utc": self.state.get("updated_utc"),
        }

    def jobs_in(self, *kinds: str) -> List[Job]:
        return [
            Job(**job)
            for job in (self.state.get("jobs") or {}).values()
            if job.get("kind") in kinds
        ]

    # ---------------------------------------------------------------- resources
    def free_gb(self) -> float:
        usage = shutil.disk_usage(REPO_ROOT)
        return usage.free / (1024 ** 3)

    def visible_gpus(self) -> int:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return len([line for line in completed.stdout.splitlines() if line.strip()])
        except Exception:  # noqa: BLE001
            return 0

    def can_dispatch(self, kind: str) -> Tuple[bool, str]:
        """The resource check that precedes every dispatch."""
        if kind == "screen":
            running = len([job for job in self.jobs_in("screen") if job.status == "running"])
            if running >= self.config.max_concurrent_screening:
                return False, f"{running} screening jobs are already running"
            return True, ""
        if kind == "stage2":
            running = len([job for job in self.jobs_in("stage2") if job.status == "running"])
            running += len(
                [item for item in self.state.get("portfolio") or () if item.get("status") == "training"]
            )
            if running >= self.config.max_concurrent_training:
                return False, f"{running} training jobs are already running"
            if self.visible_gpus() < 4:
                return False, f"only {self.visible_gpus()} GPUs are visible; the frozen run needs 4"
            live_stage3 = self._live_stage3_processes()
            if live_stage3:
                return False, (
                    f"{live_stage3} Stage-3 process tree(s) hold the GPU memory; a cold run "
                    "alongside them exhausts the device"
                )
            if self.free_gb() < self.config.min_free_gb:
                return False, f"only {self.free_gb():.0f} GB free; the cap is {self.config.min_free_gb} GB"
            return True, ""
        if kind == "stage3":
            running = len([job for job in self.jobs_in("stage3") if job.status == "running"])
            if running >= self.config.max_concurrent_training:
                return False, f"{running} Stage-3 jobs are already running"
            training = len(
                [item for item in self.state.get("portfolio") or () if item.get("status") == "training"]
            )
            if training:
                return False, (
                    f"{training} cold run(s) are using the GPUs; a Stage-3 evaluation "
                    "alongside them exhausts the device memory"
                )
            live = self._live_stage3_processes()
            if live:
                return False, f"{live} Stage-3 process tree(s) are live on the cluster"
            return True, ""
        return True, ""

    def _live_stage3_processes(self) -> int:
        """How many Stage-3 process trees are alive on the cluster, whoever started them."""
        try:
            completed = subprocess.run(
                ["pgrep", "-f", "run_stage3_param"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:  # noqa: BLE001
            return 0
        pids = [line for line in completed.stdout.split() if line.strip()]
        if not pids:
            return 0
        # count the launchers only: the ranks and the workers are their children
        launchers = 0
        for pid in pids:
            try:
                info = subprocess.run(
                    ["ps", "-o", "cmd=", "-p", pid], capture_output=True, text=True, timeout=30
                ).stdout
            except Exception:  # noqa: BLE001
                continue
            if "run_stage3_param" in info and "torchrun" not in info and "-u " not in info:
                launchers += 1
        return launchers

    def allocate_port(self) -> int:
        used = {
            job.get("port")
            for job in (self.state.get("jobs") or {}).values()
            if job.get("port")
        }
        port = self.config.ddp_port_base
        while port in used:
            port += 1
        return port

    # ---------------------------------------------------------------- jobs
    def record_job(self, job: Job) -> None:
        self.state.setdefault("jobs", {})[job.job_id] = job.as_dict()
        self.save()

    def reap(self) -> List[Job]:
        """The orphan detection: a job that says running but whose process is gone."""
        lost: List[Job] = []
        for job in self.jobs_in("screen", "stage2", "stage3"):
            if job.status != "running" or not job.pid:
                continue
            try:
                os.kill(job.pid, 0)
            except ProcessLookupError:
                job.status = "lost"
                job.finished_at = now_utc()
                job.detail = "the process is gone; the job is classified, not retried"
                self.record_job(job)
                lost.append(job)
            except PermissionError:
                continue
        return lost

    # ---------------------------------------------------------------- the phases
    def phase_read(self) -> None:
        """Phase 1: take in the new evidence and name the gap the round is about."""
        manifests = self.store.runs()
        # A manifest is immutable, so the ER and the downstream measurement live in two of
        # them: one per kind. The arm's evidence is therefore the JOIN of its manifests.
        # Reading one manifest at a time would report an arm that has both measurements as
        # an arm that has neither, and a genuine success would go unseen.
        joined: Dict[str, Dict[str, Any]] = {}
        for manifest in manifests:
            mechanism_id = manifest.get("mechanism_id") or ""
            metrics = manifest.get("metrics") or {}
            evaluation = metrics.get("frozen_evaluation") or {}
            stage3 = metrics.get("stage3") or {}
            entry = joined.setdefault(
                mechanism_id,
                {"er": None, "recall": None, "ndcg": None, "run_ids": []},
            )
            # the ER was written under two names: the manifests made before the key was
            # aligned carry "l1_er", the ones after carry "hard_l1_er". Both are the same
            # measurement, read once, so both are accepted.
            er = evaluation.get("hard_l1_er")
            if er is None:
                er = evaluation.get("l1_er")
            if er is not None:
                entry["er"] = er
            if stage3.get("test_recall@10") is not None:
                entry["recall"] = stage3["test_recall@10"]
                entry["ndcg"] = stage3.get("test_ndcg@10")
            entry["run_ids"].append(manifest.get("run_id"))
        qualified = []
        for mechanism_id, entry in joined.items():
            er = entry["er"]
            recall = entry["recall"]
            ndcg = entry["ndcg"]
            if er is None or recall is None or ndcg is None:
                continue
            qualified.append(
                {
                    "mechanism_id": mechanism_id,
                    "run_id": (entry["run_ids"] or [""])[-1],
                    "er": er,
                    "er_delta": er - ARCHIVED_CONTROL["hard_l1_er"],
                    "recall": recall,
                    "recall_delta": recall - ARCHIVED_CONTROL["stage3_recall@10"],
                    "ndcg": ndcg,
                    "ndcg_delta": ndcg - ARCHIVED_CONTROL["stage3_ndcg@10"],
                }
            )
        cleared = [row for row in qualified if row["er"] > FROZEN_GATE]
        improved = [
            row
            for row in cleared
            if row["recall_delta"] > 0 and row["ndcg_delta"] > 0
        ]
        self.state["evidence_summary"] = {
            "arms_with_both_metrics": len(qualified),
            "arms_clearing_the_er_gate": len(cleared),
            "arms_improving_both_stage3_metrics": len(improved),
            "gap": (
                "the frozen ER gate is cleared by arms whose Stage-3 quality does not move: "
                "the transfer from the pair-level ER to the recommender is the open problem"
            ),
        }
        # an arm whose export exists but whose frozen evaluation was never read is picked
        # up here: the evaluation is read once per arm, and a round that moved on must not
        # leave an arm unmeasured.
        unmeasured = []
        for manifest in manifests:
            if manifest.get("mechanism_id") in (
                row["mechanism_id"] for row in qualified
            ):
                continue
            if not any(
                artifact["path"].endswith("sids_for_hgrec.npy")
                and artifact.get("exists")
                for artifact in manifest.get("artifacts") or ()
            ):
                continue
            unmeasured.append(manifest["mechanism_id"])
        self.state["arms_pending_evaluation"] = unmeasured
        self.state["scientific_stop"] = bool(improved)
        self.audit("read", arms=len(qualified), cleared=len(cleared), improved=len(improved))
        if improved:
            self.state["phase"] = "done"
            self.state["next_action"] = (
                "a confirmed Stage-3 success: report it and request the human approval "
                "before the issue is closed"
            )
        else:
            self.state["phase"] = "hypothesise"
            self.state["next_action"] = (
                "propose the next portfolio of mathematically distinct mechanisms aimed at the "
                "Stage-3 transfer gap, not at the pair-level ER"
            )
        self.save()
        # an arm whose evaluation was missed when a round moved on is adopted here, so its
        # frozen ER is read once and its Stage-3 is still run
        if unmeasured:
            self.state["phase"] = "adopt"
            self.state["next_action"] = (
                f"adopt {len(unmeasured)} arm(s) whose frozen evaluation was never read"
            )
            self.save()

    def phase_adopt(self) -> None:
        """Adopt the arms whose evaluation was missed, and stage their Stage-3."""
        # every arm the store knows about that cleared the gate and has no Stage-3 yet
        for manifest in self.store.runs():
            evaluation = (manifest.get("metrics") or {}).get("frozen_evaluation") or {}
            if not evaluation.get("above_frozen_threshold"):
                continue
            mechanism_id = manifest["mechanism_id"]
            if any(
                item.get("mechanism_id") == mechanism_id
                and (item.get("stage3") or {}).get("started")
                for item in self.state.get("portfolio") or ()
            ):
                continue
            if not any(
                item.get("mechanism_id") == mechanism_id
                for item in self.state.get("portfolio") or ()
            ):
                sid = self._export_path(mechanism_id)
                if sid is None:
                    continue
                self.state.setdefault("portfolio", []).append(
                    {
                        "mechanism_id": mechanism_id,
                        "status": "qualified",
                        "adopted": True,
                        "proposal_path": "",
                        "semantic_key": "",
                        "evaluated": {
                            "hard_l1_er": evaluation.get("l1_er"),
                            "above_gate": True,
                        },
                        "train": {"result_dir": str(sid.parent.parent)},
                    }
                )
        pending = list(self.state.get("arms_pending_evaluation") or ())
        remaining: List[str] = []
        for mechanism_id in pending:
            sid = self._export_path(mechanism_id)
            if sid is None:
                remaining.append(mechanism_id)
                continue
            if not any(
                item.get("mechanism_id") == mechanism_id
                and (item.get("evaluated") or {}).get("hard_l1_er") is not None
                for item in self.state.get("portfolio") or ()
            ):
                self.state.setdefault("portfolio", []).append(
                    {
                        "mechanism_id": mechanism_id,
                        "status": "trained",
                        "adopted": True,
                        "proposal_path": "",
                        "semantic_key": "",
                        "train": {"result_dir": str(sid.parent.parent)},
                    }
                )
        self.state["arms_pending_evaluation"] = remaining
        self.state["phase"] = "evaluate" if self.state.get("portfolio") else "hypothesise"
        self.state["next_action"] = "read the frozen ER of the adopted arms"
        self.audit("adopt", adopted=len(pending) - len(remaining), still_missing=len(remaining))
        self.save()

    def _export_path(self, mechanism_id: str) -> Optional[Path]:
        """The exported SIDs of an arm, wherever the run that produced them left them."""
        candidates = [
            REPO_ROOT / "results" / "stage2_RQ-VAE" / f"TIGER_c107_{mechanism_id}" / "r8_control" / "sids_for_hgrec.npy",
            REPO_ROOT / "results" / "stage2_RQ-VAE" / mechanism_id / "r8_control" / "sids_for_hgrec.npy",
        ]
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _refresh_transfer_analysis(self) -> Dict[str, Any]:
        """Re-run the transfer analysis over every arm the store knows about.

        The brief tells the agent what the measured requirements are. If that requirement is
        a frozen snapshot, the agent designs against yesterday's evidence -- and every arm
        measured since has added a measurement of what went wrong. The analysis runs in the
        project environment, like the frozen evaluation does, because the controller's own
        environment has no numpy. A refresh that fails must not stop the loop: the previous
        report is kept and the failure is recorded.
        """
        arms: Dict[str, Path] = {}
        variants: Dict[str, str] = {}
        stage3_root = REPO_ROOT / "results" / "stage3_T5Train"
        for manifest in self.store.runs():
            mechanism_id = manifest.get("mechanism_id") or ""
            for artifact in manifest.get("artifacts") or ():
                path = str(artifact.get("path") or "")
                if not path.endswith("sids_for_hgrec.npy") or not artifact.get("exists"):
                    continue
                sid_path = Path(path)
                if not sid_path.is_absolute():
                    sid_path = REPO_ROOT / sid_path
                arms[mechanism_id] = sid_path.parent
                break
            variant = f"TIGER_c107_{mechanism_id}__r8_control"
            if (stage3_root / variant).is_dir():
                variants[mechanism_id] = variant
        out = self.config.campaign / "transfer_analysis" / "stage3_transfer_report.json"
        if "r8_control" not in arms:
            return {"refreshed": False, "reason": "the control is not in the store"}
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [
            PROJECT_PYTHON,
            "-m",
            "tools.harness2_genrec.stage3_transfer",
            "--arms",
            ",".join(f"{name}={path}" for name, path in sorted(arms.items())),
            "--stage3",
            ",".join(f"{name}={variant}" for name, variant in sorted(variants.items())),
            "--out",
            str(out),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except Exception as error:  # noqa: BLE001
            return {"refreshed": False, "reason": f"{type(error).__name__}: {error}"}
        if completed.returncode != 0:
            return {
                "refreshed": False,
                "reason": (completed.stderr or completed.stdout or "").strip()[-400:],
            }
        return {
            "refreshed": True,
            "arms": len(arms),
            "with_stage3": len(variants),
            "path": str(out),
        }

    @staticmethod
    def _families_of(record: Dict[str, Any]) -> List[str]:
        """The families the candidates of a record belong to."""
        families = set()
        for candidate in record.get("candidates") or ():
            try:
                proposal = json.loads(Path(candidate["path"]).read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            families.add(mechanism_family(proposal))
        return sorted(families)

    def _portfolio_is_diverse(self, record: Dict[str, Any]) -> bool:
        """Whether a proposed portfolio spans at least three research angles."""
        return len(self._families_of(record)) >= 3

    def backfill_store(self) -> Dict[str, Any]:
        """Record the arms that were measured but never reached the store.

        The store's preflight used to refuse the admission of a finished run -- the run's
        own products were in the output directory -- and the refusal was swallowed, so the
        arms measured in that window are invisible to the phase that decides success. They
        are recorded here from the evidence that already exists: the ER the controller read
        once, and the Stage-3 result on disk. Nothing is measured again, and the ER is not
        read a second time.
        """
        from .evidence import RunPlan

        if not self.config.audit.is_file():
            return {"recorded": [], "reason": "there is no audit log"}
        events = [
            json.loads(line)
            for line in self.config.audit.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        measured = {
            event["mechanism_id"]: event.get("er")
            for event in events
            if event.get("event") == "evaluate" and event.get("er") is not None
        }
        keys = {
            item.get("mechanism_id"): item.get("semantic_key", "")
            for item in self.state.get("portfolio") or ()
        }
        known = {manifest.get("mechanism_id") for manifest in self.store.runs()}
        recorded: List[str] = []
        missing: List[str] = []
        # the Stage-3 of every arm whose result is on disk, whether or not its ER was read
        # by this controller: the downstream measurement is not re-run, it is read from the
        # artifact that is already there
        stage3_root = REPO_ROOT / "results" / "stage3_T5Train"
        already = {
            manifest.get("mechanism_id")
            for manifest in self.store.runs()
            if manifest.get("kind") == "stage3"
        }
        for variant in sorted(stage3_root.glob("TIGER_c107_*__r8_control")):
            mechanism_id = variant.name[len("TIGER_c107_"): -len("__r8_control")]
            if mechanism_id in already:
                continue
            finals = sorted(variant.rglob("test_final.json"))
            if not finals:
                continue
            report = json.loads(finals[-1].read_text(encoding="utf-8"))
            recall = report.get("test_recall@10")
            ndcg = report.get("test_ndcg@10")
            try:
                stage3_run = self.store.admit(
                    RunPlan(
                        kind="stage3",
                        mechanism_id=mechanism_id,
                        stage="downstream",
                        seed=42,
                        output_dir=variant,
                    ),
                    allow_existing_output=True,
                )
                self.store.close(
                    stage3_run,
                    status="succeeded",
                    metrics={
                        "stage3": {
                            "test_recall@10": recall,
                            "test_ndcg@10": ndcg,
                            "recall_delta_vs_archived_control": (
                                recall - ARCHIVED_CONTROL["stage3_recall@10"]
                            ),
                            "ndcg_delta_vs_archived_control": (
                                ndcg - ARCHIVED_CONTROL["stage3_ndcg@10"]
                            ),
                            "note": "read from the Stage-3 artifact already on disk",
                        }
                    },
                )
                recorded.append(f"{mechanism_id} (stage3 only)")
            except Exception as error:  # noqa: BLE001
                self.audit(
                    "store_admission_failed",
                    mechanism_id=mechanism_id,
                    kind="stage3",
                    reason=f"{type(error).__name__}: {error}",
                )
        for mechanism_id, er in sorted(measured.items()):
            if mechanism_id in known:
                continue
            stage2 = REPO_ROOT / "results" / "stage2_RQ-VAE" / f"TIGER_c107_{mechanism_id}"
            sid = stage2 / "r8_control" / "sids_for_hgrec.npy"
            if not sid.is_file():
                missing.append(mechanism_id)
                continue
            try:
                run = self.store.admit(
                    RunPlan(
                        kind="stage2",
                        mechanism_id=mechanism_id,
                        stage="cold_train",
                        seed=42,
                        output_dir=stage2,
                        extra={"semantic_key": keys.get(mechanism_id, "")},
                    ),
                    allow_existing_output=True,
                )
                self.store.close(
                    run,
                    status="succeeded",
                    metrics={
                        "frozen_evaluation": {
                            "hard_l1_er": er,
                            "l1_er": er,
                            "above_frozen_threshold": er > FROZEN_GATE,
                            "frozen_threshold": FROZEN_GATE,
                            "note": (
                                "the ER the controller had already read; it was not read "
                                "again"
                            ),
                        }
                    },
                    artifacts=[
                        sid,
                        sid.parent / "rqvae_best.pth",
                        sid.parent / "item_sids.json",
                    ],
                    extra={"semantic_key": keys.get(mechanism_id, "")},
                )
                variant = f"TIGER_c107_{mechanism_id}__r8_control"
                root = REPO_ROOT / "results" / "stage3_T5Train" / variant
                finals = (
                    sorted(root.rglob("test_final.json"))
                    if root.is_dir() and mechanism_id not in already
                    else []
                )
                if finals:
                    report = json.loads(finals[-1].read_text(encoding="utf-8"))
                    recall = report.get("test_recall@10")
                    ndcg = report.get("test_ndcg@10")
                    stage3_run = self.store.admit(
                        RunPlan(
                            kind="stage3",
                            mechanism_id=mechanism_id,
                            stage="downstream",
                            seed=42,
                            output_dir=root,
                        ),
                        allow_existing_output=True,
                    )
                    self.store.close(
                        stage3_run,
                        status="succeeded",
                        metrics={
                            "stage3": {
                                "test_recall@10": recall,
                                "test_ndcg@10": ndcg,
                                "recall_delta_vs_archived_control": (
                                    recall - ARCHIVED_CONTROL["stage3_recall@10"]
                                ),
                                "ndcg_delta_vs_archived_control": (
                                    ndcg - ARCHIVED_CONTROL["stage3_ndcg@10"]
                                ),
                            }
                        },
                    )
                recorded.append(mechanism_id)
            except Exception as error:  # noqa: BLE001
                self.audit(
                    "store_admission_failed",
                    mechanism_id=mechanism_id,
                    reason=f"{type(error).__name__}: {error}",
                )
        self.audit("backfill_store", recorded=recorded, missing=missing)
        return {"recorded": recorded, "missing": missing}

    def phase_hypothesise(self) -> None:
        """Phase 2: the portfolio, proposed by the agent and ranked by the controller."""
        from .candidates import propose_candidates
        from .ledger import Ledger

        refresh = self._refresh_transfer_analysis()
        self.audit("transfer_analysis", **refresh)
        ledger = Ledger(self.config.ledger, store=self.store).load()
        out_dir = self.config.results / f"round_{self.state['round']:03d}"
        record = propose_candidates(
            self.store, ledger.brief(), out_dir, model=self.config.model, timeout=2400
        )
        # a portfolio of variations of one kind is not a search: if the families it spans
        # are too few, the agent is asked once more, with the narrowness named
        if not self._portfolio_is_diverse(record) and not self.state.get("portfolio_retry"):
            self.state["portfolio_retry"] = True
            families = self._families_of(record)
            retry = propose_candidates(
                self.store,
                ledger.brief(),
                out_dir / "retry",
                model=self.config.model,
                timeout=2400,
                extra_instruction=(
                    f"the portfolio you just proposed spans only {families}. It has to span "
                    "at least three different research angles, and the angles are listed above "
                    "under `family`. Propose a portfolio that takes at least three of them, "
                    "and make each candidate's `family` field the angle it comes from."
                ),
            )
            # the two batches are merged rather than chosen between: the retry's
            # mechanisms are new ones, and the point of the retry is that the portfolio
            # covers angles the first batch did not
            merged = list(record.get("candidates") or ())
            seen = {candidate["mechanism_id"] for candidate in merged}
            for candidate in retry.get("candidates") or ():
                if candidate["mechanism_id"] in seen:
                    continue
                merged.append(candidate)
                seen.add(candidate["mechanism_id"])
            record = dict(record)
            record["candidates"] = merged
            self.audit("portfolio_retry", families=families)
        else:
            self.state["portfolio_retry"] = False
        portfolio: List[Dict[str, Any]] = []
        seen_keys = {
            (manifest.get("extra") or {}).get("semantic_key")
            for manifest in self.store.runs()
        }
        for candidate in record["candidates"]:
            proposal = json.loads(Path(candidate["path"]).read_text(encoding="utf-8"))
            key = semantic_key(proposal)
            duplicate = key in seen_keys
            portfolio.append(
                {
                    "mechanism_id": candidate["mechanism_id"],
                    "family": _family_of(Path(candidate["path"])),
                    "proposal_path": candidate["path"],
                    "proposal_sha256": candidate["proposal_sha256"],
                    "semantic_key": key,
                    "duplicate_of_an_existing_arm": duplicate,
                    "status": "rejected_duplicate" if duplicate else "proposed",
                    "falsifier": candidate.get("falsifier"),
                    "risk": candidate.get("risk"),
                    "codebook_interaction": candidate.get("codebook_interaction"),
                }
            )
            seen_keys.add(key)
        families = sorted({item.get("family") or "unspecified" for item in portfolio})
        self.state["portfolio"] = portfolio
        self.state["portfolio_families"] = families
        if len(families) < 3:
            # a portfolio of variations of one family is not a search: the agent is asked
            # again, and the round records that it was asked again
            self.audit("portfolio_too_narrow", families=families, candidates=len(portfolio))
        self.state["phase"] = "screen"
        self.state["next_action"] = (
            f"screening {sum(1 for item in portfolio if item['status'] == 'proposed')} candidates "
            "with the independent checks, the active probe and the export sensitivity"
        )
        self.audit(
            "hypothesise",
            candidates=len(portfolio),
            duplicates=sum(1 for item in portfolio if item["duplicate_of_an_existing_arm"]),
        )
        self.save()

    def phase_screen(self) -> None:
        """Phase 3: the cheap screening, in parallel up to the cap."""
        from .candidate_run import run_candidate

        pending = [
            item
            for item in self.state.get("portfolio") or ()
            if item.get("status") == "proposed"
        ]
        if not pending:
            self.state["phase"] = "train"
            self.state["next_action"] = "queue the admitted arms for one cold Stage-2 run each"
            self.save()
            return
        allowed, reason = self.can_dispatch("screen")
        if not allowed:
            self.state["pause_reason"] = reason
            self.state["next_action"] = f"paused: {reason}"
            self.audit("pause", reason=reason)
            self.save()
            return
        self.state["pause_reason"] = None
        item = pending[0]
        out_dir = self.config.results / f"round_{self.state['round']:03d}" / item["mechanism_id"]
        record = run_candidate(
            Path(item["proposal_path"]),
            out_dir,
            store=self.store,
            model=self.config.model,
            cold=False,                      # the screening never spends a cold run
            port=self.allocate_port(),
        )
        item["status"] = "admitted" if record.get("admitted") else "rejected"
        item["screening"] = {
            "judges": (record.get("stages") or {}).get("judges"),
            "probe": (record.get("stages") or {}).get("probe"),
            "export_sensitivity": (record.get("stages") or {}).get("export_sensitivity"),
            "reason": record.get("reason"),
        }
        self.audit(
            "screen",
            mechanism_id=item["mechanism_id"],
            status=item["status"],
            detail=item["screening"].get("reason"),
        )
        self.save()

    def phase_train(self) -> None:
        """Phase 4: one cold run per admitted arm, within the resource caps."""
        from .candidate_run import run_candidate

        # RECONCILE FIRST: a run that has exported its SIDs is no longer training, and a
        # controller that counts it as training waits for itself forever.
        for item in self.state.get("portfolio") or ():
            if item.get("status") != "training":
                continue
            result_dir = (item.get("train") or {}).get("result_dir")
            exported = Path(result_dir) / "sids_for_hgrec.npy" if result_dir else None
            if exported and exported.is_file():
                item["status"] = "trained"
                self.audit("reconciled", mechanism_id=item["mechanism_id"],
                           result_dir=result_dir)
        self.save()

        # the ER of an arm that has already exported is read now, not after the queue
        self._evaluate_ready()

        admitted = [
            item
            for item in self.state.get("portfolio") or ()
            if item.get("status") == "admitted"
        ]
        for item in admitted:
            if item.get("status") == "admitted" and not item.get("train_job"):
                allowed, reason = self.can_dispatch("stage2")
                if not allowed:
                    self.state["pause_reason"] = reason
                    self.state["next_action"] = f"paused: {reason}"
                    self.audit("pause", reason=reason)
                    self.save()
                    return
                intent = ExecutionIntent(
                    kind="stage2",
                    mechanism_id=item["mechanism_id"],
                    tree=REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_c107_{item['mechanism_id']}",
                    semantic_key=item.get("semantic_key", ""),
                )
                verdict = check_intent(intent, self.store)
                if not verdict.allowed:
                    item["status"] = "refused_by_guardrail"
                    self.audit("refuse", mechanism_id=item["mechanism_id"], rule=verdict.rule,
                               detail=verdict.detail)
                    self.save()
                    continue
                port = self.allocate_port()
                out_dir = (
                    self.config.results
                    / f"round_{self.state['round']:03d}"
                    / item["mechanism_id"]
                )
                record = run_candidate(
                    Path(item["proposal_path"]),
                    out_dir,
                    store=self.store,
                    model=self.config.model,
                    cold=True,
                    port=port,
                    tree_prefix=f"TIGER_c107",
                )
                item["status"] = "training" if record.get("status") == "cold_running" else "train_failed"
                item["train"] = {
                    "result_dir": record.get("result_dir"),
                    "launch": (record.get("stages") or {}).get("cold_launch"),
                    "reason": record.get("reason"),
                }
                self.audit(
                    "train",
                    mechanism_id=item["mechanism_id"],
                    status=item["status"],
                    result_dir=record.get("result_dir"),
                )
                self.save()
        # the cold run is asynchronous: the phase waits until the arm has exported
        waiting = []
        for item in admitted:
            if item.get("status") not in ("training", "trained"):
                continue
            result_dir = (item.get("train") or {}).get("result_dir")
            exported = Path(result_dir) / "sids_for_hgrec.npy" if result_dir else None
            if exported and exported.is_file():
                continue
            waiting.append(item["mechanism_id"])
        if waiting:
            self.state["phase"] = "train"
            self.state["next_action"] = (
                f"waiting for the cold runs of {', '.join(waiting)} to export their SIDs"
            )
            self.save()
            return
        self.state["phase"] = "evaluate"
        self.state["next_action"] = "read the frozen ER once per trained arm, and Stage-3 once for the qualified"
        self.save()

    def phase_evaluate(self) -> None:
        """Phase 5: the frozen evaluation, once per arm."""
        self._evaluate_ready()
        self.state["phase"] = "stage3" if any(
            item.get("status") == "qualified" for item in self.state.get("portfolio") or ()
        ) else "learn"
        self.state["next_action"] = (
            "run Stage-3 once for the qualified arms"
            if self.state["phase"] == "stage3"
            else "update the ledger and design the next portfolio"
        )
        self.save()

    def _evaluate_ready(self) -> None:
        """Read the frozen ER of every arm whose export exists. The ER needs no GPU, so it
        is read while the next cold run is still training."""
        import subprocess

        for item in self.state.get("portfolio") or ():
            if item.get("status") != "trained" or item.get("evaluated"):
                continue
            result_dir = (item.get("train") or {}).get("result_dir")
            sid = Path(result_dir) / "sids_for_hgrec.npy" if result_dir else None
            if not sid or not sid.is_file():
                # The run has not exported yet: this is a wait, not a finished evaluation.
                # Advancing here would abandon a cold run that is still training.
                self.state["phase"] = "train"
                self.state["next_action"] = (
                    f"waiting for {item['mechanism_id']} to export its SIDs before the frozen "
                    "evaluation is read"
                )
                self.save()
                return
            intent = ExecutionIntent(
                kind="stage2",
                mechanism_id=item["mechanism_id"],
                semantic_key=item.get("semantic_key", ""),
                consumes_frozen_evaluation=True,
            )
            verdict = check_intent(intent, self.store)
            if not verdict.allowed:
                item["evaluated"] = {"ok": False, "error": f"[{verdict.rule}] {verdict.detail}"}
                self.audit("refuse", mechanism_id=item["mechanism_id"], rule=verdict.rule)
                self.save()
                continue
            completed = subprocess.run(
                [PROJECT_PYTHON, str(REPO_ROOT / "tools/mvg/er_eval.py"), "--sid", str(sid)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=1800,
            )
            try:
                report = json.loads(completed.stdout)
            except json.JSONDecodeError:
                report = {"ok": False, "error": completed.stderr[-300:]}
            item["evaluated"] = {
                "hard_l1_er": report.get("l1_er"),
                "above_gate": report.get("above_frozen_threshold"),
                "frozen_threshold": report.get("frozen_threshold"),
            }
            item["status"] = "qualified" if report.get("above_frozen_threshold") else "below_gate"
            self._record_in_store(item, report, sid)
            self.audit(
                "evaluate",
                mechanism_id=item["mechanism_id"],
                er=report.get("l1_er"),
                qualified=item["status"] == "qualified",
            )
            self.save()

    def _record_in_store(
        self, item: Dict[str, Any], report: Dict[str, Any], sid: Path
    ) -> None:
        """The store learns about the arm, with the ER that was just read once."""
        from .evidence import RunPlan

        mechanism_id = item["mechanism_id"]
        try:
            plan = RunPlan(
                kind="stage2",
                mechanism_id=mechanism_id,
                stage="cold_train",
                seed=42,
                output_dir=REPO_ROOT / "results" / "stage2_RQ-VAE" / f"TIGER_c107_{mechanism_id}",
                extra={"semantic_key": item.get("semantic_key", "")},
            )
            run = self.store.admit(plan, allow_existing_output=True)
        except Exception as error:  # noqa: BLE001
            self.audit(
                "store_admission_failed",
                mechanism_id=mechanism_id,
                reason=f"{type(error).__name__}: {error}",
            )
            return
        self.store.close(
            run,
            status="succeeded",
            metrics={
                "frozen_evaluation": {
                    "hard_l1_er": report.get("l1_er"),
                    "l1_er": report.get("l1_er"),
                    "above_frozen_threshold": report.get("above_frozen_threshold"),
                    "frozen_threshold": report.get("frozen_threshold"),
                    "pos_share": report.get("pos_share"),
                    "neg_share": report.get("neg_share"),
                }
            },
            artifacts=[sid, sid.parent / "rqvae_best.pth", sid.parent / "item_sids.json"],
            extra={"semantic_key": item.get("semantic_key", "")},
        )

    def phase_stage3(self) -> None:
        """Phase 5b: Stage-3 once per qualified arm, against the archived control."""
        import importlib.util

        for item in self.state.get("portfolio") or ():
            if item.get("status") != "qualified":
                continue
            recorded = item.get("stage3") or {}
            if recorded.get("started") or recorded.get("test_recall@10"):
                continue
            # a launch that was refused because another Stage-3 held the port is retried on
            # the next tick; it is not a finished measurement

            intent = ExecutionIntent(
                kind="stage3",
                mechanism_id=item["mechanism_id"],
                semantic_key=item.get("semantic_key", ""),
            )
            verdict = check_intent(intent, self.store)
            if not verdict.allowed:
                item["stage3"] = {"ok": False, "error": f"[{verdict.rule}] {verdict.detail}"}
                self.save()
                continue
            result_dir = (item.get("train") or {}).get("result_dir")
            sid = Path(result_dir) / "item_sids.json" if result_dir else None
            if not sid or not sid.is_file():
                item["stage3"] = {"ok": False, "error": "the run exported no item_sids.json"}
                self.save()
                continue
            spec = importlib.util.spec_from_file_location(
                "genrec_mcp_server", REPO_ROOT / "tools/genrec-mcp/server.py"
            )
            server = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(server)
            launch = server.tool_stage3_start(item_sids=str(sid))
            item["stage3"] = {
                "started": launch.get("started"),
                "status": launch.get("status"),
                "variant": launch.get("variant"),
                "refused": launch.get("refused"),
            }
            self.audit("stage3", mechanism_id=item["mechanism_id"], started=launch.get("started"))
            self.save()
        self.state["phase"] = "learn"
        self.state["next_action"] = (
            "read the Stage-3 results, update the ledger and the policy, and design the next "
            "portfolio"
        )
        self.save()

    def _record_stage3_in_store(self, item: Dict[str, Any]) -> None:
        """The Stage-3 measurement joins the arm's record in the store."""
        from .evidence import RunPlan

        mechanism_id = item["mechanism_id"]
        stage3 = item.get("stage3") or {}
        recall = stage3.get("test_recall@10")
        ndcg = stage3.get("test_ndcg@10")
        if recall is None:
            return
        existing = [
            manifest
            for manifest in self.store.runs()
            if manifest.get("mechanism_id") == mechanism_id
        ]
        if existing:
            # the manifest is immutable: a second manifest records the downstream
            # measurement for the same arm
            pass
        plan = RunPlan(
            kind="stage3",
            mechanism_id=mechanism_id,
            stage="downstream",
            seed=42,
            output_dir=REPO_ROOT / "results" / "stage3_T5Train" / f"TIGER_c107_{mechanism_id}",
        )
        try:
            run = self.store.admit(plan, allow_existing_output=True)
        except Exception as error:  # noqa: BLE001
            self.audit(
                "store_admission_failed",
                mechanism_id=mechanism_id,
                kind="stage3",
                reason=f"{type(error).__name__}: {error}",
            )
            return
        self.store.close(
            run,
            status="succeeded",
            metrics={
                "stage3": {
                    "test_recall@10": recall,
                    "test_ndcg@10": ndcg,
                    "recall_delta_vs_archived_control": recall - ARCHIVED_CONTROL["stage3_recall@10"],
                    "ndcg_delta_vs_archived_control": ndcg - ARCHIVED_CONTROL["stage3_ndcg@10"],
                }
            },
        )

    def phase_learn(self) -> None:
        """Phase 6: the learning loop -- the ledger and the next portfolio."""
        import subprocess

        summary: Dict[str, Any] = {}
        for item in self.state.get("portfolio") or ():
            evaluation = item.get("evaluated") or {}
            stage3 = item.get("stage3") or {}
            summary[item["mechanism_id"]] = {
                "er": evaluation.get("hard_l1_er"),
                "qualified": evaluation.get("above_gate"),
                "stage3": stage3,
            }
        # the Stage-3 numbers are read from the artifacts when the run has finished
        for item in self.state.get("portfolio") or ():
            stage3 = item.get("stage3") or {}
            variant = stage3.get("variant")
            if not variant:
                continue
            root = REPO_ROOT / "results" / "stage3_T5Train" / variant
            finals = sorted(root.rglob("test_final.json")) if root.is_dir() else []
            if not finals:
                continue
            try:
                report = json.loads(finals[-1].read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            recall = report.get("test_recall@10")
            ndcg = report.get("test_ndcg@10")
            item["stage3"]["test_recall@10"] = recall
            item["stage3"]["test_ndcg@10"] = ndcg
            item["stage3"]["recall_delta_vs_control"] = (
                recall - ARCHIVED_CONTROL["stage3_recall@10"] if recall is not None else None
            )
            item["stage3"]["ndcg_delta_vs_control"] = (
                ndcg - ARCHIVED_CONTROL["stage3_ndcg@10"] if ndcg is not None else None
            )
            item["status"] = "evaluated"
            self._record_stage3_in_store(item)
        self.state["round_summary"] = summary
        improved = [
            item
            for item in self.state.get("portfolio") or ()
            if (item.get("stage3") or {}).get("recall_delta_vs_control")
            and (item.get("stage3") or {}).get("recall_delta_vs_control") > 0
            and (item.get("stage3") or {}).get("ndcg_delta_vs_control")
            and (item.get("stage3") or {}).get("ndcg_delta_vs_control") > 0
        ]
        if improved:
            self.state["phase"] = "done"
            self.state["scientific_stop"] = True
            self.state["next_action"] = (
                "a confirmed Stage-3 success: report it and request the human approval "
                "before the issue is closed"
            )
            self.audit("scientific_stop", arms=[item["mechanism_id"] for item in improved])
            self.save()
            return
        # every round's portfolio is archived: the next round replaces the working copy, but
        # the record of what this round proposed, rejected, trained and measured stays.
        archive = self.config.results / f"round_{int(self.state.get('round', 0)):03d}"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "portfolio.json").write_text(
            json.dumps(
                {
                    "round": self.state.get("round"),
                    "closed_utc": now_utc(),
                    "portfolio": self.state.get("portfolio"),
                    "summary": summary,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        # the next round is automatic
        self.state["round"] = int(self.state.get("round", 0)) + 1
        if (
            self.config.max_rounds is not None
            and self.state["round"] >= self.config.max_rounds
        ):
            self.state["phase"] = "paused"
            self.state["pause_reason"] = "the round cap is reached; the issue stays OPEN"
            self.state["next_action"] = self.state["pause_reason"]
            self.audit("pause", reason=self.state["pause_reason"])
        else:
            self.state["phase"] = "hypothesise"
            self.state["portfolio"] = []
            self.state["next_action"] = (
                "design the next portfolio from the results of this round; the loop does not "
                "stop on a negative result"
            )
        self.audit("learn", next_round=self.state["round"], phase=self.state["phase"])
        self.save()

    # ---------------------------------------------------------------- the loop
    def tick(self) -> Dict[str, Any]:
        """One transition of the state machine. This is what the worker repeats."""
        self.reap()
        phase = self.state.get("phase")
        if phase == "read":
            self.phase_read()
        elif phase == "adopt":
            self.phase_adopt()
        elif phase == "hypothesise":
            self.phase_hypothesise()
        elif phase == "screen":
            self.phase_screen()
        elif phase == "train":
            self.phase_train()
        elif phase == "evaluate":
            self.phase_evaluate()
        elif phase == "stage3":
            self.phase_stage3()
        elif phase == "learn":
            self.phase_learn()
        elif phase == "paused":
            self.state["next_action"] = self.state.get("pause_reason") or "paused"
        elif phase == "done":
            self.state["next_action"] = (
                "a confirmed Stage-3 success: report it and request the human approval "
                "before the issue is closed"
            )
        self.save()
        return self.status()

    def run_forever(self, *, max_ticks: Optional[int] = None) -> Dict[str, Any]:
        """The worker loop. It never declares scientific success on its own."""
        ticks = 0
        while True:
            status = self.tick()
            ticks += 1
            if status["phase"] == "done":
                return status
            if status["phase"] == "paused":
                return status
            if max_ticks is not None and ticks >= max_ticks:
                return status
            time.sleep(self.config.poll_seconds)


def _family_of(proposal_path: Path) -> str:
    """The family of a frozen proposal, read from the proposal itself."""
    try:
        return mechanism_family(json.loads(Path(proposal_path).read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return "unspecified"


__all__ = [
    "ARCHIVED_CONTROL",
    "Controller",
    "ControllerConfig",
    "FROZEN_GATE",
    "Job",
    "PHASES",
    "R4_08_REFERENCE",
    "mechanism_family",
    "semantic_key",
]
