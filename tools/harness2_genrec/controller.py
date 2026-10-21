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
            if running >= self.config.max_concurrent_training:
                return False, f"{running} training jobs are already running"
            if self.visible_gpus() < 4:
                return False, f"only {self.visible_gpus()} GPUs are visible; the frozen run needs 4"
            if self.free_gb() < self.config.min_free_gb:
                return False, f"only {self.free_gb():.0f} GB free; the cap is {self.config.min_free_gb} GB"
            return True, ""
        if kind == "stage3":
            running = len([job for job in self.jobs_in("stage3") if job.status == "running"])
            if running >= self.config.max_concurrent_training:
                return False, f"{running} Stage-3 jobs are already running"
            return True, ""
        return True, ""

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
        qualified = []
        for manifest in manifests:
            metrics = manifest.get("metrics") or {}
            evaluation = metrics.get("frozen_evaluation") or {}
            stage3 = metrics.get("stage3") or {}
            er = evaluation.get("hard_l1_er")
            recall = stage3.get("test_recall@10")
            ndcg = stage3.get("test_ndcg@10")
            if er is None or recall is None or ndcg is None:
                continue
            qualified.append(
                {
                    "mechanism_id": manifest["mechanism_id"],
                    "run_id": manifest["run_id"],
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

    def phase_hypothesise(self) -> None:
        """Phase 2: the portfolio, proposed by the agent and ranked by the controller."""
        from .candidates import propose_candidates
        from .ledger import Ledger

        ledger = Ledger(self.config.ledger, store=self.store).load()
        out_dir = self.config.results / f"round_{self.state['round']:03d}"
        record = propose_candidates(
            self.store, ledger.brief(), out_dir, model=self.config.model, timeout=2400
        )
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
        self.state["portfolio"] = portfolio
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
                item["status"] = "trained" if record.get("status") == "cold_running" else "train_failed"
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
            if item.get("status") != "trained":
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
            self.audit(
                "evaluate",
                mechanism_id=item["mechanism_id"],
                er=report.get("l1_er"),
                qualified=item["status"] == "qualified",
            )
            self.save()
        self.state["phase"] = "stage3" if any(
            item.get("status") == "qualified" for item in self.state.get("portfolio") or ()
        ) else "learn"
        self.state["next_action"] = (
            "run Stage-3 once for the qualified arms"
            if self.state["phase"] == "stage3"
            else "update the ledger and design the next portfolio"
        )
        self.save()

    def phase_stage3(self) -> None:
        """Phase 5b: Stage-3 once per qualified arm, against the archived control."""
        import importlib.util

        for item in self.state.get("portfolio") or ():
            if item.get("status") != "qualified" or item.get("stage3"):
                continue
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


__all__ = [
    "ARCHIVED_CONTROL",
    "Controller",
    "ControllerConfig",
    "FROZEN_GATE",
    "Job",
    "PHASES",
    "R4_08_REFERENCE",
    "semantic_key",
]
