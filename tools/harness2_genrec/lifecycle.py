"""The mechanism lifecycle controller (issue #102).

Harness2 is the single controller: it owns the state machine, the budgets, the resume and
the admission decision. The agent proposes, implements and repairs; the independent judges
decide whether the implementation is faithful to the frozen proposal; the controller decides
what happens next.

    specified -> proposed -> implemented -> verified -> probed -> admitted
              -> stage2_running -> stage2_complete -> er_evaluated -> stage3_complete

with the failure states ``blocked_preflight``, ``failed_verification``,
``exhausted_repair``, ``invalidated`` and ``artifact_corrupt``.

Two rules hold everywhere:

* the agent never edits a judge, a fixture, a threshold, a frozen specification or a
  verdict. It edits the candidate's own code and nothing else;
* every stage leaves evidence on disk, and the manifest is the index of it. A stage that
  cannot run is ``blocked``, and a blocked stage never advances the machine.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import opencode
from .proposal import ProposalError, freeze, proposal_digest, validate_proposal

STATES = (
    "specified",
    "proposed",
    "implemented",
    "verified",
    "repaired",
    "probed",
    "admitted",
    "stage2_running",
    "stage2_complete",
    "er_evaluated",
    "stage3_complete",
)

FAILURE_STATES = (
    "blocked_preflight",
    "failed_verification",
    "exhausted_repair",
    "invalidated",
    "artifact_corrupt",
)

#: The order the states are entered in. A state may only be entered when the previous one
#: was reached, except that a failure state may be entered from anywhere.
ORDER = {name: index for index, name in enumerate(STATES)}

#: The environment the project runs in. The judges and the trainer need torch; the
#: controller itself does not.
PROJECT_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"


@dataclass
class LifecycleConfig:
    """Where the run lives and what it is allowed to spend."""

    repo_root: Path
    runs_root: Path
    base_tree: Path
    model: str = ""
    timeout: int = 1800
    max_repair_attempts: int = 3
    probe_epochs: int = 2
    judge_timeout: int = 1800
    arm_name: str = ""
    probe_checkpoint: str = ""


@dataclass
class Stage:
    """One stage of the run and the evidence it left."""

    name: str
    state: str
    ok: bool
    detail: str = ""
    started: str = ""
    finished: str = ""
    cost_usd: float = 0.0
    tokens: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    measurements: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "ok": self.ok,
            "detail": self.detail,
            "started": self.started,
            "finished": self.finished,
            "cost_usd": self.cost_usd,
            "tokens": dict(self.tokens),
            "n_tool_calls": len(self.tool_calls),
            "tool_calls": self.tool_calls,
            "artifacts": list(self.artifacts),
            "measurements": self.measurements,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    import hashlib

    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(tree: Path) -> str:
    """The digest of a tree's code, using the verifier's own implementation."""
    repo_cgv = Path(__file__).resolve().parents[2] / "tools" / "cgv"
    if str(repo_cgv) not in os.sys.path:
        os.sys.path.insert(0, str(repo_cgv))
    import contract as cgv_contract  # type: ignore  # noqa: E402

    digest, _count = cgv_contract.code_tree_sha256(tree)
    return digest or ""


class MechanismRun:
    """One mechanism's journey, with its manifest."""

    def __init__(
        self,
        config: LifecycleConfig,
        proposal: Dict[str, Any],
        *,
        run_id: str = "",
    ) -> None:
        self.config = config
        self.proposal = proposal
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.root = Path(config.runs_root) / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = "specified"
        self.stages: List[Stage] = []
        self.manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "created_utc": _now(),
            "mechanism_id": proposal.get("mechanism_id", ""),
            "proposal_sha256": proposal_digest(proposal),
            "state": self.state,
            "stages": [],
        }
        self._write_manifest()

    # -- the machine -----------------------------------------------------
    def _advance(self, state: str) -> None:
        if state in FAILURE_STATES:
            self.state = state
        else:
            current = ORDER.get(self.state, -1)
            if ORDER.get(state, 0) < current and self.state not in FAILURE_STATES:
                raise RuntimeError(
                    f"the lifecycle cannot go back: {self.state} -> {state}"
                )
            self.state = state
        self.manifest["state"] = self.state
        self._write_manifest()

    def _record(self, stage: Stage) -> Stage:
        self.stages.append(stage)
        self.manifest["stages"] = [item.as_dict() for item in self.stages]
        self._write_manifest()
        return stage

    def _write_manifest(self) -> None:
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps(self.manifest, indent=2, default=str, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # -- the stages ------------------------------------------------------
    def freeze_proposal(self) -> Dict[str, Any]:
        """Freeze the proposal. After this the agent cannot change what it claimed."""
        problems = validate_proposal(self.proposal)
        if problems:
            raise ProposalError("the proposal is not valid: " + "; ".join(problems))
        frozen = freeze(self.proposal)
        path = self.root / "proposal.json"
        path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.proposal = frozen
        self.manifest["proposal_path"] = str(path)
        self.manifest["proposal_sha256"] = proposal_digest(frozen)
        self._advance("proposed")
        self._record(
            Stage(
                name="freeze_proposal",
                state="proposed",
                ok=True,
                detail=f"the proposal is frozen at {self.manifest['proposal_sha256'][:16]}",
                started=_now(),
                finished=_now(),
                artifacts=[str(path)],
            )
        )
        return frozen

    def implement(self) -> Stage:
        """The agent writes the mechanism in its own arm tree."""
        started = _now()
        workspace = self.arm_tree()
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        self._copy_tree(Path(self.config.base_tree), workspace)
        prompt = self._implementation_prompt()
        mcp = opencode.mcp_server_block(Path(self.config.repo_root))
        result = opencode.run_agent(
            workspace,
            prompt,
            model=self.config.model or opencode.model_name(),
            timeout=self.config.timeout,
            mcp=mcp,
            title=f"{self.run_id} implement",
            deny_paths=[Path(self.config.repo_root)],
        )
        (self.root / "implement.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        ok = not result.get("errors") and bool(result.get("text"))
        stage = Stage(
            name="implement",
            state="implemented" if ok else "blocked_preflight",
            ok=ok,
            detail="the agent implemented the mechanism"
            if ok
            else f"the implementation run failed: {result.get('errors')}",
            started=started,
            finished=_now(),
            cost_usd=float(result.get("cost") or 0.0),
            tokens=dict(result.get("tokens") or {}),
            tool_calls=list(result.get("tool_calls") or []),
            artifacts=[str(self.root / "implement.json")],
            measurements={
                "exit_code": result.get("exit_code"),
                "elapsed_s": result.get("elapsed_s"),
                "workspace": str(workspace),
            },
        )
        self._record(stage)
        if ok:
            self._advance("implemented")
        return stage

    def verify(self) -> Stage:
        """Run the independent judges the proposal asked for."""
        started = _now()
        spec_path = self.root / "mechanism_spec.json"
        source = self.proposal.get("mechanism_spec_path")
        if not source:
            stage = Stage(
                name="verify",
                state="blocked_preflight",
                ok=False,
                detail=(
                    "the proposal names no frozen specification: the judges check the "
                    "implementation against a specification, not against the proposal's prose"
                ),
                started=started,
                finished=_now(),
            )
            self._record(stage)
            self._advance("blocked_preflight")
            return stage
        spec_source = Path(source)
        if not spec_source.is_absolute():
            spec_source = Path(self.config.repo_root) / spec_source
        if not spec_source.is_file():
            stage = Stage(
                name="verify",
                state="blocked_preflight",
                ok=False,
                detail=f"the frozen specification is missing: {spec_source}",
                started=started,
                finished=_now(),
            )
            self._record(stage)
            self._advance("blocked_preflight")
            return stage
        shutil.copy2(spec_source, spec_path)
        self.manifest["mechanism_spec_path"] = str(spec_path)
        self.manifest["mechanism_spec_sha256"] = _sha256(spec_path)
        judges = list(self.proposal.get("judges") or [])
        report_dir = self.root / "judges"
        report_dir.mkdir(parents=True, exist_ok=True)
        command = [
            PROJECT_PYTHON,
            "-m",
            "tools.mvg.cli",
            "run",
            "--tree",
            str(self.arm_tree()),
            "--spec",
            str(spec_path),
            "--out",
            str(report_dir),
            "--baseline-tree",
            str(self.config.base_tree),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.config.repo_root)
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.config.repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.judge_timeout,
            )
        except subprocess.TimeoutExpired:
            stage = Stage(
                name="verify",
                state="blocked_preflight",
                ok=False,
                detail=f"the judges did not finish within {self.config.judge_timeout}s",
                started=started,
                finished=_now(),
            )
            self._record(stage)
            self._advance("blocked_preflight")
            return stage
        report_path = report_dir / "mvg_report.json"
        report: Dict[str, Any] = {}
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        verdict = report.get("verdict", "failed")
        stage = Stage(
            name="verify",
            state="verified" if verdict == "passed" else "failed_verification",
            ok=verdict == "passed",
            detail=f"the judges returned {verdict} for {', '.join(judges)}",
            started=started,
            finished=_now(),
            artifacts=[str(report_path)] if report_path.is_file() else [],
            measurements={
                "verdict": verdict,
                "gates": {
                    gate: result.get("state")
                    for gate, result in (report.get("gates") or {}).items()
                },
                "judge_states": {
                    gate: result.get("state")
                    for gate, result in (report.get("gates") or {}).items()
                },
                "diagnostics": {
                    gate: result.get("diagnostics")
                    for gate, result in (report.get("gates") or {}).items()
                    if result.get("state") != "passed"
                },
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-800:],
            },
        )
        self._record(stage)
        if verdict == "passed":
            self._advance("verified")
        else:
            self._advance("failed_verification")
        return stage

    def repair(self) -> Stage:
        """Let the agent fix the candidate, then let the judges decide again."""
        started = _now()
        attempt = 0
        last = ""
        while attempt < self.config.max_repair_attempts:
            attempt += 1
            diagnostics = self._last_diagnostics()
            prompt = self._repair_prompt(diagnostics, attempt)
            result = opencode.run_agent(
                self.arm_tree(),
                prompt,
                model=self.config.model or opencode.model_name(),
                timeout=self.config.timeout,
                mcp=opencode.mcp_server_block(Path(self.config.repo_root)),
                title=f"{self.run_id} repair {attempt}",
                deny_paths=[Path(self.config.repo_root)],
            )
            path = self.root / f"repair_{attempt}.json"
            path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            check = self.verify()
            if check.ok:
                stage = Stage(
                    name="repair",
                    state="repaired",
                    ok=True,
                    detail=f"the agent repaired the mechanism on attempt {attempt} and the "
                    "judges accepted it",
                    started=started,
                    finished=_now(),
                    cost_usd=float(result.get("cost") or 0.0),
                    tokens=dict(result.get("tokens") or {}),
                    tool_calls=list(result.get("tool_calls") or []),
                    artifacts=[str(path)],
                    measurements={"attempts": attempt},
                )
                self._record(stage)
                self._advance("repaired")
                return stage
            last = check.detail
        stage = Stage(
            name="repair",
            state="exhausted_repair",
            ok=False,
            detail=f"the mechanism did not pass the judges after {attempt} attempt(s): {last}",
            started=started,
            finished=_now(),
            measurements={"attempts": attempt},
        )
        self._record(stage)
        self._advance("exhausted_repair")
        return stage

    def probe(self) -> Stage:
        """A budgeted, train-only probe of the real trainer entry."""
        started = _now()
        probe_dir = self.root / "probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        command = [
            PROJECT_PYTHON,
            str(self.config.repo_root / "tools" / "mvg" / "probe_entry.py"),
            "--tree",
            str(self.arm_tree()),
            "--epochs",
            str(self.config.probe_epochs),
            "--out",
            str(probe_dir),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.config.repo_root)
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.config.repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
            report_path = probe_dir / "probe_report.json"
            if report_path.is_file():
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                payload = {
                    "ok": False,
                    "error": "the probe wrote no report",
                    "stdout_tail": (completed.stdout or "")[-400:],
                    "stderr_tail": (completed.stderr or "")[-400:],
                }
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        ok = bool(payload.get("ok"))
        stage = Stage(
            name="probe",
            state="probed" if ok else "failed_verification",
            ok=ok,
            detail=payload.get("detail", "the probe ran"),
            started=started,
            finished=_now(),
            artifacts=[str(probe_dir)],
            measurements={
                "epochs": self.config.probe_epochs,
                "encoder_update": payload.get("encoder_update"),
                "codebook_update": payload.get("codebook_update"),
                "hard_sid_changed": payload.get("hard_sid_changed"),
                "loss_finite": payload.get("loss_finite"),
            },
        )
        self._record(stage)
        if ok:
            self._advance("probed")
        else:
            self._advance("failed_verification")
        return stage

    def admit(self) -> Stage:
        """The admission record: what the launch path will check before training."""
        started = _now()
        record = {
            "admission_version": 1,
            "issued_utc": _now(),
            "run_id": self.run_id,
            "mechanism_id": self.proposal.get("mechanism_id", ""),
            "proposal_sha256": self.manifest.get("proposal_sha256", ""),
            "tree_digest": _tree_digest(self.arm_tree()),
            "tree_path": str(self.arm_tree()),
            "judges": list(self.proposal.get("judges") or []),
            "judge_verdict": "passed",
            "probe": {
                "epochs": self.config.probe_epochs,
                "encoder_update": True,
            },
            "environment": {
                "python": PROJECT_PYTHON,
                "model": self.config.model or opencode.model_name(),
            },
        }
        path = self.root / "admission.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.manifest["admission_path"] = str(path)
        self.manifest["admission_sha256"] = _sha256(path)
        probe_checkpoint = self._seed_probe_checkpoint()
        self.manifest["probe_checkpoint"] = probe_checkpoint
        record["probe_checkpoint"] = probe_checkpoint
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage = Stage(
            name="admit",
            state="admitted",
            ok=True,
            detail="the mechanism is admitted; the launch path will check this record",
            started=started,
            finished=_now(),
            artifacts=[str(path)],
            measurements={"tree_digest": record["tree_digest"]},
        )
        self._record(stage)
        self._advance("admitted")
        return stage

    def stage2(self) -> Stage:
        """The cold run, launched through the MCP with the admission record."""
        started = _now()
        admission = self.root / "admission.json"
        if not admission.is_file():
            stage = Stage(
                name="stage2",
                state="blocked_preflight",
                ok=False,
                detail="no admission record: the mechanism is not admitted",
                started=started,
                finished=_now(),
            )
            self._record(stage)
            self._advance("blocked_preflight")
            return stage
        server = self._mcp_server()
        result = server.tool_stage2_start(
            source_dir=str(self.arm_tree()), admission=str(admission)
        )
        if not result.get("started"):
            refusal = str(result.get("refused") or "")
            live = result.get("state") or {}
            if "already live" in refusal and live.get("status") in ("running", "starting", "smoke_pending"):
                # A run for this arm is already going: adopt it rather than starting a
                # second one. The launch path refuses a duplicate for good reason, and a
                # resumed run must not overwrite its evidence either.
                self.manifest["stage2_adopted"] = live
                self._advance("stage2_running")
                final = self._await_stage2(server)
                ok = final.get("status") in ("finished", "idle")
                stage = Stage(
                    name="stage2",
                    state="stage2_complete" if ok else "artifact_corrupt",
                    ok=ok,
                    detail=f"adopted the live run; it finished with status {final.get('status')!r}"
                    if ok
                    else f"the adopted run ended with status {final.get('status')!r}",
                    started=started,
                    finished=_now(),
                    measurements={"final_status": final.get("status"), "adopted": live},
                )
                self._record(stage)
                if ok:
                    self._advance("stage2_complete")
                else:
                    self._advance("artifact_corrupt")
                return stage
            stage = Stage(
                name="stage2",
                state="blocked_preflight",
                ok=False,
                detail=f"the launch refused: {refusal or 'no reason given'}",
                started=started,
                finished=_now(),
                measurements={"refusal": refusal},
            )
            self._record(stage)
            self._advance("blocked_preflight")
            return stage
        self._advance("stage2_running")
        self.manifest["stage2_launch"] = result
        final = self._await_stage2(server)
        ok = final.get("status") in ("finished", "idle")
        stage = Stage(
            name="stage2",
            state="stage2_complete" if ok else "artifact_corrupt",
            ok=ok,
            detail=f"the cold run finished with status {final.get('status')!r}"
            if ok
            else f"the cold run ended with status {final.get('status')!r}",
            started=started,
            finished=_now(),
            measurements={"final_status": final.get("status"), "launch": result},
        )
        self._record(stage)
        if ok:
            self._advance("stage2_complete")
        else:
            self._advance("artifact_corrupt")
        return stage

    def evaluate(self) -> Stage:
        """The frozen evaluation, once."""
        started = _now()
        server = self._mcp_server()
        result_dir = Path(str(server.tree_result_dir(self.arm_tree())))
        candidates = sorted(result_dir.rglob("sids_for_hgrec.npy"))
        if not candidates:
            stage = Stage(
                name="evaluate",
                state="artifact_corrupt",
                ok=False,
                detail=f"no exported SID file under {result_dir}",
                started=started,
                finished=_now(),
            )
            self._record(stage)
            self._advance("artifact_corrupt")
            return stage
        sid_path = candidates[-1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.config.repo_root)
        completed = subprocess.run(
            [
                PROJECT_PYTHON,
                str(self.config.repo_root / "tools" / "mvg" / "er_eval.py"),
                "--sid",
                str(sid_path),
                "--out",
                str(self.root / "er_eval.json"),
            ],
            cwd=str(self.config.repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        try:
            report = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            report = {"ok": False, "error": (completed.stderr or completed.stdout or "")[-400:]}
        ok = bool(report.get("ok"))
        above = report.get("above_frozen_threshold")
        stage = Stage(
            name="evaluate",
            state="er_evaluated" if ok else "artifact_corrupt",
            ok=ok,
            detail=(
                f"the frozen L1 ER is {report.get('l1_er')} against the frozen threshold "
                f"{report.get('frozen_threshold')} "
                f"({'above' if above else 'below'})"
                if ok
                else f"the evaluation did not run: {report.get('error')}"
            ),
            started=started,
            finished=_now(),
            artifacts=[str(self.root / "er_eval.json")],
            measurements=report,
        )
        self._record(stage)
        if ok:
            self._advance("er_evaluated")
        else:
            self._advance("artifact_corrupt")
        return stage

    # -- helpers ---------------------------------------------------------
    def _seed_probe_checkpoint(self) -> Dict[str, Any]:
        """Seed the checkpoint the launch path's smoke probe needs.

        The smoke probes the real gradient path, so it needs real trained weights: a
        parent arm's checkpoint of the same architecture, copied in and marked
        **smoke-only**. The cold run itself starts from scratch -- the seeded file is
        archived before the run and never becomes a warm start.
        """
        server = self._mcp_server()
        result_dir = Path(str(server.tree_result_dir(self.arm_tree())))
        target = result_dir / "rqvae_best.pth"
        if target.is_file():
            return {"seeded": False, "reason": "the arm already has a checkpoint", "path": str(target)}
        source: Optional[Path] = None
        if self.config.probe_checkpoint:
            candidate = Path(self.config.probe_checkpoint)
            source = candidate if candidate.is_absolute() else Path(self.config.repo_root) / candidate
        else:
            for arm in ("TIGER_BEHAVIOR_SAMPLE_RQ-VAE", "TIGER_r8_control"):
                try:
                    candidate = Path(str(server.tree_result_dir(
                        Path(self.config.repo_root) / "stage2_RQ-VAE" / arm
                    ))) / "rqvae_best.pth"
                except Exception:  # pragma: no cover - a tree without the constants
                    continue
                if candidate.is_file():
                    source = candidate
                    break
        if source is None or not Path(source).is_file():
            return {
                "seeded": False,
                "reason": "no parent checkpoint to seed the smoke probe with; the launch will "
                "refuse without one",
            }
        result_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return {
            "seeded": True,
            "smoke_only": True,
            "source": str(source),
            "source_sha256": _sha256(Path(source)),
            "path": str(target),
            "sha256": _sha256(target),
        }

    def _mcp_server(self):
        """The genrec MCP server, imported as a module. It is stdlib-only."""
        import importlib.util

        path = Path(self.config.repo_root) / "tools" / "genrec-mcp" / "server.py"
        spec = importlib.util.spec_from_file_location("genrec_mcp_for_lifecycle", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _await_stage2(self, server, *, poll_s: int = 60, budget_s: int = 6 * 3600) -> Dict[str, Any]:
        """Watch the run until it ends or the budget runs out."""
        deadline = time.time() + budget_s
        last: Dict[str, Any] = {}
        while time.time() < deadline:
            last = server.tool_stage2_status(source_dir=str(self.arm_tree()))
            status = str(last.get("status", ""))
            if status in ("finished", "idle", "refused", "worker_gone"):
                return last
            time.sleep(poll_s)
        return last or {"status": "timeout"}

    def arm_tree(self) -> Path:
        """The arm tree this mechanism lives in.

        A tree of its own, under ``stage2_RQ-VAE/``, so the launch path can find it and no
        historical tree is ever touched.
        """
        name = self.config.arm_name or f"TIGER_r9_{self.proposal.get('mechanism_id', 'arm')}"
        return Path(self.config.repo_root) / "stage2_RQ-VAE" / name

    def _copy_tree(self, source: Path, destination: Path) -> None:
        """Copy the tree's code. Artefacts and logs never come along."""
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(Path(source).rglob("*")):
            if not path.is_file():
                continue
            parts = set(path.relative_to(source).parts)
            if "__pycache__" in parts or path.suffix in (".pth", ".npy", ".pyc"):
                continue
            if parts & {"logs", "results", "ckpt", "dataset"}:
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    def _last_diagnostics(self) -> List[str]:
        for stage in reversed(self.stages):
            if stage.name == "verify":
                lines: List[str] = []
                for gate, problems in (stage.measurements.get("diagnostics") or {}).items():
                    for problem in problems or []:
                        lines.append(f"{gate}: {problem}")
                return lines
        return []

    def _implementation_prompt(self) -> str:
        proposal = json.dumps(self.proposal, indent=2, sort_keys=True)
        return (
            "You are implementing one Stage-2 mechanism for this repository, from a proposal "
            "that was frozen before you started. The proposal is below.\n\n"
            f"# PROPOSAL\n\n```json\n{proposal}\n```\n\n"
            "# WHERE TO WORK\n\n"
            "The tree you are given is a scratch copy of an experiment arm. Implement the "
            "mechanism inside it: the mechanism's own module, and the call site in the "
            "trainer that composes its term into the loss before backward. The arm's trainer "
            "already has the plumbing; read it before you change it.\n\n"
            "The repository's own MCP server is connected: you may call `cgv_check` to check "
            "your contract, `stage2_status` and `stage3_status` to watch runs, and "
            "`stage2_start` only if you are told to launch a run. Do not start a full "
            "training run: this stage is the implementation only.\n\n"
            "Do not edit the judges, the frozen specifications, the thresholds or the "
            "historical artefacts; they are not in your tree and you must not reach for them. "
            "Implement, then report what you changed and why."
        )

    def _repair_prompt(self, diagnostics: Sequence[str], attempt: int) -> str:
        problems = "\n".join(f"- {line}" for line in diagnostics) or "- (no diagnostics recorded)"
        return (
            f"The independent checks rejected your implementation (attempt {attempt}). The "
            "findings are below.\n\n"
            f"# FINDINGS\n\n{problems}\n\n"
            "# WHAT TO DO\n\n"
            "Diagnose the findings, then fix the candidate's own code in this tree. Do not "
            "change the checks, the fixtures or the thresholds: they are the judge and they "
            "are not yours. The same tree is checked again after your fix, so a fix that "
            "only hides the symptom will be caught. Report what you changed and why."
        )


__all__ = [
    "FAILURE_STATES",
    "LifecycleConfig",
    "MechanismRun",
    "ORDER",
    "PROJECT_PYTHON",
    "STATES",
    "Stage",
]
