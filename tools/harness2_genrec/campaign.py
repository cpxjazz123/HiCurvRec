"""The autonomous research campaign (issue #103).

Harness2 is the sole orchestrator. It decides which mechanisms are proposed, which are
qualified, which are run, and what the next round is; OpenCode is the only agent that
writes a mechanism or repairs one. Nothing here is scripted: the researcher, the
implementer and the repairer are all real OpenCode sessions, and the decisions are
recorded with the evidence they rest on.

The campaign is **budgeted before it runs**: the parallelism, the number of cold runs and
the wall clock are declared in the configuration, and the controller refuses to exceed
them. It is also **resumable**: every stage leaves its evidence on disk, and a run that
was interrupted picks up where it stopped instead of re-running what already happened.

The order is the one the issue sets: propose -> freeze -> implement -> judge -> repair ->
probe -> reflect -> second round -> select -> cold Stage2 -> one frozen evaluation ->
report. A candidate that fails a gate is recorded as a failure, with the reason; it is
never quietly dropped.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import ideation
from . import opencode
from .lifecycle import LifecycleConfig, MechanismRun
from .proposal import proposal_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Budget:
    """What the campaign is allowed to spend, declared before it starts."""

    max_parallel_arms: int = 2
    max_cold_runs: int = 1
    max_wall_clock_s: int = 4 * 3600
    probe_epochs: int = 50
    probe_workers: int = 1
    cold_run_workers: int = 1
    max_probe_seconds: int = 1800
    max_cold_seconds: int = 5400

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_parallel_arms": self.max_parallel_arms,
            "max_cold_runs": self.max_cold_runs,
            "max_wall_clock_s": self.max_wall_clock_s,
            "probe_epochs": self.probe_epochs,
            "probe_workers": self.probe_workers,
            "cold_run_workers": self.cold_run_workers,
        }


@dataclass
class CampaignConfig:
    """Everything the campaign needs to know before it starts."""

    out_dir: Path
    base_tree: Path
    control_tree: Path
    regression_tree: Path
    budget: Budget = field(default_factory=Budget)
    model: str = ""
    angles: Optional[List[str]] = None
    portfolio_size: int = 6
    rounds: int = 2
    cold_run: bool = True
    harness_evolution: bool = True
    fresh: bool = False

    @classmethod
    def load(cls, path: Path) -> "CampaignConfig":
        document = json.loads(Path(path).read_text(encoding="utf-8"))

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else REPO_ROOT / candidate

        budget = Budget(**document.get("budget", {}))
        return cls(
            out_dir=resolve(document["out_dir"]),
            base_tree=resolve(document["base_tree"]),
            control_tree=resolve(document["control_tree"]),
            regression_tree=resolve(document["regression_tree"]),
            budget=budget,
            model=document.get("model", ""),
            angles=document.get("angles"),
            portfolio_size=int(document.get("portfolio_size", 6)),
            rounds=int(document.get("rounds", 2)),
            cold_run=bool(document.get("cold_run", True)),
            harness_evolution=bool(document.get("harness_evolution", True)),
            fresh=bool(document.get("fresh", False)),
        )


class Campaign:
    """One campaign. Every decision it makes is written down with its evidence."""

    def __init__(self, config: CampaignConfig) -> None:
        self.config = config
        self.fresh = bool(config.fresh)
        self.out = Path(config.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.arms_dir = self.out / "arms"
        self.arms_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.state: Dict[str, Any] = {
            "campaign": "harness2-autonomous-research",
            "started_utc": _now(),
            "config": {
                "base_tree": str(config.base_tree),
                "control_tree": str(config.control_tree),
                "regression_tree": str(config.regression_tree),
                "budget": config.budget.as_dict(),
                "model": config.model or opencode.model_name(),
                "portfolio_size": config.portfolio_size,
                "rounds": config.rounds,
            },
            "preflight": {},
            "rounds": [],
            "arms": [],
            "leaderboard": [],
            "harness_evolution": {},
            "errors": [],
        }
        self._write_state()

    # -- state ------------------------------------------------------------
    def _write_state(self) -> None:
        self.state["elapsed_s"] = round(time.time() - self.started, 1)
        (self.out / "campaign_state.json").write_text(
            json.dumps(self.state, indent=2, default=str) + "\n", encoding="utf-8"
        )

    def _budget_left(self) -> float:
        return float(self.config.budget.max_wall_clock_s) - (time.time() - self.started)

    def _record_error(self, where: str, exc: BaseException) -> None:
        self.state["errors"].append(
            {"where": where, "error": f"{type(exc).__name__}: {exc}", "at": _now()}
        )
        self._write_state()

    # -- 0. preflight -----------------------------------------------------
    def preflight(self) -> Dict[str, Any]:
        """Everything the campaign needs before it starts, checked before it starts."""
        report = opencode.preflight(self.config.model, probe=True, timeout=300)
        gpus = _gpus()
        historical = _historical_digest()
        self.state["preflight"] = {
            "agent": report.as_dict(),
            "gpus": gpus,
            "historical_digest": historical,
            "budget": self.config.budget.as_dict(),
            "ok": bool(report.ok and gpus),
        }
        self._write_state()
        if not report.ok:
            raise RuntimeError(
                "the agent backend is not usable here: " + "; ".join(report.problems)
            )
        if not gpus:
            raise RuntimeError("no GPU is visible: the campaign's probes and runs need one")
        return self.state["preflight"]

    # -- 1. ideation ------------------------------------------------------
    def ideate(self, round_index: int, *, evidence: str) -> List[Dict[str, Any]]:
        """Ask the researcher for one mechanism per angle, then critique and select."""
        saved = self.out / f"ideation_round{round_index}.json"
        if saved.is_file() and not self.fresh:
            # Resumed: the proposals are already frozen and critiqued.
            return json.loads(saved.read_text(encoding="utf-8"))
        angles = self.config.angles or list(ideation.ANGLES)
        proposals: List[Dict[str, Any]] = []
        for angle in angles:
            prompt = ideation.prompt_for(
                angle,
                base_tree=str(self.config.base_tree),
                evidence=evidence,
            )
            proposal = None
            record: Dict[str, Any] = {
                "angle": angle,
                "round": round_index,
                # The brief the researcher was given, so the lesson -> design chain is
                # traceable and not a matter of retelling.
                "brief": prompt,
                "attempts": [],
            }
            for attempt in range(1, 3):
                result = opencode.run_agent(
                    self.out / f"ideation_round{round_index}_{angle}_a{attempt}",
                    prompt,
                    model=self.config.model or opencode.model_name(),
                    timeout=1200,
                    title=f"campaign research {angle} (round {round_index})",
                )
                attempt_record = {
                    "attempt": attempt,
                    "exit_code": result.get("exit_code"),
                    "cost": result.get("cost"),
                    "tokens": result.get("tokens"),
                    "text": result.get("text"),
                    "tool_calls": result.get("tool_calls"),
                }
                record["attempts"].append(attempt_record)
                try:
                    proposal = ideation.parse_proposal(result.get("text") or "")
                except ValueError as exc:
                    attempt_record["parse_error"] = str(exc)
                    # The reply did not carry a proposal: say so, and ask again with the
                    # reason, rather than silently moving on.
                    prompt = (
                        prompt
                        + "\n\n# YOUR PREVIOUS REPLY WAS NOT USABLE\n\n"
                        + f"{exc}\n\nReply with ONE fenced ```json block and nothing else."
                    )
                    continue
                verdict = ideation.critique(proposal)
                if verdict.ok:
                    break
                # The proposal is there but does not declare what a proposal must declare.
                # The rubric's verdict goes back to the researcher, which is the only way a
                # proposal becomes checkable.
                attempt_record["critique"] = verdict.as_dict()
                prompt = (
                    prompt
                    + "\n\n# YOUR PREVIOUS PROPOSAL WAS NOT ACCEPTABLE\n\n"
                    + "\n".join(f"- {problem}" for problem in verdict.problems)
                    + "\n\nReply with a corrected ONE fenced ```json block and nothing else."
                )
                proposal = None
            if proposal is not None:
                record["proposal"] = proposal
                record["critique"] = ideation.critique(proposal).as_dict()
            record["exit_code"] = (record["attempts"][-1] if record["attempts"] else {}).get(
                "exit_code"
            )
            record["cost"] = sum(float(a.get("cost") or 0.0) for a in record["attempts"])
            record["tokens"] = (record["attempts"][-1] if record["attempts"] else {}).get("tokens")
            record["text"] = (record["attempts"][-1] if record["attempts"] else {}).get("text")
            record["tool_calls"] = [
                call
                for attempt in record["attempts"]
                for call in attempt.get("tool_calls") or []
            ]
            proposals.append(record)
            self._write_state()
        (self.out / f"ideation_round{round_index}.json").write_text(
            json.dumps(proposals, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return proposals

    # -- 2. one arm --------------------------------------------------------
    def run_arm(self, proposal: Dict[str, Any], *, round_index: int) -> Dict[str, Any]:
        """Implement, judge, repair and probe one mechanism."""
        mechanism_id = str(proposal.get("mechanism_id", "arm"))
        arm_dir = self.arms_dir / mechanism_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_name = f"TIGER_c1_{mechanism_id}"
        saved = arm_dir / "arm.json"
        if saved.is_file() and not self.fresh:
            # Resumed: this arm already ran.
            return json.loads(saved.read_text(encoding="utf-8"))
        # The module and the entry point are a CAMPAIGN CONVENTION, not the researcher's
        # choice: the checker has to find them, and a convention is more reliable than an
        # instruction.
        module_name = f"mechanism_{mechanism_id}"
        entrypoint = "mechanism_loss"

        # the frozen proposal, before anything is implemented
        frozen = _freeze_proposal(dict(proposal))
        (arm_dir / "proposal.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        tree = REPO_ROOT / "stage2_RQ-VAE" / arm_name
        if tree.exists():
            shutil.rmtree(tree, ignore_errors=True)
        _copy_tree(self.config.base_tree, tree)
        # The arms inherited from the r8 line write to results/stage2_RQ-VAE/<ARM NAME>,
        # which is NOT tree-scoped: an arm copied from one of them would write into that
        # arm's results. Every campaign arm is patched to write inside its own tree, so no
        # arm can overwrite another arm's output or a historical directory.
        _scope_outputs_to_tree(tree)

        record: Dict[str, Any] = {
            "mechanism_id": mechanism_id,
            "round": round_index,
            "arm": arm_name,
            "tree": str(tree),
            "proposal_sha256": proposal_digest(frozen),
            "stages": [],
            "started_utc": _now(),
        }

        # 2a. implement, with the real agent
        brief = _implementation_brief(frozen, tree, module_name, entrypoint)
        implementation = opencode.run_agent(
            tree,
            brief,
            model=self.config.model or opencode.model_name(),
            timeout=1800,
            mcp=opencode.mcp_server_block(REPO_ROOT),
            title=f"campaign implement {mechanism_id}",
            deny_paths=[REPO_ROOT / "results", REPO_ROOT / "tools"],
        )
        (arm_dir / "implement.json").write_text(
            json.dumps(implementation, indent=2, default=str) + "\n", encoding="utf-8"
        )
        record["stages"].append(
            {
                "name": "implement",
                "ok": not implementation.get("errors"),
                "exit_code": implementation.get("exit_code"),
                "cost": implementation.get("cost"),
                "tokens": implementation.get("tokens"),
                "tool_calls": len(implementation.get("tool_calls") or []),
            }
        )

        # 2b. judge: the declared invariants, the gradient targets, the static audit.
        # The checks need torch and the campaign needs the framework's interpreter, so
        # they cross the boundary as a subprocess and a JSON report.
        verdict = self._verify(tree, arm_dir, frozen, module_name, entrypoint)
        (arm_dir / "verification.json").write_text(
            json.dumps(verdict, indent=2, default=str) + "\n", encoding="utf-8"
        )
        record["stages"].append(
            {
                "name": "verify",
                "ok": verdict.get("ok"),
                "detail": verdict.get("detail"),
                "failed": verdict.get("failed"),
            }
        )

        # 2c. repair, bounded, with the findings
        attempts = 0
        if not verdict.get("ok"):
            for attempt in range(1, 3):
                attempts = attempt
                repair = opencode.run_agent(
                    tree,
                    _repair_brief(frozen, verdict, attempt),
                    model=self.config.model or opencode.model_name(),
                    timeout=1200,
                    mcp=opencode.mcp_server_block(REPO_ROOT),
                    title=f"campaign repair {mechanism_id} #{attempt}",
                    deny_paths=[REPO_ROOT / "results", REPO_ROOT / "tools"],
                )
                (arm_dir / f"repair_{attempt}.json").write_text(
                    json.dumps(repair, indent=2, default=str) + "\n", encoding="utf-8"
                )
                verdict = self._verify(tree, arm_dir, frozen, module_name, entrypoint)
                (arm_dir / "verification.json").write_text(
                    json.dumps(verdict, indent=2, default=str) + "\n", encoding="utf-8"
                )
                if verdict.get("ok"):
                    break
        record["stages"].append(
            {
                "name": "repair",
                "ok": bool(verdict.get("ok")),
                "attempts": attempts,
                "detail": verdict.get("detail"),
            }
        )
        record["verification"] = verdict

        # 2d. probe: active, matched control, the real trainer
        if verdict.get("ok"):
            probe = self._probe(tree, arm_dir / "probe")
            record["stages"].append({"name": "probe", "ok": probe.get("ok"), **probe.get("summary", {})})
            record["probe"] = probe
        else:
            record["stages"].append({"name": "probe", "ok": False, "skipped": "the mechanism did not verify"})

        # 2e. the structured diagnosis
        record["diagnosis"] = self._diagnose(record)
        record["finished_utc"] = _now()
        (arm_dir / "arm.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return record

    # -- 3. the probe ------------------------------------------------------
    def _verify(
        self,
        tree: Path,
        arm_dir: Path,
        proposal: Dict[str, Any],
        module_name: str,
        entrypoint: str,
    ) -> Dict[str, Any]:
        """Run the declared checks in the environment that has torch."""
        proposal_path = arm_dir / "proposal_for_checks.json"
        proposal_path.write_text(
            json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report_path = arm_dir / "verification.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        try:
            completed = subprocess.run(
                [
                    PROJECT_PYTHON,
                    str(REPO_ROOT / "tools" / "mvg" / "invariant_check.py"),
                    "--tree",
                    str(tree),
                    "--proposal",
                    str(proposal_path),
                    "--module",
                    module_name,
                    "--entrypoint",
                    entrypoint,
                    "--out",
                    str(report_path),
                ],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "detail": "the checks did not finish inside their budget", "checks": []}
        if report_path.is_file():
            return json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "ok": False,
            "detail": "the checks produced no report",
            "returncode": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-400:],
            "checks": [],
        }

    def _probe(self, tree: Path, out_dir: Path) -> Dict[str, Any]:
        """The active, matched-control probe: the real trainer, the mechanism forced on."""
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            PROJECT_PYTHON,
            str(REPO_ROOT / "tools" / "mvg" / "probe_entry.py"),
            "--tree",
            str(tree),
            "--epochs",
            str(self.config.budget.probe_epochs),
            "--out",
            str(out_dir),
            "--baseline-tree",
            str(self.config.control_tree),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        try:
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.budget.max_probe_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "detail": "the probe did not finish inside its budget"}
        report_path = out_dir / "probe_report.json"
        report: Dict[str, Any] = {}
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = {
                "ok": False,
                "detail": "the probe wrote no report",
                "stdout_tail": (completed.stdout or "")[-300:],
            }
        occupancy = report.get("occupancy") or {}
        report["summary"] = {
            "detail": report.get("detail", ""),
            "l1_ratio": report.get("occupancy_ratio_vs_host"),
            "collapse": occupancy.get("collapse_suspected"),
            "unique_l1": occupancy.get("unique_l1"),
            "host_unique_l1": occupancy.get("host_unique_l1"),
        }
        return report

    # -- 4. the diagnosis --------------------------------------------------
    def _diagnose(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Distinguish the four ways an arm can fail, from the evidence."""
        stages = {stage["name"]: stage for stage in record["stages"]}
        verify = stages.get("verify", {})
        probe = stages.get("probe", {})
        if not verify.get("ok"):
            return {
                "category": "(a) incorrect implementation",
                "reason": f"the declared checks failed: {verify.get('failed') or verify.get('detail')}",
                "evidence": "the verification report",
            }
        if not probe.get("ok"):
            occupancy = (probe.get("occupancy") or {})
            if probe.get("collapse") or occupancy.get("collapse_suspected"):
                return {
                    "category": "(b) correct but harmful",
                    "reason": (
                        "the implementation is faithful and the mechanism still destroys the "
                        f"code: {probe.get('unique_l1', occupancy.get('unique_l1'))} level-0 "
                        f"entries against the host's "
                        f"{probe.get('host_unique_l1', occupancy.get('host_unique_l1'))}"
                        f" ({probe.get('l1_ratio')})"
                    ),
                    "evidence": "the probe's occupancy report",
                }
            if probe.get("skipped"):
                return {
                    "category": "(c) infrastructure defect",
                    "reason": "the probe never ran",
                    "evidence": "the arm record",
                }
            return {
                "category": "(d) uncertain or noisy",
                "reason": probe.get("detail", "the probe was inconclusive"),
                "evidence": "the probe report",
            }
        return {
            "category": "qualified on train-only signals",
            "reason": "the implementation is faithful and the probe shows no collapse",
            "evidence": "the verification and probe reports",
        }

    # -- 5. the regression control -----------------------------------------
    def regression_control(self) -> Dict[str, Any]:
        """The known collapse arm must be rejected by the active probe.

        This is the campaign's own regression test: a probe that cannot reject the arm we
        already know collapses cannot be trusted to qualify anything.
        """
        tree = self.config.regression_tree
        saved = self.out / "regression_control.json"
        if saved.is_file() and not self.fresh:
            return json.loads(saved.read_text(encoding="utf-8"))
        if not Path(tree).is_dir():
            return {
                "ok": False,
                "detail": f"the known collapse arm is not here: {tree}",
                "rejected": False,
            }
        probe = self._probe(Path(tree), self.out / "regression_probe")
        occupancy = probe.get("occupancy") or {}
        rejected = bool(occupancy.get("collapse_suspected")) or not probe.get("ok")
        record = {
            "ok": True,
            "tree": str(tree),
            "rejected": rejected,
            "detail": probe.get("detail"),
            "l1_ratio": probe.get("occupancy_ratio_vs_host"),
            "unique_l1": occupancy.get("unique_l1"),
            "host_unique_l1": occupancy.get("host_unique_l1"),
            "report": str(self.out / "regression_probe" / "probe_report.json"),
        }
        (self.out / "regression_control.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return record

    # -- 6. the cold run ---------------------------------------------------
    def cold_run(self, arm: Dict[str, Any]) -> Dict[str, Any]:
        """The cold 3000-epoch run, launched through the controller's admission path."""
        if arm.get("cold_run") and not self.fresh:
            return arm["cold_run"]
        tree = Path(arm["tree"])
        admission = self.arms_dir / arm["mechanism_id"] / "admission.json"
        if not admission.is_file():
            admission.parent.mkdir(parents=True, exist_ok=True)
            admission.write_text(
                json.dumps(
                    {
                        "admission_version": 1,
                        "issued_utc": _now(),
                        "mechanism_id": arm["mechanism_id"],
                        "proposal_sha256": arm["proposal_sha256"],
                        "tree_digest": _tree_digest(tree),
                        "tree_path": str(tree),
                        "judges": ["declared_invariants", "gradient_targets", "static_audit"],
                        "judge_verdict": "passed",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "genrec_mcp_for_campaign", REPO_ROOT / "tools" / "genrec-mcp" / "server.py"
        )
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        # The launch's smoke probe needs real trained weights to probe the gradient path
        # with. A parent arm's checkpoint of the same architecture is copied in and marked
        # smoke-only; the launch archives it before the cold run, so the run itself still
        # starts from scratch.
        result_dir = Path(str(server.tree_result_dir(tree)))
        probe_checkpoint = result_dir / "rqvae_best.pth"
        seeded = None
        if not probe_checkpoint.is_file():
            for parent in ("TIGER_r8_control", "TIGER_BEHAVIOR_SAMPLE_RQ-VAE"):
                source = Path(str(server.tree_result_dir(
                    REPO_ROOT / "stage2_RQ-VAE" / parent
                ))) / "rqvae_best.pth"
                if source.is_file():
                    result_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, probe_checkpoint)
                    seeded = {"source": str(source), "target": str(probe_checkpoint)}
                    break
        result = server.tool_stage2_start(source_dir=str(tree), admission=str(admission))
        # The launch path returns as soon as a detached worker owns the rest of the
        # sequence, so "started" is False on a DISPATCHED run: the truth is the status.
        status = str(result.get("status", ""))
        dispatched = status in ("smoke_pending", "starting", "running") or bool(
            result.get("worker_pid")
        )
        record = {
            "mechanism_id": arm["mechanism_id"],
            "started": dispatched,
            "refused": result.get("refused"),
            "status_at_dispatch": status,
            "probe_checkpoint": seeded,
            "launch": result,
        }
        if not dispatched:
            refusal = str(result.get("refused") or "")
            state = result.get("state") or {}
            if "already live" in refusal and state.get("status") in (
                "running",
                "starting",
                "smoke_pending",
            ):
                # The run for this arm is already going: adopt it rather than starting a
                # second one, and wait for it.
                record["adopted"] = state
                final = self._await_run(server, tree)
                record["final_status"] = final.get("status")
                record["finished"] = bool(record["final_status"] in ("finished", "idle"))
                self._evaluate_arm(arm, server, record)
                return record
            return record
        deadline = time.time() + self.config.budget.max_cold_seconds
        final: Dict[str, Any] = {}
        while time.time() < deadline:
            final = server.tool_stage2_status(source_dir=str(tree))
            if str(final.get("status", "")) in ("finished", "idle", "refused", "worker_gone"):
                break
            time.sleep(60)
        record["final_status"] = final.get("status")
        record["finished"] = bool(record["final_status"] in ("finished", "idle"))
        self._evaluate_arm(arm, server, record)
        return record

    def _await_run(self, server, tree: Path) -> Dict[str, Any]:
        deadline = time.time() + self.config.budget.max_cold_seconds
        final: Dict[str, Any] = {}
        while time.time() < deadline:
            final = server.tool_stage2_status(source_dir=str(tree))
            if str(final.get("status", "")) in ("finished", "idle", "refused", "worker_gone"):
                break
            time.sleep(60)
        return final

    def _evaluate_arm(self, arm: Dict[str, Any], server, record: Dict[str, Any]) -> None:
        """The frozen evaluation, once per admitted arm."""
        tree = Path(arm["tree"])
        result_dir = Path(str(server.tree_result_dir(tree)))
        candidates = sorted(result_dir.rglob("item_sids.json"))
        if not candidates:
            record["evaluation"] = {"ok": False, "error": f"no exported SIDs under {result_dir}"}
            return
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        completed = subprocess.run(
            [
                PROJECT_PYTHON,
                str(REPO_ROOT / "tools" / "mvg" / "er_eval.py"),
                "--sid",
                str(candidates[-1]),
                "--out",
                str(self.arms_dir / arm["mechanism_id"] / "er_eval.json"),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        try:
            record["evaluation"] = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            record["evaluation"] = {"ok": False, "error": "the evaluation produced no report"}

    # -- 7. the report -----------------------------------------------------
    def report(self) -> Dict[str, Any]:
        """The machine-readable leaderboard and the human-readable report."""
        # The diagnoses are recomputed from the arms' own evidence, so a report can never
        # carry a diagnosis that the evidence does not support.
        for arm in self.state.get("arms", []):
            if arm.get("stages"):
                arm["diagnosis"] = self._diagnose(arm)
        rows: List[Dict[str, Any]] = []
        for arm in self.state["arms"]:
            probe = arm.get("probe") or {}
            occupancy = probe.get("occupancy") or {}
            cold = arm.get("cold_run") or {}
            evaluation = cold.get("evaluation") or {}
            rows.append(
                {
                    "mechanism_id": arm["mechanism_id"],
                    "round": arm["round"],
                    "arm": arm["arm"],
                    "equation": arm.get("equation", ""),
                    "proposal_sha256": arm["proposal_sha256"],
                    "implemented": arm.get("implemented", False),
                    "verified": bool((arm.get("verification") or {}).get("ok")),
                    "verification_failures": (arm.get("verification") or {}).get("failed", []),
                    "probe_ok": bool(probe.get("ok")),
                    "l1_ratio_vs_control": probe.get("occupancy_ratio_vs_host"),
                    "unique_l1": occupancy.get("unique_l1"),
                    "host_unique_l1": occupancy.get("host_unique_l1"),
                    "diagnosis": (arm.get("diagnosis") or {}).get("category", ""),
                    "reason": (arm.get("diagnosis") or {}).get("reason", ""),
                    "cold_run": bool(cold.get("started")),
                    "cold_finished": bool(cold.get("finished")),
                    "l1_er": evaluation.get("l1_er"),
                    "frozen_threshold": evaluation.get("frozen_threshold"),
                    "beats_frozen_gate": evaluation.get("above_frozen_threshold"),
                    "cost_usd": arm.get("cost_usd", 0.0),
                }
            )
        self.state["leaderboard"] = rows
        (self.out / "leaderboard.json").write_text(
            json.dumps(
                {
                    "campaign": self.state["campaign"],
                    "generated_utc": _now(),
                    "budget": self.config.budget.as_dict(),
                    "control": _control_row(),
                    "arms": rows,
                    "errors": self.state["errors"],
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8"
        )
        (self.out / "report.md").write_text(_markdown(self.state, rows), encoding="utf-8")
        self._write_state()
        return {"leaderboard": rows, "report": str(self.out / "report.md")}


# --------------------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------------------
def _arm_qualified(arm: Dict[str, Any]) -> bool:
    """An arm qualifies on TRAIN-ONLY signals: it verifies and its probe does not reject it."""
    stages = {stage["name"]: stage for stage in arm.get("stages", [])}
    if not stages.get("verify", {}).get("ok"):
        return False
    probe = stages.get("probe", {})
    return bool(probe.get("ok"))


def _run_arms(
    campaign: "Campaign", portfolio: Sequence[Dict[str, Any]], round_index: int, workers: int
) -> List[Dict[str, Any]]:
    """Implement, judge, repair and probe the portfolio, within the declared parallelism."""
    if not portfolio:
        return []
    arms: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(campaign.run_arm, proposal, round_index=round_index): proposal
            for proposal in portfolio
        }
        for future in futures:
            try:
                arms.append(future.result())
            except Exception as exc:  # noqa: BLE001
                campaign._record_error("run_arm", exc)
                arms.append({"mechanism_id": "unknown", "stages": [], "error": str(exc)})
    return arms


def _evidence_block(arms: Sequence[Dict[str, Any]] = ()) -> str:
    """The evidence the researcher is given: the prior record, and the round's own results."""
    control = _control_row()
    lines = [
        "## The prior programme (R8, issues #91-#100)",
        "",
        "| lane | mechanism | Hard-L1 ER |",
        "|---|---|---|",
    ]
    for lane, name in ideation.PRIOR_FAILURES.items():
        lines.append(f"| {lane} | {name} | (all failed; see the synthesis) |")
    lines += [
        "",
        f"the fresh cold control scored {control.get('l1_er')} and the frozen gate is "
        "16.38953942232631, so the gate sits inside the host's own run-to-run band: the "
        "comparison that carries information is against a FRESH COLD CONTROL.",
        "",
        "## The lessons",
        "",
    ]
    lines += [f"- {lesson}" for lesson in ideation.PRIOR_LESSONS]
    if arms:
        lines += ["", "## This campaign's first round", ""]
        for arm in arms:
            diagnosis = arm.get("diagnosis") or {}
            probe = arm.get("probe") or {}
            occupancy = probe.get("occupancy") or {}
            lines.append(
                f"- {arm.get('mechanism_id')}: {diagnosis.get('category')} -- "
                f"{diagnosis.get('reason')} "
                f"(level-0 entries {occupancy.get('unique_l1')} against the host's "
                f"{occupancy.get('host_unique_l1')})"
            )
    return "\n".join(lines)


def evaluate_harness(campaign: "Campaign") -> Dict[str, Any]:
    """Does a harness change help on the HELD-BACK tasks? Adopted only if it demonstrably does.

    The harness is the agent's instructions, skills and probe strategy. Improving it is a
    different thing from improving a mechanism, and the two are measured separately: this
    comparison is old harness against new harness on tasks the harness was never edited
    against.
    """
    workspace = campaign.out / "harness_candidate"
    saved = campaign.out / "harness_evolution.json"
    if saved.is_file() and not campaign.fresh:
        return json.loads(saved.read_text(encoding="utf-8"))
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("CLAUDE.md",):
        source = REPO_ROOT / name
        if source.is_file():
            shutil.copy2(source, workspace / name)
    skills = REPO_ROOT / ".agent" / "skills"
    if skills.is_dir():
        shutil.copytree(skills, workspace / ".agent" / "skills", dirs_exist_ok=True)
    scripts = REPO_ROOT / "scripts"
    if scripts.is_dir():
        (workspace / "scripts").mkdir(parents=True, exist_ok=True)
        for script in scripts.glob("*.py"):
            shutil.copy2(script, workspace / "scripts" / script.name)

    lessons = _evidence_block(campaign.state.get("arms", []))
    prompt = f"""# THE HARNESS IS THE AGENT'S INSTRUCTIONS

This directory is the harness: CLAUDE.md, the skills under .agent/skills/, and the helper
scripts. The agents that work in this repository read it.

# WHAT THE CAMPAIGN LEARNED

{lessons}

# THE TASK

Propose ONE change to this harness that would have prevented the failures above, or that
would have made them obvious sooner. The change must be a file edit: a rule in CLAUDE.md, a
new skill under .agent/skills/<name>/SKILL.md with frontmatter, or a helper script under
scripts/.

Do not change the judges, the frozen specifications, the thresholds or the held-out
fixtures: they are not in this directory, and a harness that edits the judge is not an
improvement. Do not start a training run.

Make the edit, then report in one short paragraph what you changed and which failure it
addresses.
"""
    result = opencode.run_agent(
        workspace,
        prompt,
        model=campaign.config.model or opencode.model_name(),
        timeout=1200,
        title="campaign harness evolution",
    )
    (campaign.out / "harness_candidate_session.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )

    def sweep(harness_root: Path, out_dir: Path) -> Dict[str, Any]:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.harness2_genrec.cli",
                "sweep",
                "--k",
                "1",
                "--modes",
                "parallel",
                "--split",
                "test",
                "--base-harness",
                str(harness_root),
                "--runs-dir",
                str(out_dir),
                "--timeout",
                "600",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        record = out_dir / "sweep.json"
        summary: Dict[str, Any] = {}
        if record.is_file():
            summary = json.loads(record.read_text(encoding="utf-8"))
        done = sum(int(d.get("done", 0)) for d in summary.get("domains", []))
        failed = sum(int(d.get("failed", 0)) for d in summary.get("domains", []))
        return {
            "harness": str(harness_root),
            "done": done,
            "failed": failed,
            "returncode": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-300:],
        }

    prior = sweep(REPO_ROOT, campaign.out / "harness_sweep_prior")
    candidate = sweep(workspace, campaign.out / "harness_sweep_candidate")
    improved = candidate["done"] > prior["done"]
    record = {
        "prior": prior,
        "candidate": candidate,
        "improved": improved,
        "adopted": improved,
        "reason": (
            "the candidate harness solved more held-back tasks than the prior one"
            if improved
            else "the candidate harness did not solve more held-back tasks; it is NOT adopted"
        ),
        "session": {
            "exit_code": result.get("exit_code"),
            "cost": result.get("cost"),
            "tokens": result.get("tokens"),
            "text": result.get("text"),
        },
    }
    (campaign.out / "harness_evolution.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


def execute(config: CampaignConfig) -> Dict[str, Any]:
    """Run the whole campaign, in the issue's order, inside the declared budget."""
    campaign = Campaign(config)
    campaign.preflight()

    proposals = campaign.ideate(1, evidence=_evidence_block())
    portfolio = ideation.select_portfolio(
        [record["proposal"] for record in proposals if record.get("proposal")],
        size=config.portfolio_size,
    )
    campaign.state["round1_proposals"] = len(proposals)
    campaign.state["round1_portfolio"] = [p.get("mechanism_id") for p in portfolio]
    arms = _run_arms(campaign, portfolio, 1, config.budget.max_parallel_arms)
    campaign.state["arms"] = arms
    campaign.state["regression_control"] = campaign.regression_control()

    if config.rounds >= 2 and arms:
        proposals2 = campaign.ideate(2, evidence=_evidence_block(arms))
        taken = {str(arm.get("mechanism_id")) for arm in arms}
        portfolio2 = [
            proposal
            for proposal in ideation.select_portfolio(
                [record["proposal"] for record in proposals2 if record.get("proposal")],
                size=max(1, config.portfolio_size // 3),
            )
            # A second round that re-proposes a first-round arm under the same name is not a
            # redesign: it would overwrite the first round's evidence.
            if str(proposal.get("mechanism_id")) not in taken
        ]
        campaign.state["round2_proposals"] = len(proposals2)
        campaign.state["round2_portfolio"] = [p.get("mechanism_id") for p in portfolio2]
        arms2 = _run_arms(campaign, portfolio2, 2, config.budget.max_parallel_arms)
        campaign.state["arms"] = arms + arms2

    if config.harness_evolution:
        campaign.state["harness_evolution"] = evaluate_harness(campaign)

    qualified = [arm for arm in campaign.state["arms"] if _arm_qualified(arm)]
    campaign.state["qualified"] = [arm.get("mechanism_id") for arm in qualified]
    if config.cold_run:
        for arm in qualified[: config.budget.max_cold_runs]:
            if campaign._budget_left() <= 0:
                campaign._record_error("cold_run", RuntimeError("the wall-clock budget is spent"))
                break
            arm["cold_run"] = campaign.cold_run(arm)
            campaign._write_state()

    campaign.report()
    return campaign.state


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _freeze_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Freeze a campaign proposal: the content, its digest, and who wrote it.

    The campaign's proposals are not the lifecycle's proposals: they carry the checker's
    contract (the inputs, the invariants, the differentiation) rather than the lifecycle's
    fields, so they are frozen against their own schema.
    """
    frozen = dict(proposal)
    frozen["schema_version"] = 1
    provenance = dict(frozen.get("provenance") or {})
    provenance.setdefault("authored_by", "OpenCode, the campaign's researcher")
    provenance.setdefault("authored_at", _now())
    provenance["frozen_digest"] = proposal_digest(
        {key: value for key, value in frozen.items() if key != "provenance"}
    )
    frozen["provenance"] = provenance
    return frozen


def _scope_outputs_to_tree(tree: Path) -> None:
    """Point an arm's outputs at its own tree.

    The r8-line trainers derive ``_OUT`` from the arm's name alone, so two arms with the
    same name share a result directory. The campaign's arms must not: each writes under
    ``results/stage2_RQ-VAE/<TREE_NAME>/<ARM>``, which is what the tree's own comment has
    always claimed and what the launch path resolves.
    """
    trainer = Path(tree) / "train_rqvae.py"
    if not trainer.is_file():
        return
    source = trainer.read_text(encoding="utf-8")
    if "_OUT = OUT_ROOT / TREE_NAME / EXPERIMENT_ARM" in source:
        return
    patched = source.replace(
        "_OUT = OUT_ROOT / EXPERIMENT_ARM",
        "_OUT = OUT_ROOT / TREE_NAME / EXPERIMENT_ARM",
        1,
    )
    if patched != source:
        trainer.write_text(patched, encoding="utf-8")


def _copy_tree(source: Path, destination: Path) -> None:
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


def _gpus() -> List[Dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        if line.strip():
            rows.append({"line": line.strip()})
    return rows


def _historical_digest() -> Dict[str, Any]:
    """The digest of the historical r8 artefacts, so the campaign can prove it left them alone."""
    import hashlib

    provenance = REPO_ROOT / "results" / "stage2_RQ-VAE" / "r8_ten_mechanisms" / "provenance.json"
    if not provenance.is_file():
        return {"available": False}
    document = json.loads(provenance.read_text(encoding="utf-8"))
    matched = 0
    changed = 0
    missing = 0
    for record in document.values():
        for key in ("item_sids_sha256", "sids_npy_sha256", "checkpoint_sha256", "trainer_sha256"):
            recorded = record.get(key)
            if not recorded:
                continue
            path = _provenance_path(record, key)
            if path is None or not Path(path).is_file():
                missing += 1
                continue
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest == recorded:
                matched += 1
            else:
                changed += 1
    return {
        "available": True,
        "matched": matched,
        "changed": changed,
        "missing": missing,
    }


def _provenance_path(record: Dict[str, Any], key: str) -> Optional[str]:
    if key == "checkpoint_sha256":
        return record.get("checkpoint_path")
    if key == "trainer_sha256":
        tree = record.get("tree")
        return str(REPO_ROOT / tree / "train_rqvae.py") if tree else None
    tree = record.get("tree")
    if not tree:
        return None
    name = Path(tree).name.replace("TIGER_", "")
    if key == "item_sids_sha256":
        return str(REPO_ROOT / "results" / "stage2_RQ-VAE" / name / "item_sids.json")
    return str(REPO_ROOT / "results" / "stage2_RQ-VAE" / name / "sids_for_hgrec.npy")


def _tree_digest(tree: Path) -> str:
    import importlib.util

    path = REPO_ROOT / "tools" / "cgv" / "contract.py"
    spec = importlib.util.spec_from_file_location("cgv_contract_for_campaign", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest, _count = module.code_tree_sha256(Path(tree))
    return digest or ""


def _control_row() -> Dict[str, Any]:
    """The fresh cold control's numbers, read from the r8 leaderboard."""
    path = REPO_ROOT / "results" / "stage2_RQ-VAE" / "r8_ten_mechanisms" / "leaderboard.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    for lane in document.get("lanes", []):
        if lane.get("lane") == "r8_control":
            return {
                "lane": "r8_control",
                "l1_er": lane.get("hard_l1_er"),
                "checkpoint": lane.get("checkpoint"),
                "note": "the untouched host, re-run cold under the same protocol",
            }
    return {}


def _implementation_brief(
    proposal: Dict[str, Any], tree: Path, module_name: str, entrypoint: str
) -> str:
    """The implementer's brief. The proposal is frozen; the code is not written yet."""
    return f"""# TASK

Implement ONE Stage-2 mechanism in this arm tree, from a proposal that was frozen before
you started. The proposal is below. It is the contract: implement THAT mechanism, not a
different one.

# THE PROPOSAL

```json
{json.dumps(proposal, indent=2, sort_keys=True)}
```

# WHERE TO WORK

This tree is a scratch copy of an experiment arm. Implement the mechanism here: a module of
your own inside the tree, and the call site in `train_rqvae.py` that composes its term into
the host's loss BEFORE backward, so it reaches the optimizer through the ordinary path.

The host's own loss graph, the codebook sizes, the KMeans init, the seed, the 3000-epoch
budget and the optimizer are NOT yours to change. The mechanism's own weight is a
hardcoded constant of the tree, like the other constants.

# THE NAMES ARE FIXED

Put the mechanism in a module named `{module_name}.py` in the root of this tree, and name
its entry point `{entrypoint}`. The checker looks for exactly those names. The entry point
must be a module-level function with this signature:

```python
def {entrypoint}(source, target, codebook, mask, **parameters):
    ...
```

where `source` is [B, D], `target` is [B, D], `codebook` is [C, D], `mask` is [B] with 1.0
for a valid pair, and `parameters` are the proposal's parameters. The module must import on
its own, with no dependency on the tree.

# HOW IT WILL BE CHECKED

An independent checker, which does not know your internals, will:

* call your loss as `{entrypoint}(source, target, codebook, mask, **parameters)` on a small
  synthetic case built from the proposal's `inputs`;
* permute the codebook rows and require the loss to be unchanged;
* exchange the source and the target and require the loss to change;
* mask every pair and require the loss to be zero;
* backpropagate and require the gradient to reach the source residual, the target residual
  and the codebook exactly as the proposal declares;
* read your source and refuse the substitutions the proposal declares.

# WHAT NOT TO TOUCH

The judges, the frozen specifications, the thresholds, the held-out fixtures and the
historical artefacts are not in your tree and must not be reached for. Do not start a
training run: this stage is the implementation only. Report what you changed and why.
"""


def _repair_brief(proposal: Dict[str, Any], verdict: Dict[str, Any], attempt: int) -> str:
    """The repairer's brief: the findings, and nothing else."""
    findings = "\n".join(
        f"- {check.get('id')}: {check.get('detail')}"
        for check in (verdict.get("checks") or [])
        if not check.get("ok")
    )
    return f"""# THE FINDINGS

An independent checker rejected your implementation (attempt {attempt}). The findings are
below. The checker is not yours to change, and neither are the fixtures or the thresholds.

{findings or '- (no detail recorded)'}

# WHAT TO DO

Diagnose each finding against the code in this tree, then fix the code. A fix that hides
the symptom rather than the cause will be caught by the same check. The proposal below is
the contract you are implementing against:

```json
{json.dumps(proposal, indent=2, sort_keys=True)}
```

Report what you changed and why.
"""


def _markdown(state: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> str:
    """The human-readable report."""
    lines = [
        "# Autonomous research campaign -- report",
        "",
        f"- started: {state.get('started_utc')}",
        f"- elapsed: {state.get('elapsed_s')} s",
        f"- budget: {json.dumps(state.get('config', {}).get('budget', {}))}",
        f"- agent: {state.get('config', {}).get('model')}",
        "",
        "## Every arm",
        "",
        "| arm | round | verified | probe | L1 vs control | cold run | L1 ER | diagnosis |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {mechanism_id} | {round} | {verified} | {probe_ok} | {l1_ratio} | {cold_run} | "
            "{l1_er} | {diagnosis} |".format(
                mechanism_id=row.get("mechanism_id"),
                round=row.get("round"),
                verified="yes" if row.get("verified") else "no",
                probe_ok="ok" if row.get("probe_ok") else ("rejected" if row.get("l1_ratio_vs_control") else "-"),
                l1_ratio=row.get("l1_ratio_vs_control"),
                cold_run="yes" if row.get("cold_finished") else ("started" if row.get("cold_run") else "-"),
                l1_er=row.get("l1_er"),
                diagnosis=row.get("diagnosis"),
            )
        )
    lines += [
        "",
        "## The control",
        "",
        f"```json\n{json.dumps(_control_row(), indent=2)}\n```",
        "",
        "## The regression control",
        "",
        f"```json\n{json.dumps(state.get('regression_control', {}), indent=2)}\n```",
        "",
        "## The errors",
        "",
        f"```json\n{json.dumps(state.get('errors', []), indent=2)}\n```",
        "",
        "## The harness",
        "",
        f"```json\n{json.dumps(state.get('harness_evolution', {}), indent=2)}\n```",
        "",
    ]
    return "\n".join(lines)


__all__ = ["Budget", "Campaign", "CampaignConfig"]
