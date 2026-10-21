"""One candidate's route through the pipeline (issue #106, workstream C).

    frozen proposal -> implementation -> unchanged independent checks -> active probe and
    export sensitivity -> one cold 3000-epoch Stage-2 -> the frozen ER once

Every rejection is recorded with its evidence, and the cold run happens only if the probe
and the export sensitivity admit the arm. The guardrails are consulted before the launch:
the seed is the project's frozen one, the control is never re-trained, and an arm that
already failed is never run again.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence import EvidenceStore, now_utc

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The base tree every candidate starts from: the frozen host.
BASE_TREE = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_r8_control"

#: The archived control the comparison is made against. Its artifacts are read-only.
CONTROL_SID = REPO_ROOT / "results" / "stage2_RQ-VAE" / "r8_control" / "sids_for_hgrec.npy"


def _judge(tree: Path, proposal_path: Path, mechanism_id: str, out: Path) -> Dict[str, Any]:
    import subprocess

    completed = subprocess.run(
        [
            "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9",
            str(REPO_ROOT / "tools" / "mvg" / "invariant_check.py"),
            "--tree",
            str(tree),
            "--proposal",
            str(proposal_path),
            "--module",
            f"mechanism_{mechanism_id}",
            "--entrypoint",
            "mechanism_loss",
            "--out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))
    return {"ok": False, "detail": f"the judge produced no report: {completed.stderr[-300:]}"}


def _probe(tree: Path, out_dir: Path) -> Dict[str, Any]:
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9",
            str(REPO_ROOT / "tools" / "mvg" / "probe_entry.py"),
            "--tree",
            str(tree),
            "--epochs",
            "50",
            "--baseline-tree",
            str(REPO_ROOT / "stage2_RQ-VAE" / "TIGER_r8_control"),
            "--out",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=3600,
    )
    report = out_dir / "probe_report.json"
    if report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))
    return {"ok": False, "detail": completed.stderr[-400:]}


def _export_sensitivity(arm_sid: Path, control_sid: Path) -> Dict[str, Any]:
    import subprocess

    if not arm_sid.is_file() or not control_sid.is_file():
        return {}
    completed = subprocess.run(
        [
            "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9",
            str(REPO_ROOT / "tools" / "mvg" / "export_sensitivity.py"),
            "--arm-sid",
            str(arm_sid),
            "--control-sid",
            str(control_sid),
        ],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr[-300:]}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": "the export-sensitivity output is not JSON"}


def run_candidate(
    proposal_path: Path,
    out_dir: Path,
    *,
    store: Optional[EvidenceStore] = None,
    model: str = "",
    timeout: int = 2400,
    cold: bool = True,
    port: int = 50231,
    tree_prefix: str = "TIGER_c107",
) -> Dict[str, Any]:
    """The whole route for one candidate."""
    from . import guardrails, mini_campaign, redesign

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal = json.loads(Path(proposal_path).read_text(encoding="utf-8"))
    mechanism_id = proposal["mechanism_id"]
    started = time.time()
    record: Dict[str, Any] = {
        "mechanism_id": mechanism_id,
        "proposal_path": str(proposal_path),
        "started_utc": now_utc(),
        "stages": {},
    }

    # 1. the implementation, by the agent
    tree = REPO_ROOT / "stage2_RQ-VAE" / f"{tree_prefix}_{mechanism_id}"
    if tree.is_dir():
        shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(BASE_TREE, tree)
    implementation = redesign.implement(proposal, tree, out_dir, model=model, timeout=timeout)
    record["stages"]["implement"] = {
        "tree": str(tree),
        "tokens": implementation.get("tokens"),
        "exit_code": implementation.get("exit_code"),
    }

    # 2. the unchanged independent checks
    verification = _judge(
        tree, Path(proposal_path), mechanism_id, out_dir / "verification.json"
    )
    record["stages"]["judges"] = {
        "ok": verification.get("ok"),
        "detail": verification.get("detail"),
        "failed": verification.get("failed"),
    }
    if not verification.get("ok"):
        record["status"] = "rejected"
        record["reason"] = "the independent checks rejected the implementation"
        record["wall_clock_s"] = round(time.time() - started, 1)
        record["cold_run_gpu_minutes"] = 0.0
        (out_dir / "candidate.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return record

    # 3. the active probe and the export sensitivity against the matched control
    probe = _probe(tree, out_dir / "probe")
    record["stages"]["probe"] = {
        "ok": probe.get("ok"),
        "detail": probe.get("detail"),
        "occupancy_ratio_vs_host": probe.get("occupancy_ratio_vs_host"),
    }
    arm_export = next(iter(out_dir.glob("probe/redirected/**/sids_for_hgrec.npy")), None)
    control_export = next(iter(out_dir.glob("probe/baseline/redirected/**/sids_for_hgrec.npy")), None)
    sensitivity = (
        _export_sensitivity(arm_export, control_export) if arm_export and control_export else {}
    )
    record["stages"]["export_sensitivity"] = {
        "verdict": sensitivity.get("verdict"),
        "changed_fraction": sensitivity.get("changed_fraction"),
        "level0_occupancy_ratio_vs_control": sensitivity.get("level0_occupancy_ratio_vs_control"),
    }
    admitted = bool(probe.get("ok")) and sensitivity.get("verdict") not in ("collapsed",)
    record["admitted"] = admitted
    if not admitted:
        record["status"] = "rejected"
        record["reason"] = (
            "the probe or the export sensitivity rejected the arm before a cold run; the "
            "candidate cost no cold-run GPU time"
        )
        record["wall_clock_s"] = round(time.time() - started, 1)
        record["cold_run_gpu_minutes"] = 0.0
        (out_dir / "candidate.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return record

    # 4. the guardrails, before anything is launched
    intent = guardrails.ExecutionIntent(
        kind="stage2", mechanism_id=mechanism_id, seed=42, policy="current", tree=tree
    )
    verdict = guardrails.check_intent(intent, store)
    record["stages"]["guardrails"] = verdict.as_dict()
    if not verdict.allowed:
        record["status"] = "refused"
        record["reason"] = f"[{verdict.rule}] {verdict.detail}"
        record["cold_run_gpu_minutes"] = 0.0
        (out_dir / "candidate.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return record

    # 5. the cold run
    parameters = {
        str(k): float(v) for k, v in (proposal.get("parameters") or {}).items()
    }
    weight = float((proposal.get("activation") or {}).get("weight", 0.05))
    mini_campaign._write_hooks(tree, mechanism_id, parameters, weight)
    mini_campaign._write_spec(out_dir / "spec.json", mechanism_id, parameters, weight)
    trainer = tree / "train_rqvae.py"
    trainer.write_text(
        re.sub(r'"master_port":\s*\d+', f'"master_port": {port}', trainer.read_text(), count=1),
        encoding="utf-8",
    )
    if not cold:
        record["status"] = "admitted"
        record["wall_clock_s"] = round(time.time() - started, 1)
        record["cold_run_gpu_minutes"] = 0.0
        (out_dir / "candidate.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return record

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "genrec_mcp_server", REPO_ROOT / "tools" / "genrec-mcp" / "server.py"
    )
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    launch = server.tool_stage2_start(
        source_dir=str(tree), spec=json.loads((out_dir / "spec.json").read_text())
    )
    record["stages"]["cold_launch"] = {
        "started": launch.get("started"),
        "status": launch.get("status"),
        "refused": launch.get("refused"),
        "cgv_gate": launch.get("cgv_gate"),
        "result_dir": launch.get("result_dir"),
    }
    record["status"] = "cold_running" if launch.get("status") in ("smoke_pending", "running") else "failed"
    record["result_dir"] = launch.get("result_dir")
    record["wall_clock_s"] = round(time.time() - started, 1)
    record["cold_run_gpu_minutes"] = None
    (out_dir / "candidate.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


__all__ = ["BASE_TREE", "CONTROL_SID", "run_candidate"]
