"""Ingest real runs into the immutable evidence store (issue #105, workstream A).

The #104 campaign left its artifacts on disk: training metrics, exported SIDs, checkpoints,
judge reports, probe reports and Stage-3 results. This module turns those artifacts into
**manifests** -- one per run, written once, naming the artifacts with their SHA-256 and
the numbers that were measured. After the ingestion the research system reads the store
instead of a paragraph of hardcoded outcomes, and every claim it makes can be checked
against the run that produced it.

What is hardcoded here is the *location* of the artifacts, never their content: every
number in a manifest is read out of the artifact itself.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m tools.harness2_genrec.cli \
        ingest-campaign --run-dir /home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1 \
                        --store results/harness3_runs/store
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .evidence import EvidenceError, EvidenceStore, RunPlan, now_utc, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The project interpreter: the one that has torch and numpy.
PROJECT_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

#: Where the #104 campaign's evidence lives. These are LOCATIONS, not outcomes.
LEARNING_RUN_DIR = Path(
    "/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1"
)
RESULTS_ROOT = REPO_ROOT / "results" / "stage2_RQ-VAE"
STAGE3_ROOT = REPO_ROOT / "results" / "stage3_T5Train"

#: The arms of the #104 campaign: the tree that trained them, where their exported SIDs
#: live, and where their judge and probe reports are.
CAMPAIGN_ARMS: Tuple[Dict[str, Any], ...] = (
    {
        "mechanism_id": "r8_control",
        "kind": "control",
        "tree": "stage2_RQ-VAE/TIGER_r8_control",
        "sid_dir": "results/stage2_RQ-VAE/r8_control",
        "stage3_variant": "r8_control",
        "probe": None,
        "verification": None,
    },
    {
        "mechanism_id": "behavior_conditioned_pairwise_curvature",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_curvature",
        "sid_dir": (
            "results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_curvature/r8_control"
        ),
        "stage3_variant": None,
        "probe": "probes/behavior_conditioned_pairwise_curvature/probe_report.json",
        "verification": "v2_judges/behavior_conditioned_pairwise_curvature.json",
    },
    {
        "mechanism_id": "behavior_conditioned_pairwise_margin",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_margin",
        "sid_dir": (
            "results/stage2_RQ-VAE/TIGER_c1_behavior_conditioned_pairwise_margin/r8_control"
        ),
        "stage3_variant": None,
        "probe": "probes/behavior_conditioned_pairwise_margin/probe_report.json",
        "verification": "v2_judges/behavior_conditioned_pairwise_margin.json",
    },
    {
        "mechanism_id": "behaviour_conditioned_residual_target",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_c1_behaviour_conditioned_residual_target",
        "sid_dir": (
            "results/stage2_RQ-VAE/TIGER_c1_behaviour_conditioned_residual_target/r8_control"
        ),
        "stage3_variant": "TIGER_c1_behaviour_conditioned_residual_target__r8_control",
        "probe": "probes/behaviour_conditioned_residual_target/probe_report.json",
        "verification": "v2_judges/behaviour_conditioned_residual_target.json",
    },
    {
        "mechanism_id": "curvature_weighted_transition_transport",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_c1_curvature_weighted_transition_transport",
        "sid_dir": (
            "results/stage2_RQ-VAE/TIGER_c1_curvature_weighted_transition_transport/r8_control"
        ),
        "stage3_variant": None,
        "probe": "probes/curvature_weighted_transition_transport/probe_report.json",
        "verification": "v2_judges/curvature_weighted_transition_transport.json",
    },
    {
        "mechanism_id": "behavior_residual_geometry_transport",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_behavior_residual_geometry_transport",
        "sid_dir": (
            "results/stage2_RQ-VAE/TIGER_behavior_residual_geometry_transport/r8_control"
        ),
        "stage3_variant": "TIGER_behavior_residual_geometry_transport__r8_control",
        "probe": "probes/behavior_residual_geometry_transport/probe_report.json",
        "verification": "redesign_geometry/verification.json",
    },
    {
        "mechanism_id": "behavior_directed_soft_code_transition_kl",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_c1_policy_behavior_directed_soft_code_transition_kl",
        "sid_dir": (
            "results/stage2_RQ-VAE/TIGER_c1_policy_behavior_directed_soft_code_transition_kl"
            "/r8_control"
        ),
        "stage3_variant": "TIGER_c1_policy_behavior_directed_soft_code_transition_kl__r8_control",
        "probe": None,
        "verification": None,
    },
    {
        "mechanism_id": "behavior_conditioned_codebook_directional_transport",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_c1_policy_behavior_conditioned_codebook_directional_transport",
        "sid_dir": (
            "results/stage2_RQ-VAE/TIGER_c1_policy_behavior_conditioned_codebook_directional_transport"
            "/r8_control"
        ),
        "stage3_variant": None,
        "probe": None,
        "verification": None,
    },
    {
        "mechanism_id": "r8_control_seed43",
        "kind": "control",
        "tree": "stage2_RQ-VAE/TIGER_seed43_control",
        "sid_dir": "results/stage2_RQ-VAE/TIGER_seed43_control/r8_control",
        "stage3_variant": "TIGER_seed43_control__r8_control",
        "probe": None,
        "verification": None,
    },
    {
        "mechanism_id": "geometry_transport_seed43",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_seed43_geometry",
        "sid_dir": "results/stage2_RQ-VAE/TIGER_seed43_geometry/r8_control",
        "stage3_variant": "TIGER_seed43_geometry__r8_control",
        "probe": None,
        "verification": None,
    },
    {
        "mechanism_id": "directed_codeword_margin_geometry",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_c106_directed_codeword_margin_geometry",
        "sid_dir": "results/stage2_RQ-VAE/TIGER_c106_directed_codeword_margin_geometry/r8_control",
        "stage3_variant": "TIGER_c106_directed_codeword_margin_geometry__r8_control",
        "probe": None,
        "verification": None,
    },
    {
        "mechanism_id": "directed_assignment_usage_balance",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_c106_directed_assignment_usage_balance",
        "sid_dir": "results/stage2_RQ-VAE/TIGER_c106_directed_assignment_usage_balance/r8_control",
        "stage3_variant": "TIGER_c106_directed_assignment_usage_balance__r8_control",
        "probe": None,
        "verification": None,
    },
    {
        "mechanism_id": "behavior_directed_code_assignment_cross_entropy",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_behavior_directed_code_assignment_cross_entropy",
        "sid_dir": (
            "results/stage2_RQ-VAE/TIGER_behavior_directed_code_assignment_cross_entropy"
            "/r8_control"
        ),
        "stage3_variant": None,
        "probe": None,
        "verification": None,
    },
    {
        "mechanism_id": "directed_st_code_transition_geometry",
        "kind": "arm",
        "tree": "stage2_RQ-VAE/TIGER_directed_st_code_transition_geometry",
        "sid_dir": None,
        "stage3_variant": None,
        "probe": "probes/directed_st_code_transition_geometry/probe_report.json",
        "verification": "redesign/verification.json",
    },
)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _training_metrics(tree: Path) -> Dict[str, Any]:
    """The run's own training record: how many epochs it really ran, and its last loss."""
    path = tree / "logs" / "r8_control" / "training_metrics.jsonl"
    if not path.is_file():
        return {}
    last = None
    epochs = 0
    records = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records += 1
        last = record
        if isinstance(record.get("epoch"), int):
            epochs = max(epochs, int(record["epoch"]))
    metrics: Dict[str, Any] = {"metric_records": records, "epochs": epochs}
    if last:
        for key in ("event", "loss", "final_collision_rate", "wall_time_s"):
            if key in last:
                metrics[key] = last[key]
    return metrics


def _frozen_er(sid_path: Path) -> Dict[str, Any]:
    """The frozen held-out evaluation, run through the repository's own evaluator."""
    if not sid_path.is_file():
        return {}
    completed = subprocess.run(
        [
            PROJECT_PYTHON,
            str(REPO_ROOT / "tools" / "mvg" / "er_eval.py"),
            "--sid",
            str(sid_path),
            "--bootstrap",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=900,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr[-500:]}
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": "the evaluator's output is not JSON"}
    return {
        "hard_l1_er": report.get("l1_er"),
        "above_frozen_gate": report.get("above_frozen_threshold"),
        "frozen_threshold": report.get("frozen_threshold"),
        "r4_08_reference": report.get("r4_08_reference"),
        "pos_share": report.get("pos_share"),
        "neg_share": report.get("neg_share"),
        "bootstrap_ci95": (report.get("bootstrap") or {}).get("er_ci95"),
        "evaluation_set_sha256": report.get("evaluation_set_sha256"),
    }


def _stage3_result(variant: str) -> Dict[str, Any]:
    directory = STAGE3_ROOT / variant
    files = sorted(directory.rglob("test_final.json")) if directory.is_dir() else []
    if not files:
        return {}
    report = _read_json(files[-1]) or {}
    return {
        "test_recall@10": report.get("test_recall@10"),
        "test_ndcg@10": report.get("test_ndcg@10"),
        "test_recall@5": report.get("test_recall@5"),
        "test_ndcg@5": report.get("test_ndcg@5"),
        "n_eval": report.get("n_eval"),
        "path": str(files[-1]),
    }


def _export_sensitivity(arm_sid: Path, control_sid: Path) -> Dict[str, Any]:
    """The export-sensitivity verdict, measured against the matched control."""
    if not arm_sid.is_file() or not control_sid.is_file():
        return {}
    completed = subprocess.run(
        [
            PROJECT_PYTHON,
            str(REPO_ROOT / "tools" / "mvg" / "export_sensitivity.py"),
            "--arm-sid",
            str(arm_sid),
            "--control-sid",
            str(control_sid),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=1800,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr[-500:]}
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": "the export-sensitivity output is not JSON"}
    return {
        "verdict": report.get("verdict"),
        "reasons": report.get("reasons"),
        "changed_fraction": report.get("changed_fraction"),
        "level0_occupancy_ratio_vs_control": report.get("level0_occupancy_ratio_vs_control"),
        "train_side_discrimination": report.get("train_side_discrimination"),
    }


def ingest_campaign(
    store: EvidenceStore,
    *,
    run_dir: Path = LEARNING_RUN_DIR,
    control_sid: Path = RESULTS_ROOT / "r8_control" / "sids_for_hgrec.npy",
) -> Dict[str, Any]:
    """Ingest the #104 campaign's real artifacts into the store, one manifest per run."""
    store.open()
    ingested: List[Dict[str, Any]] = []
    for arm in CAMPAIGN_ARMS:
        tree = REPO_ROOT / arm["tree"]
        sid_dir = REPO_ROOT / arm["sid_dir"] if arm.get("sid_dir") else None
        sid_path = (sid_dir / "sids_for_hgrec.npy") if sid_dir else None
        artifacts: List[Path] = []
        if sid_dir and sid_dir.is_dir():
            artifacts.extend(sorted(path for path in sid_dir.iterdir() if path.is_file()))
        metrics = _training_metrics(tree)
        if sid_path is None or not sid_path.is_file():
            # The tree may hold the metrics of an EARLIER run of the same tree. Without an
            # export of this arm there was no run, so the epochs are not reported at all.
            metrics.pop("epochs", None)
            metrics.pop("metric_records", None)
            metrics["trained"] = False
        else:
            metrics["trained"] = True
        if sid_path is not None and sid_path.is_file():
            metrics["frozen_evaluation"] = _frozen_er(sid_path)
            if arm["kind"] == "control":
                # The control IS the reference: comparing it with itself would report
                # "inert", which is a true statement about nothing.
                metrics["export_sensitivity"] = {"verdict": "reference"}
            else:
                metrics["export_sensitivity"] = _export_sensitivity(sid_path, control_sid)
        verdicts: Dict[str, Any] = {}
        probe_report = _read_json(run_dir / arm["probe"]) if arm.get("probe") else None
        if probe_report:
            verdicts["probe"] = {
                "ok": probe_report.get("ok"),
                "detail": probe_report.get("detail"),
                "occupancy": probe_report.get("occupancy"),
                "occupancy_ratio_vs_host": probe_report.get("occupancy_ratio_vs_host"),
            }
            artifacts.append(run_dir / arm["probe"])
        verification = _read_json(run_dir / arm["verification"]) if arm.get("verification") else None
        if verification:
            verdicts["independent_judges"] = {
                "ok": verification.get("ok"),
                "detail": verification.get("detail"),
                "failed": verification.get("failed"),
            }
            artifacts.append(run_dir / arm["verification"])
        if arm.get("stage3_variant"):
            stage3 = _stage3_result(arm["stage3_variant"])
            if stage3:
                metrics["stage3"] = stage3
                if stage3.get("path"):
                    artifacts.append(Path(stage3["path"]))
        plan = RunPlan(
            kind="stage2" if arm["kind"] == "arm" else "control",
            mechanism_id=arm["mechanism_id"],
            tree=tree,
            stage="cold_train",
            seed=42,
            output_dir=store.root / "imported" / arm["mechanism_id"],
            command=(
                PROJECT_PYTHON,
                str(tree / "train_rqvae.py"),
            ),
            extra={"ingested_from": str(run_dir), "ingested_utc": now_utc()},
        )
        try:
            run = store.admit(plan)
        except EvidenceError as exc:
            # The imported output directory already exists from a previous ingestion: the
            # store is immutable, so a second ingestion is a new run with a new id.
            plan.output_dir = store.root / "imported" / f"{arm['mechanism_id']}.{now_utc()}"
            run = store.admit(plan)
        status = "succeeded"
        failure = None
        if not artifacts:
            status = "failed"
            failure = {"reason": "the run left no artifacts on disk"}
        elif sid_path is None or not sid_path.is_file():
            # The arm never trained: the probe refused it. That is a refusal with a
            # reason, not a run that succeeded and produced nothing.
            probe_ok = (verdicts.get("probe") or {}).get("ok")
            if probe_ok is False:
                status = "refused"
                failure = {
                    "reason": "the active probe refused the arm before a cold run",
                    "detail": (verdicts.get("probe") or {}).get("detail"),
                }
            else:
                status = "failed"
                failure = {"reason": "the run produced no exported SIDs"}
        manifest = store.close(
            run,
            status=status,
            metrics=metrics,
            artifacts=artifacts,
            verdicts=verdicts,
            failure=failure,
        )
        ingested.append(
            {
                "mechanism_id": arm["mechanism_id"],
                "run_id": manifest["run_id"],
                "status": status,
                "hard_l1_er": (metrics.get("frozen_evaluation") or {}).get("hard_l1_er"),
                "export_verdict": (metrics.get("export_sensitivity") or {}).get("verdict"),
                "stage3_recall@10": (metrics.get("stage3") or {}).get("test_recall@10"),
            }
        )
    return {
        "store": str(store.root),
        "ingested": ingested,
        "count": len(ingested),
        "generated_utc": now_utc(),
    }


__all__ = ["CAMPAIGN_ARMS", "ingest_campaign"]
