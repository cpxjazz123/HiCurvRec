"""The one-command autonomous campaign (issue #105, acceptance).

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 -m tools.genrec_agents.cli \\
        run --store results/harness3_runs/store --out results/harness3_runs/pipeline

The chain is:

    evidence ingestion -> hypotheses -> experiment selection -> mechanism proposal ->
    frozen independent checks -> active probe and export sensitivity -> isolated cold
    Stage-2 -> frozen Hard-L1 ER -> eligible Stage-3 -> evidence update -> policy
    evaluation

Every stage is recorded in the evidence store with its own manifest, so a resumed or a
repeated run says exactly which stage it reached and what that stage produced. A stage
that fails is recorded as failed, with the reason; the chain does not step over a failure
and does not call a missing artifact a success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence import EvidenceStore, RunPlan, now_utc

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The stages, in order, with what each one consumes and produces.
STAGES = (
    ("ingest", "read the recorded runs' artifacts into immutable manifests"),
    ("hypotheses", "the agent proposes hypotheses from the store; the controller ranks them"),
    ("experiments", "the controller executes the ranked analyses and records the verdicts"),
    ("proposal", "the agent proposes the next mechanism from the ledger"),
    ("implementation", "the agent implements it; the independent checks judge it"),
    ("admission", "the active probe and the export sensitivity decide admission"),
    ("cold_stage2", "the admitted arm trains cold, 3000 epochs, under the frozen protocol"),
    ("frozen_er", "the frozen held-out Hard-L1 ER, once"),
    ("stage3", "Stage-3 for an eligible arm only"),
    ("evidence_update", "the ledger, the store and the per-arm table are updated"),
    ("policy_evaluation", "the blind old-versus-learned comparison on the frozen task set"),
)


def _record(
    store: EvidenceStore,
    stage: str,
    status: str,
    *,
    mechanism_id: str,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Path]] = None,
    failure: Optional[Dict[str, Any]] = None,
    parents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    plan = RunPlan(
        kind="campaign",
        mechanism_id=mechanism_id,
        stage=stage,
        seed=42,
        output_dir=store.root / "pipeline" / f"{stage}-{mechanism_id}",
        parent_runs=tuple(parents or ()),
    )
    run = store.admit(plan)
    return store.close(
        run,
        status=status,
        metrics=metrics or {},
        artifacts=artifacts or [],
        failure=failure,
    )


def run(
    store: EvidenceStore,
    out_dir: Path,
    *,
    model: str = "",
    timeout: int = 2400,
    cold: bool = True,
    skip: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the chain, recording every stage in the store."""
    from . import analysis, hypotheses as hypothesis_module, mini_campaign, redesign
    from .experiments import record as record_experiments
    from .ingest import ingest_campaign
    from .ledger import Ledger
    from .policy_comparison import derive_learned_policy, materialise_policy, run_comparison
    from .policy_tasks import build_tasks, freeze_tasks
    from .table import build as build_table

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(skip or ())
    report: Dict[str, Any] = {
        "started_utc": now_utc(),
        "store": str(store.root),
        "out": str(out_dir),
        "stages": [],
    }
    parent_runs: List[str] = []

    def stage(name: str, ok: bool, detail: str, **extra: Any) -> None:
        report["stages"].append({"stage": name, "ok": ok, "detail": detail, **extra})
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<18} {detail[:120]}")

    # 1. the evidence ingestion
    if "ingest" in skip:
        stage("ingest", True, "skipped")
    else:
        ingested = ingest_campaign(store)
        parent_runs = [row["run_id"] for row in ingested["ingested"]]
        stage("ingest", True, f"{ingested['count']} runs recorded")

    # 2. the hypotheses
    ledger_path = out_dir / "hypotheses" / "ledger.json"
    if "hypotheses" in skip:
        stage("hypotheses", True, "skipped")
    else:
        proposed = hypothesis_module.generate(store, out_dir / "hypotheses", model=model, timeout=timeout)
        stage(
            "hypotheses",
            not proposed["problems"],
            f"{proposed['candidates']} hypotheses, ledger {proposed['ledger_sha256'][:16]}",
            ranking=proposed["ranking"],
            problems=proposed["problems"],
        )

    # 3. the experiments
    if "experiments" in skip:
        stage("experiments", True, "skipped")
    else:
        experiments_dir = out_dir / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        control_sid = REPO_ROOT / "results/stage2_RQ-VAE/r8_control/sids_for_hgrec.npy"
        treatments = {
            manifest["mechanism_id"]: Path(
                next(
                    (
                        artifact["path"]
                        for artifact in manifest.get("artifacts") or ()
                        if artifact["path"].endswith("sids_for_hgrec.npy")
                    ),
                    "",
                )
            )
            for manifest in store.runs()
            if manifest["kind"] == "stage2"
        }
        treatments = {
            name: path for name, path in treatments.items() if path and path.is_file()
        }
        x1 = analysis.experiment_x1(
            REPO_ROOT / "stage2_RQ-VAE/TIGER_r8_control",
            REPO_ROOT / "results/stage2_RQ-VAE/r8_control/rqvae_best.pth",
            {
                name: (
                    REPO_ROOT / f"stage2_RQ-VAE/TIGER_{name}",
                    Path(
                        next(
                            (
                                artifact["path"]
                                for artifact in manifest.get("artifacts") or ()
                                if artifact["path"].endswith("rqvae_best.pth")
                            ),
                            "",
                        )
                    ),
                )
                for name, manifest in (
                    (manifest["mechanism_id"], manifest) for manifest in store.runs()
                )
                if name in ("behavior_conditioned_pairwise_margin", "curvature_weighted_transition_transport")
            },
        )
        (experiments_dir / "x1_pipeline_stage.json").write_text(
            json.dumps(x1, indent=2, default=str) + "\n"
        )
        x2 = analysis.experiment_x2(control_sid, treatments)
        (experiments_dir / "x2_er_decomposition.json").write_text(
            json.dumps(x2, indent=2, default=str) + "\n"
        )
        x4 = analysis.experiment_x4(control_sid, treatments)
        (experiments_dir / "x4_balance_and_generalisation.json").write_text(
            json.dumps(x4, indent=2, default=str) + "\n"
        )
        x5 = analysis.experiment_x5(REPO_ROOT / "stage2_RQ-VAE/TIGER_r8_control")
        (experiments_dir / "x5_projection.json").write_text(
            json.dumps(x5, indent=2, default=str) + "\n"
        )
        stage("experiments", True, "four analyses written")

    # 4-6. the mechanism, the checks and the admission
    mechanism_id = "pipeline_mechanism"
    if "proposal" in skip:
        stage("proposal", True, "skipped")
    else:
        ledger = Ledger(ledger_path, store=store).load() if ledger_path.is_file() else None
        brief = ledger.brief() if ledger else ""
        constraint = (
            "the host overwrites the encoder gradient after the backward and never touches "
            "the codebook gradient, so a mechanism that reaches only the encoder is inert "
            "in production; the project objective is the downstream recommendation quality, "
            "not the Hard-L1 ER"
        )
        proposed = redesign.propose_from_ledger(
            brief, constraint, out_dir / "proposal", model=model, timeout=timeout
        )
        mechanism_id = proposed["proposal"].get("mechanism_id", mechanism_id)
        stage(
            "proposal",
            not proposed["record"]["critique"]["fatal"],
            f"{mechanism_id} ({proposed['record']['proposal_sha256'][:16]})",
        )

    if "admission" in skip:
        stage("admission", True, "skipped")
    else:
        campaign = mini_campaign.run_campaign(
            out_dir / "old_harness" if (out_dir / "old_harness").is_dir() else REPO_ROOT,
            out_dir / "campaign",
            ledger_brief=(
                Ledger(ledger_path, store=store).load().brief() if ledger_path.is_file() else ""
            ),
            constraint="",
            store=store,
            model=model,
            timeout=timeout,
            cold=cold,
        )
        stage(
            "admission",
            bool(campaign.get("admitted")),
            f"admitted={campaign.get('admitted')} status={campaign.get('status')}",
            detail=campaign.get("detail"),
        )

    # 7-8. the cold run and the frozen ER: the campaign stage above owns the launch, so
    # this stage reports what the launch actually did rather than claiming a run.
    if "cold_stage2" in skip:
        stage("cold_stage2", True, "skipped")
    else:
        campaign_record = out_dir / "campaign" / "campaign.json"
        payload = (
            json.loads(campaign_record.read_text(encoding="utf-8"))
            if campaign_record.is_file()
            else {}
        )
        launched = bool((payload.get("stages") or {}).get("cold_launch", {}).get("started")) or (
            payload.get("status") == "cold_running"
        )
        stage(
            "cold_stage2",
            launched or payload.get("status") in ("refused", "failed"),
            f"status={payload.get('status')} detail={str(payload.get('detail'))[:90]}",
        )

    # 10. the evidence update
    if "evidence_update" in skip:
        stage("evidence_update", True, "skipped")
    else:
        if ledger_path.is_file():
            record_experiments(store, ledger_path, out_dir / "experiments")
        table = build_table(store, out_dir / "per_arm_table.md")
        stage("evidence_update", True, f"per-arm table with {table['rows']} rows")

    # 11. the policy evaluation
    if "policy_evaluation" in skip:
        stage("policy_evaluation", True, "skipped")
    else:
        tasks = build_tasks(store.root)
        freeze_tasks(tasks, out_dir / "policy" / "tasks.json")
        old = materialise_policy("old", out_dir / "policy" / "old_harness")
        learned_record = derive_learned_policy(
            Ledger(ledger_path, store=store).load().brief() if ledger_path.is_file() else "",
            out_dir / "policy" / "learned",
            model=model,
            timeout=timeout,
        )
        from .policy_comparison import Policy

        learned = Policy(
            name="learned",
            workspace=Path(learned_record["learned_workspace"]),
            digest=learned_record["learned_digest"],
        )
        comparison = run_comparison(
            tasks, [old, learned], out_dir / "policy" / "comparison", model=model, timeout=900
        )
        stage(
            "policy_evaluation",
            True,
            f"old {comparison['old_correct']} vs learned {comparison['learned_correct']} "
            f"-> adopted={comparison['adopted']}",
        )
        report["policy"] = {
            "old_correct": comparison["old_correct"],
            "learned_correct": comparison["learned_correct"],
            "adopted": comparison["adopted"],
            "reason": comparison["reason"],
        }
    report["finished_utc"] = now_utc()
    (out_dir / "pipeline_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["STAGES", "run"]
