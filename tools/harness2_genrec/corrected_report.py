"""The corrected #105 report, from the existing artifacts only (issue #106, workstream A).

The issue names two discrepancies in the #105 reporting and asks for them to be resolved
using the existing evidence, with the original reports and their hashes preserved. This
module therefore **reads** the store and the campaign records and **writes a new report**;
it never rewrites a manifest, never re-runs an evaluation and never touches an original
report.

The two discrepancies it resolves:

* the per-arm table recorded 3000 epochs for `directed_st_code_transition_geometry`, an arm
  the probe rejected and which therefore never trained. The epochs came from the metrics
  file of its *tree*, which belongs to an earlier run of the same tree. The corrected report
  derives the epochs from the run's own artifacts: an arm trained if and only if it has its
  own exported SIDs.
* the policy comparison recorded "G6 failed, launch refused" and an ER for the same arm.
  Both are true of *different attempts*: the first attempt failed on the controller's own
  wiring and the relaunch passed all six gates and trained. The corrected report shows the
  attempt history instead of collapsing it into one ambiguous field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence import EvidenceStore, now_utc, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The frozen references.
FROZEN_GATE = 16.38953942232631
R4_08_REFERENCE = 16.402627511591962


def _has_own_export(manifest: Dict[str, Any]) -> bool:
    """A run trained if and only if it left its own exported SIDs."""
    for artifact in manifest.get("artifacts") or ():
        if artifact["path"].endswith("sids_for_hgrec.npy") and artifact.get("exists"):
            return True
    return False


def _epochs(manifest: Dict[str, Any]) -> Any:
    """The epochs THIS run ran -- not the epochs of whatever else used the tree."""
    if not _has_own_export(manifest):
        return "not trained"
    metrics = manifest.get("metrics") or {}
    epochs = metrics.get("epochs")
    return epochs if epochs else "unknown"


def _launch_status(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The launch facts of this run, and whether they are consistent with its ER."""
    verdicts = manifest.get("verdicts") or {}
    evaluation = (manifest.get("metrics") or {}).get("frozen_evaluation") or {}
    has_er = evaluation.get("hard_l1_er") is not None
    probe = verdicts.get("probe") or {}
    judges = verdicts.get("independent_judges") or {}
    facts = {
        "judges_ok": judges.get("ok"),
        "probe_ok": probe.get("ok"),
        "has_export": _has_own_export(manifest),
        "has_frozen_er": has_er,
    }
    if has_er and not facts["has_export"]:
        facts["consistency"] = "INCONSISTENT: an ER is reported without the run's own export"
    elif has_er and facts["probe_ok"] is False:
        facts["consistency"] = "INCONSISTENT: an ER is reported for an arm the probe refused"
    else:
        facts["consistency"] = "consistent"
    return facts


def _attempt_history(campaign_dir: Path) -> List[Dict[str, Any]]:
    """Every launch attempt of a campaign arm, in order, as the records show them."""
    attempts: List[Dict[str, Any]] = []
    if not campaign_dir.is_dir():
        return attempts
    for path in sorted(campaign_dir.glob("*/campaign.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        launch = (record.get("stages") or {}).get("cold_launch") or {}
        attempts.append(
            {
                "policy": path.parent.name,
                "status": record.get("status"),
                "admitted": record.get("admitted"),
                "launch_started": launch.get("started"),
                "launch_refused": launch.get("refused"),
                "cgv_gates": (launch.get("cgv_gate") or {}).get("gates"),
                "result_dir": launch.get("result_dir"),
            }
        )
    return attempts


def build(store: EvidenceStore, out_dir: Path) -> Dict[str, Any]:
    """The corrected report, written next to the originals and leaving them untouched."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for manifest in store.runs():
        metrics = manifest.get("metrics") or {}
        evaluation = metrics.get("frozen_evaluation") or {}
        sensitivity = metrics.get("export_sensitivity") or {}
        stage3 = metrics.get("stage3") or {}
        er = evaluation.get("hard_l1_er")
        rows.append(
            {
                "arm": manifest["mechanism_id"],
                "run_id": manifest["run_id"],
                "status": manifest["status"],
                "manifest_sha256": manifest.get("manifest_sha256"),
                "epochs": _epochs(manifest),
                "hard_l1_er": er,
                "er_provenance": {
                    "evaluation_set_sha256": evaluation.get("evaluation_set_sha256"),
                    "frozen_threshold": evaluation.get("frozen_threshold"),
                    "bootstrap_ci95": evaluation.get("bootstrap_ci95"),
                    "measured_by_run": manifest["run_id"] if er is not None else None,
                },
                "export_verdict": sensitivity.get("verdict"),
                "stage3_recall@10": stage3.get("test_recall@10"),
                "stage3_ndcg@10": stage3.get("test_ndcg@10"),
                "launch": _launch_status(manifest),
                "artifacts": [
                    {
                        "path": artifact["path"],
                        "sha256": artifact.get("sha256"),
                        "exists": artifact.get("exists"),
                    }
                    for artifact in manifest.get("artifacts") or ()
                ],
            }
        )
    rows.sort(key=lambda row: (row["arm"] != "r8_control", row["arm"]))
    attempts = _attempt_history(REPO_ROOT / "results/harness3_runs/campaigns_v2")
    report = {
        "generated_utc": now_utc(),
        "basis": "the existing store's manifests and campaign records only; no run, no "
                 "evaluation and no manifest was re-executed or rewritten",
        "frozen_gate": FROZEN_GATE,
        "r4_08_reference": R4_08_REFERENCE,
        "rows": rows,
        "launch_attempts": attempts,
        "corrections": [
            {
                "arm": "directed_st_code_transition_geometry",
                "was": "3000 epochs recorded, with no ER and no export",
                "now": "not trained: the active probe refused it and it has no export of its own; "
                       "the 3000 epochs in the original manifest came from its tree's metrics "
                       "file, which belongs to an earlier run of the same tree",
                "original_preserved": True,
            },
            {
                "arm": "the policy campaign arms",
                "was": "one record that said G6 failed and the launch was refused, and also "
                       "carried an ER",
                "now": "the attempt history: the first attempt was refused on the controller's "
                       "own wiring and the relaunch passed all six gates and trained; the ER "
                       "belongs to the relaunch",
                "original_preserved": True,
            },
        ],
    }
    path = out_dir / "corrected_per_arm_report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# The corrected per-arm report (issue #106)",
        "",
        report["basis"] + ".",
        "",
        f"The frozen Hard-L1 ER gate is {FROZEN_GATE} and the historical R4-08 reference is "
        f"{R4_08_REFERENCE}.",
        "",
        "| arm | status | epochs | Hard-L1 ER | export | Stage-3 R@10 | consistency | run id |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {arm} | {status} | {epochs} | {hard_l1_er} | {export_verdict} | "
            "{stage3_recall} | {consistency} | {run_id} |".format(
                stage3_recall=row["stage3_recall@10"],
                consistency=row["launch"]["consistency"],
                **row,
            )
        )
    lines.append("")
    lines.append("## The corrections")
    lines.append("")
    for correction in report["corrections"]:
        lines.append(f"* **{correction['arm']}** -- was: {correction['was']}. Now: {correction['now']}.")
    lines.append("")
    lines.append("## The launch attempts")
    lines.append("")
    for attempt in attempts:
        lines.append(
            f"* {attempt['policy']}: admitted={attempt['admitted']} status={attempt['status']} "
            f"started={attempt['launch_started']} refused={str(attempt['launch_refused'])[:120]} "
            f"gates={attempt['cgv_gates']}"
        )
    (out_dir / "corrected_per_arm_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(rows), "json": str(path), "markdown": str(out_dir / "corrected_per_arm_report.md")}


__all__ = ["build"]
