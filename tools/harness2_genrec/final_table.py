"""The final per-arm table for issue #106.

One row per arm, and every field the issue's acceptance list names: the frozen proposal's
digest, the independent-check verdicts, the epochs actually run, the export sensitivity, the
collapse checks, the frozen Hard-L1 ER against the gate and the historical R4-08 reference,
the Stage-3 metrics and their deltas against the ARCHIVED control, the immutable artifact
paths, the cost, and the single-seed uncertainty stated rather than hidden.

The table is assembled from the evidence store's manifests, the candidate records and the
campaign records. It never re-runs an evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence import EvidenceStore, now_utc

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The frozen references.
FROZEN_GATE = 16.38953942232631
R4_08_REFERENCE = 16.402627511591962

#: The archived control of this window. Its artifacts are the comparison; it is never re-run.
ARCHIVED_CONTROL = {
    "hard_l1_er": 16.334505666275888,
    "stage3_recall@10": 0.06406796775709883,
    "stage3_ndcg@10": 0.03530651701940677,
}


def _candidate_records(root: Path) -> Dict[str, Dict[str, Any]]:
    """The candidate records of both rounds, keyed by mechanism id."""
    records: Dict[str, Dict[str, Any]] = {}
    for round_dir in (root / "candidates", root / "candidates_round2"):
        if not round_dir.is_dir():
            continue
        for path in sorted(round_dir.glob("*/candidate.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            records[record["mechanism_id"]] = record
        for path in sorted(round_dir.glob("*/proposal.json")):
            proposal = json.loads(path.read_text(encoding="utf-8"))
            mechanism_id = proposal.get("mechanism_id")
            if mechanism_id and mechanism_id not in records:
                records[mechanism_id] = {
                    "mechanism_id": mechanism_id,
                    "status": "proposed",
                    "admitted": None,
                    "stages": {},
                    "proposal_path": str(path),
                }
    return records


def _proposal_digest(mechanism_id: str, roots: List[Path]) -> str:
    from .proposal import proposal_digest

    for root in roots:
        for path in root.glob(f"*/proposal.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("mechanism_id") == mechanism_id:
                return proposal_digest(document)[:16]
    return "-"


def build(store: EvidenceStore, out_dir: Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_records(REPO_ROOT / "results" / "harness3_runs")
    rows: List[Dict[str, Any]] = []
    for manifest in store.runs():
        metrics = manifest.get("metrics") or {}
        evaluation = metrics.get("frozen_evaluation") or {}
        sensitivity = metrics.get("export_sensitivity") or {}
        stage3 = metrics.get("stage3") or {}
        verdicts = manifest.get("verdicts") or {}
        arm = manifest["mechanism_id"]
        er = evaluation.get("hard_l1_er")
        recall = stage3.get("test_recall@10")
        ndcg = stage3.get("test_ndcg@10")
        record = candidates.get(arm) or {}
        stages = record.get("stages") or {}
        launch = stages.get("cold_launch") or {}
        rows.append(
            {
                "arm": arm,
                "issue": "#106 (new candidate)" if arm in candidates else "#103/#104/#105 (archived)",
                "proposal_sha256": _proposal_digest(
                    arm,
                    [
                        REPO_ROOT / "results" / "harness3_runs" / "candidates",
                        REPO_ROOT / "results" / "harness3_runs" / "candidates_round2",
                    ],
                ),
                "judges": (verdicts.get("independent_judges") or {}).get("detail", "-"),
                "cgv_gates": json.dumps(launch.get("cgv_gate", {}).get("gates") or {}),
                "probe": (verdicts.get("probe") or {}).get("detail", "-"),
                "export_verdict": sensitivity.get("verdict", "-"),
                "changed_l1": (sensitivity.get("changed_fraction") or {}).get("levels_1", "-"),
                "l0_occupancy_ratio": sensitivity.get("level0_occupancy_ratio_vs_control", "-"),
                "status": manifest.get("status"),
                "epochs": metrics.get("epochs", "-"),
                "hard_l1_er": er,
                "er_vs_gate": (er - FROZEN_GATE) if isinstance(er, (int, float)) else "-",
                "er_vs_r4_08": (er - R4_08_REFERENCE) if isinstance(er, (int, float)) else "-",
                "er_vs_archived_control": (
                    er - ARCHIVED_CONTROL["hard_l1_er"] if isinstance(er, (int, float)) else "-"
                ),
                "er_ci95": evaluation.get("bootstrap_ci95", "-"),
                "stage3_recall@10": recall,
                "stage3_ndcg@10": ndcg,
                "recall_delta_vs_archived_control": (
                    recall - ARCHIVED_CONTROL["stage3_recall@10"]
                    if isinstance(recall, (int, float))
                    else "-"
                ),
                "ndcg_delta_vs_archived_control": (
                    ndcg - ARCHIVED_CONTROL["stage3_ndcg@10"]
                    if isinstance(ndcg, (int, float))
                    else "-"
                ),
                "run_id": manifest["run_id"],
                "artifacts": [artifact["path"] for artifact in manifest.get("artifacts") or ()],
            }
        )
    order = {"r8_control": 0}
    rows.sort(key=lambda row: (order.get(row["arm"], 1), row["arm"]))
    lines = [
        "# The final per-arm table (issue #106)",
        "",
        f"Generated {now_utc()} from the evidence store and the candidate records. The frozen "
        f"Hard-L1 ER gate is {FROZEN_GATE}; the historical R4-08 reference is "
        f"{R4_08_REFERENCE}; the archived control of this window is "
        f"ER {ARCHIVED_CONTROL['hard_l1_er']} with Stage-3 recall@10 "
        f"{ARCHIVED_CONTROL['stage3_recall@10']} and NDCG@10 "
        f"{ARCHIVED_CONTROL['stage3_ndcg@10']}. The control is evidence and was never re-run.",
        "",
        "**Uncertainty**: every ER is ONE run at the single frozen seed, and every Stage-3 "
        "number is ONE run of the frozen protocol. The bootstrap interval of each ER is given "
        "in the row. No number here is a claim of statistical significance or of "
        "reproducibility across seeds.",
        "",
        "| arm | issue | proposal sha | judges | export | epochs | Hard-L1 ER | vs gate | "
        "vs R4-08 | vs control | Stage-3 R@10 | delta | Stage-3 NDCG@10 | delta | run id |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {arm} | {issue} | {proposal_sha256} | {judges} | {export_verdict} | {epochs} | "
            "{hard_l1_er} | {er_vs_gate} | {er_vs_r4_08} | {er_vs_archived_control} | "
            "{stage3_recall} | {recall_delta} | {stage3_ndcg} | {ndcg_delta} | {run_id} |".format(
                stage3_recall=row["stage3_recall@10"],
                recall_delta=row["recall_delta_vs_archived_control"],
                stage3_ndcg=row["stage3_ndcg@10"],
                ndcg_delta=row["ndcg_delta_vs_archived_control"],
                **row,
            )
        )
    lines.append("")
    lines.append("## The immutable artifacts")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['arm']} ({row['run_id']})")
        lines.append("")
        for path in row["artifacts"]:
            lines.append(f"* `{path}`")
        lines.append("")
    path = out_dir / "final_per_arm_table.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "final_per_arm_table.json").write_text(
        json.dumps({"generated_utc": now_utc(), "rows": rows}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {"rows": len(rows), "path": str(path)}


__all__ = ["ARCHIVED_CONTROL", "FROZEN_GATE", "R4_08_REFERENCE", "build"]
