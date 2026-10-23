"""The per-arm table (issue #105, acceptance).

One row per arm, every field measured, every path immutable. The table is assembled from
the evidence store's manifests -- not from a hand-written summary -- so a reader can check
each cell against the run it came from, and the table can be regenerated from the store
alone.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 -m tools.genrec_agents.cli \\
        table --store results/harness3_runs/store --out results/harness3_runs/per_arm_table.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence import EvidenceStore, now_utc

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The frozen references every row is compared against.
FROZEN_GATE = 16.38953942232631
R4_08_REFERENCE = 16.402627511591962

#: The issue each arm belongs to.
ISSUE_OF = {
    "r8_control": "#105 (the matched control for this window)",
    "behavior_conditioned_pairwise_curvature": "#103 (repaired, judged in #104)",
    "behavior_conditioned_pairwise_margin": "#103 (repaired, judged in #104)",
    "behaviour_conditioned_residual_target": "#103 (repaired, judged in #104)",
    "curvature_weighted_transition_transport": "#103 (repaired, judged in #104)",
    "behavior_residual_geometry_transport": "#104 (versioned redesign)",
    "directed_st_code_transition_geometry": "#104 (versioned redesign, refused at the probe)",
    "behavior_directed_code_assignment_cross_entropy": "#105 (ledger-derived, judged and probed)",
}


#: Where the frozen proposals live, so the table can name their digests.
PROPOSAL_PATHS = (
    REPO_ROOT / "results/harness3_runs/mechanism/proposal.json",
    REPO_ROOT / "results/harness3_runs/mechanism/proposal_v2.json",
    Path("/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/redesign/proposal.json"),
    Path("/home/wlia0047/ar57_scratch/wenyu/harness2_runs/learning_c1/redesign_geometry/proposal.json"),
    Path("/home/wlia0047/ar57_scratch/wenyu/harness2_runs/campaign_c1/arms"),
)


def _proposal_digest(mechanism_id: str) -> str:
    """The frozen proposal's digest, read from the proposal file itself."""
    for path in PROPOSAL_PATHS:
        if path.is_dir():
            candidate = path / mechanism_id / "proposal.json"
        else:
            candidate = path
        if not candidate.is_file():
            continue
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if document.get("mechanism_id") != mechanism_id:
            continue
        from .proposal import proposal_digest

        return proposal_digest(document)[:16]
    return "-"


def _arm_rows(store: EvidenceStore) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for manifest in store.runs():
        metrics = manifest.get("metrics") or {}
        evaluation = metrics.get("frozen_evaluation") or {}
        sensitivity = metrics.get("export_sensitivity") or {}
        stage3 = metrics.get("stage3") or {}
        verdicts = manifest.get("verdicts") or {}
        arm = manifest["mechanism_id"]
        er = evaluation.get("hard_l1_er")
        rows.append(
            {
                "arm": arm,
                "issue": ISSUE_OF.get(arm, "the run's own manifest"),
                "run_id": manifest["run_id"],
                "status": manifest["status"],
                "declaration_sha256": _proposal_digest(arm),
                "manifest_sha256": manifest.get("manifest_sha256"),
                "judges": verdicts.get("independent_judges", {}).get("detail")
                or verdicts.get("independent_judges", {}).get("ok")
                or "-",
                "probe": verdicts.get("probe", {}).get("detail") or "-",
                "export_verdict": sensitivity.get("verdict", "-"),
                "changed_l1": (sensitivity.get("changed_fraction") or {}).get("levels_1", "-"),
                "occupancy_ratio_vs_control": sensitivity.get(
                    "level0_occupancy_ratio_vs_control", "-"
                ),
                "epochs": metrics.get("epochs", "-"),
                "hard_l1_er": er,
                "vs_gate": (er - FROZEN_GATE) if isinstance(er, (int, float)) else "-",
                "vs_r4_08": (er - R4_08_REFERENCE) if isinstance(er, (int, float)) else "-",
                "er_ci95": evaluation.get("bootstrap_ci95", "-"),
                "stage3_recall@10": stage3.get("test_recall@10", "-"),
                "stage3_ndcg@10": stage3.get("test_ndcg@10", "-"),
                "gpu_minutes": metrics.get("wall_time_s", "-"),
                "artifacts": [artifact["path"] for artifact in manifest.get("artifacts") or ()],
            }
        )
    rows.sort(key=lambda row: (row["arm"] != "r8_control", row["arm"]))
    return rows


def _markdown(rows: List[Dict[str, Any]]) -> str:
    control = next((row for row in rows if row["arm"] == "r8_control"), None)
    lines: List[str] = []
    lines.append("# The per-arm table")
    lines.append("")
    lines.append(
        f"Generated {now_utc()} from the evidence store. The frozen Hard-L1 ER gate is "
        f"{FROZEN_GATE} and the historical R4-08 reference is {R4_08_REFERENCE}. The matched "
        "control for this window is the first row."
    )
    lines.append("")
    header = (
        "| arm | issue | declaration sha | judges | probe | export | epochs | Hard-L1 ER | "
        "vs gate | vs R4-08 | ER delta vs control | ER CI95 | Stage-3 R@10 | "
        "Stage-3 NDCG@10 | run id |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 16)
    for row in rows:
        delta = "-"
        if (
            control
            and isinstance(row["hard_l1_er"], (int, float))
            and isinstance(control["hard_l1_er"], (int, float))
        ):
            delta = f"{row['hard_l1_er'] - control['hard_l1_er']:+.4f}"
        lines.append(
            "| {arm} | {issue} | {declaration_sha256} | {judges} | {probe} | {export_verdict} "
            "| {epochs} | {hard_l1_er} | {vs_gate} | {vs_r4_08} | {delta} | {er_ci95} | "
            "{stage3_recall@10} | {stage3_ndcg@10} | {run_id} |".format(delta=delta, **row)
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
    return "\n".join(lines) + "\n"


def build(store: EvidenceStore, out_path: Path) -> Dict[str, Any]:
    rows = _arm_rows(store)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_markdown(rows), encoding="utf-8")
    (out_path.with_suffix(".json")).write_text(
        json.dumps({"generated_utc": now_utc(), "rows": rows}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {"rows": len(rows), "path": str(out_path)}


__all__ = ["FROZEN_GATE", "ISSUE_OF", "R4_08_REFERENCE", "build"]
