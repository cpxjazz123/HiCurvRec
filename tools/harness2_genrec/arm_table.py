"""The per-arm table of the campaign: every arm, every number, from the evidence chain.

Issue #109's output contract asks for a readable report per arm and per round, and the
table is the readable form of the same evidence: the mechanism, its fingerprints, its
gates, its cold run, the frozen evaluation with its shares and its interval, the conditional
level-2 discrimination with its support, the history-conditioned probe, the structural
diagnostics, the downstream result and the deltas against the archived control.

The table is built from the store's immutable manifests and the transfer analysis, never
from a running summary, so it can be rebuilt at any time and always says the same thing.
An arm that was measured before a signal existed says which signal it is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The archived control, the baseline every delta is against.
CONTROL = {
    "hard_l1_er": 16.334505666275888,
    "test_recall@10": 0.06406796775709883,
    "test_ndcg@10": 0.03530651701940677,
}

#: The frozen ER gate.
ER_GATE = 16.38953942232631

#: The historical R4-08 value, a descriptive reference and not a gate.
R4_08_ER = 16.402627511591962


def build_table(
    store_runs: List[Dict[str, Any]],
    *,
    transfer: Optional[Dict[str, Any]] = None,
    diagnoses: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The table of every arm the store knows."""
    transfer = transfer or {}
    diagnoses = diagnoses or {}
    rows: List[Dict[str, Any]] = []
    for mechanism_id in sorted({m.get("mechanism_id") or "" for m in store_runs}):
        manifests = [m for m in store_runs if (m.get("mechanism_id") or "") == mechanism_id]
        evaluation: Dict[str, Any] = {}
        stage3: Dict[str, Any] = {}
        probe: Dict[str, Any] = {}
        run_ids: List[str] = []
        gates: Optional[Dict[str, Any]] = None
        semantic_key = ""
        fingerprint = ""
        for manifest in manifests:
            metrics = manifest.get("metrics") or {}
            for key, value in (metrics.get("frozen_evaluation") or {}).items():
                evaluation.setdefault(key, value)
            for key, value in (metrics.get("stage3") or {}).items():
                stage3.setdefault(key, value)
            for key, value in (metrics.get("gh1_probe") or {}).items():
                probe.setdefault(key, value)
            run_ids.append(manifest.get("run_id") or "")
            extra = manifest.get("extra") or {}
            semantic_key = semantic_key or (extra.get("semantic_key") or "")
            fingerprint = fingerprint or (extra.get("fingerprint") or "")
            if extra.get("cgv_gate"):
                gates = extra["cgv_gate"]
        if not evaluation and not stage3 and not probe:
            continue
        er = evaluation.get("hard_l1_er")
        if er is None:
            er = evaluation.get("l1_er")
        recall = stage3.get("test_recall@10")
        ndcg = stage3.get("test_ndcg@10")
        structure = transfer.get(mechanism_id) or {}
        balance = (structure.get("token_balance") or {}).get("level_1") or {}
        branches = (structure.get("prefix_structure") or {}).get("l1_to_l2") or {}
        collisions = (structure.get("collisions") or {}).get("level_1") or {}
        diagnosis = diagnoses.get(mechanism_id) or {}
        missing: List[str] = []
        if er is None:
            missing.append("the frozen ER")
        if not probe:
            missing.append("the history-conditioned probe")
        if evaluation.get("d2") is None and not evaluation.get("d2_block"):
            missing.append("the conditional level-2 discrimination")
        if recall is None:
            missing.append("the downstream measurement")
        rows.append(
            {
                "arm": mechanism_id,
                "semantic_key": semantic_key,
                "fingerprint": fingerprint,
                "runs": run_ids,
                "cgv_gates": gates,
                "hard_l1_er": er,
                "er_vs_gate": (er - ER_GATE) if er is not None else None,
                "er_vs_archived_control": (er - CONTROL["hard_l1_er"]) if er is not None else None,
                "er_ci95": evaluation.get("er_ci95"),
                "pos_share": evaluation.get("pos_share"),
                "neg_share": evaluation.get("neg_share"),
                "d2": evaluation.get("d2"),
                "d2_support": (evaluation.get("d2_block") or {}).get("flags"),
                "g_h1": probe.get("g_h1"),
                "g_h1_prior_ce": probe.get("prior_ce_l1"),
                "g_h1_probe_ce": probe.get("probe_ce_l1"),
                "g_h1_top1_acc_l1": probe.get("top1_acc_l1"),
                "l0_distinct_codes": balance.get("distinct_codes"),
                "l0_gini": balance.get("gini"),
                "l0_largest_cell": collisions.get("largest_cell"),
                "l1_to_l2_branches": branches.get("mean_branches"),
                "stage3_recall@10": recall,
                "stage3_ndcg@10": ndcg,
                "recall_delta_vs_control": (recall - CONTROL["test_recall@10"])
                if recall is not None
                else None,
                "ndcg_delta_vs_control": (ndcg - CONTROL["test_ndcg@10"]) if ndcg is not None else None,
                "cleared_the_gate": bool(er is not None and er > ER_GATE),
                "improves_both": bool(
                    recall is not None
                    and ndcg is not None
                    and recall > CONTROL["test_recall@10"]
                    and ndcg > CONTROL["test_ndcg@10"]
                ),
                "diagnosis": diagnosis.get("primary_mode"),
                "missing": missing,
            }
        )
    rows.sort(key=lambda row: -(row["hard_l1_er"] or 0))
    return {
        "generated_utc": _now(),
        "control": CONTROL,
        "er_gate": ER_GATE,
        "r4_08_er": R4_08_ER,
        "rows": rows,
        "arms": len(rows),
        "arms_clearing_the_gate": sum(1 for row in rows if row["cleared_the_gate"]),
        "arms_with_both_stage3_metrics": sum(
            1 for row in rows if row["stage3_recall@10"] is not None
        ),
        "arms_improving_both": sum(1 for row in rows if row["improves_both"]),
    }


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def to_markdown(table: Dict[str, Any]) -> str:
    """The table in a readable form."""
    lines = [
        "# The campaign's per-arm table",
        "",
        f"Generated {table['generated_utc']}. The archived control: ER "
        f"{table['control']['hard_l1_er']}, Recall@10 {table['control']['test_recall@10']}, "
        f"NDCG@10 {table['control']['test_ndcg@10']}. The gate: ER > {table['er_gate']}.",
        "",
        f"{table['arms']} arms; {table['arms_clearing_the_gate']} clear the gate; "
        f"{table['arms_with_both_stage3_metrics']} carry both downstream metrics; "
        f"**{table['arms_improving_both']} improve both**.",
        "",
        "| arm | ER | vs gate | D2 | G_H1 | dRecall | dNDCG | diagnosis | missing |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in table["rows"]:
        def fmt(value: Any, spec: str = ".4f") -> str:
            return format(value, spec) if isinstance(value, (int, float)) else "-"

        lines.append(
            f"| {row['arm'][:44]} | {fmt(row['hard_l1_er'])} | {fmt(row['er_vs_gate'], '+.4f')} | "
            f"{fmt(row['d2'], '+.5f')} | {fmt(row['g_h1'])} | {fmt(row['recall_delta_vs_control'], '+.6f')} | "
            f"{fmt(row['ndcg_delta_vs_control'], '+.6f')} | {row['diagnosis'] or '-'} | "
            f"{', '.join(row['missing']) or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_table(table: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    """Write the table as JSON and as markdown."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "per_arm_table.json"
    md_path = out_dir / "per_arm_table.md"
    json_path.write_text(
        json.dumps(table, indent=2, default=str) + "\n", encoding="utf-8"
    )
    md_path.write_text(to_markdown(table), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


__all__ = [
    "CONTROL",
    "ER_GATE",
    "R4_08_ER",
    "build_table",
    "to_markdown",
    "write_table",
]
