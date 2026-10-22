"""The per-arm report: everything issue #109's output contract asks for, in one place.

The contract is long and each item has to be traceable to an artifact, so the report is
assembled from the evidence chain rather than written: the store's immutable manifests for
the frozen evaluation, the conditional level-2 discrimination, the history-conditioned probe
and the downstream measurement; the screening record for the six gates and the GPU cost; the
transfer analysis for the structural diagnostics; the fingerprint registry for the
mechanism's novelty; the hypothesis ledger for the diagnosis; and the historical baseline
for the out-of-support flags.

The report says what is missing as loudly as what is there: an arm that was measured before
the signals existed has no D2 and no probe, and its report says so rather than leaving the
field out.
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


def _read(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def arm_report(
    mechanism_id: str,
    *,
    store_runs: List[Dict[str, Any]],
    transfer: Optional[Dict[str, Any]] = None,
    candidate: Optional[Dict[str, Any]] = None,
    fingerprint: Optional[Dict[str, Any]] = None,
    diagnosis: Optional[Dict[str, Any]] = None,
    historical: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The report of one arm, assembled from the evidence chain."""
    manifests = [m for m in store_runs if m.get("mechanism_id") == mechanism_id]
    evaluation: Dict[str, Any] = {}
    stage3: Dict[str, Any] = {}
    probe: Dict[str, Any] = {}
    run_ids: List[str] = []
    artifacts: List[Dict[str, Any]] = []
    cgv_gate: Optional[Dict[str, Any]] = None
    semantic_key = ""
    for manifest in manifests:
        metrics = manifest.get("metrics") or {}
        evaluation = evaluation or {}
        for key, value in (metrics.get("frozen_evaluation") or {}).items():
            evaluation.setdefault(key, value)
        for key, value in (metrics.get("stage3") or {}).items():
            stage3.setdefault(key, value)
        for key, value in (metrics.get("gh1_probe") or {}).items():
            probe.setdefault(key, value)
        run_ids.append(manifest.get("run_id") or "")
        artifacts.extend(manifest.get("artifacts") or ())
        extra = manifest.get("extra") or {}
        semantic_key = semantic_key or (extra.get("semantic_key") or "")
        if extra.get("cgv_gate"):
            cgv_gate = extra["cgv_gate"]

    er = evaluation.get("hard_l1_er")
    if er is None:
        er = evaluation.get("l1_er")
    recall = stage3.get("test_recall@10")
    ndcg = stage3.get("test_ndcg@10")
    structure = (transfer or {}).get(mechanism_id) or {}
    balance = (structure.get("token_balance") or {}).get("level_1") or {}
    branches = (structure.get("prefix_structure") or {}).get("l1_to_l2") or {}
    collisions = (structure.get("collisions") or {}).get("level_1") or {}
    stages = (candidate or {}).get("stages") or {}
    judges = stages.get("judges") or {}
    sensitivity = stages.get("export_sensitivity") or {}

    missing: List[str] = []
    if er is None:
        missing.append("the frozen ER")
    if not probe:
        missing.append("the history-conditioned probe")
    if not evaluation.get("d2") and evaluation.get("d2") is None:
        missing.append("the conditional level-2 discrimination")
    if recall is None:
        missing.append("the downstream measurement")

    report: Dict[str, Any] = {
        "mechanism_id": mechanism_id,
        "semantic_key": semantic_key,
        "fingerprints": fingerprint or {},
        "novelty": (
            "not measured against the historical record"
            if not fingerprint
            else "the mathematics is compared with the historical record in the fingerprint "
            "registry"
        ),
        "seed": 42,
        "runs": run_ids,
        "official_cold_run": {
            "status": (candidate or {}).get("status", "unknown"),
            "gpu_minutes": (candidate or {}).get("cold_run_gpu_minutes"),
            "wall_clock_s": (candidate or {}).get("wall_clock_s"),
        },
        "cgv_gates": cgv_gate or {
            "passed": None,
            "note": "the six gates are run at the launch; their states are recorded when the "
            "launch happened through this controller",
        },
        "independent_checks": {
            "ok": judges.get("ok"),
            "detail": judges.get("detail"),
            "failed": judges.get("failed"),
        },
        "export_sensitivity": {
            "verdict": sensitivity.get("verdict"),
            "changed_fraction": sensitivity.get("changed_fraction"),
            "level0_occupancy_ratio_vs_control": sensitivity.get(
                "level0_occupancy_ratio_vs_control"
            ),
        },
        "frozen_evaluation": {
            "hard_l1_er": er,
            "frozen_threshold": ER_GATE,
            "cleared_the_gate": bool(er is not None and er > ER_GATE),
            "pos_share": evaluation.get("pos_share"),
            "neg_share": evaluation.get("neg_share"),
            "ci95": evaluation.get("er_ci95"),
            "d2": evaluation.get("d2"),
            "d2_support": (evaluation.get("d2_block") or {}).get("flags"),
            "d2_counts": {
                "numerator_positive": (evaluation.get("d2_block") or {}).get("numerator_positive"),
                "denominator_positive": (evaluation.get("d2_block") or {}).get(
                    "denominator_positive"
                ),
                "numerator_negative": (evaluation.get("d2_block") or {}).get("numerator_negative"),
                "denominator_negative": (evaluation.get("d2_block") or {}).get(
                    "denominator_negative"
                ),
                "anchor_cluster_ci95": (evaluation.get("d2_block") or {}).get(
                    "anchor_cluster_ci95"
                ),
            },
        },
        "gh1_probe": {
            "g_h1": probe.get("g_h1"),
            "prior_ce_l1": probe.get("prior_ce_l1"),
            "probe_ce_l1": probe.get("probe_ce_l1"),
            "top1_acc_l1": probe.get("top1_acc_l1"),
            "top5_acc_l1": probe.get("top5_acc_l1"),
            "top20_acc_l1": probe.get("top20_acc_l1"),
            "split": probe.get("split"),
            "probe_version": probe.get("probe_version"),
            "budget": probe.get("budget"),
        },
        "l1_structural_diagnostics": {
            "distinct_codes": balance.get("distinct_codes"),
            "gini": balance.get("gini"),
            "largest_cell": collisions.get("largest_cell"),
            "prefix_branches": branches.get("mean_branches"),
            "sequence_information": (structure.get("sequential_predictability") or {})
            .get("level_1", {})
            .get("excess_over_the_marginal"),
        },
        "stage3": {
            "test_recall@10": recall,
            "test_ndcg@10": ndcg,
            "delta_recall_vs_control": (recall - CONTROL["test_recall@10"])
            if recall is not None
            else None,
            "delta_ndcg_vs_control": (ndcg - CONTROL["test_ndcg@10"]) if ndcg is not None else None,
            "per_token": (structure.get("stage3_slices") or {}).get("levels"),
        },
        "deltas_vs_archived_control": {
            "er": (er - CONTROL["hard_l1_er"]) if er is not None else None,
            "recall": (recall - CONTROL["test_recall@10"]) if recall is not None else None,
            "ndcg": (ndcg - CONTROL["test_ndcg@10"]) if ndcg is not None else None,
        },
        "artifacts": [
            {
                "path": artifact.get("path"),
                "sha256": artifact.get("sha256"),
                "bytes": artifact.get("bytes"),
            }
            for artifact in artifacts
        ],
        "posterior_explanation": diagnosis or {},
        "evidence_and_ood_flags": {
            "historical_feature_ranges": (historical or {}).get("out_of_support", {}).get(
                "feature_ranges"
            ),
            "certified_for_screening": (historical or {}).get("abstention", {}).get(
                "certified_for_screening"
            ),
            "the_archived_features_are_on_another_protocol": (historical or {})
            .get("protocol_comparability", {})
            .get("the_archived_features_are_the_frozen_pair_product"),
        },
        "missing": missing,
    }
    return report


def write_report(report: Dict[str, Any], out_dir: Path) -> Path:
    """Write one arm's report, and return where it went."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['mechanism_id']}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


__all__ = [
    "CONTROL",
    "ER_GATE",
    "arm_report",
    "write_report",
]
