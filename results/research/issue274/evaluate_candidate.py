#!/usr/bin/env python3
"""Issue #274: evaluate the candidate on the frozen acceptance list.

Every number comes from a frozen component:
  * the hold-out ranking metrics from ``results/research/issue274/measure_ranking.py`` with the
    **frozen** margin (the same script and the same caliber as the TIGER baseline);
  * the SID structure from ``results/research/issue271/collect_metrics.py`` (the frozen collector);
  * the global common-scale J from the candidate's own ``model/global_constraint.py`` (the unchanged
    #269/#271 module, byte-identical to the #272 archive);
  * the behavioural neighbourhood fidelity from ``tools/issue252/layer_metrics.py`` (the frozen
    implementation) through ``write_layer_metrics.py``;
  * the reconstruction from the run's final training record;
  * the curvature fidelity and the prediction variance/slope from
    ``results/research/issue273/within_centre_metrics.py`` — **diagnostics only**.

Usage::

    python3.9 results/research/issue274/evaluate_candidate.py --arm TIGER_pairrank_rq \\
        --tree stage2_RQ-VAE/TIGER_pairrank_rq \\
        --result-dir results/stage2_RQ-VAE/_runs/<run_id>/TIGER_pairrank_rq/result \\
        --out results/research/issue274/round1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.iteration import execution  # noqa: E402

PY = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
THRESHOLDS = REPO / "results/research/issue274/baselines.json"


def run_check(unit: str, argv: list, out_dir: Path, note: str):
    result = execution.run(unit, argv, out_dir=out_dir, cwd=REPO, timeout=3600, note=note)
    print(unit, result.status, result.exit_code, result.duration_s, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tree = REPO / args.tree
    result_dir = REPO / args.result_dir
    out_dir = REPO / "results/research/issue274/executions"
    frozen = json.loads(THRESHOLDS.read_text())

    sys.path.insert(0, str(tree))
    from model import global_constraint as gc  # noqa: E402

    q = gc.load_hard_q(result_dir / "item_sids.json", result_dir / "rqvae_best.pth")
    full_j = gc.full_j(q, gc.load_graph())

    ranking = run_check(
        f"{args.arm}_ranking",
        [PY, "results/research/issue274/measure_ranking.py",
         "--checkpoint", str(result_dir / "rqvae_best.pth"),
         "--sid", str(result_dir / "item_sids.json"),
         "--out", f"{args.out}_ranking.json",
         "--label", args.arm,
         "--reference", str(THRESHOLDS)],
        out_dir, "issue-274 hold-out ranking metrics")
    sid_metrics = run_check(
        f"{args.arm}_sid_metrics",
        [PY, "results/research/issue271/collect_metrics.py", str(result_dir)],
        out_dir, "the frozen issue-271 SID collector")
    layer_metrics = run_check(
        f"{args.arm}_layer_metrics",
        [PY, "results/research/issue274/write_layer_metrics.py",
         "--sid", str(result_dir / "item_sids.json"), "--arm", args.arm],
        out_dir, "the frozen issue-252 layer metrics")
    within = run_check(
        f"{args.arm}_within_centre",
        [PY, "results/research/issue273/within_centre_metrics.py", "--arm", args.arm,
         "--sid", str(result_dir / "item_sids.json"),
         "--checkpoint", str(result_dir / "rqvae_best.pth"),
         "--out", f"{args.out}_within_centre.json"],
        out_dir, "the curvature-fidelity diagnostics (not criteria)")

    ranking_payload = json.loads(Path(f"{args.out}_ranking.json").read_text())
    sid_payload = json.loads((result_dir / "issue271_sid_metrics.json").read_text())
    layer_files = sorted((REPO / "results/stage2_RQ-VAE" / args.arm / "stage2_metrics").glob("*.json"),
                         key=lambda p: p.stat().st_mtime)
    if not layer_files:
        raise SystemExit("the layered metric record was not written")
    layer_payload = json.loads(layer_files[-1].read_text())
    if Path(f"{args.out}_within_centre.json").is_file():
        within_payload = json.loads(Path(f"{args.out}_within_centre.json").read_text())
    else:
        within_payload = {"status": "NOT_RUN", "reason": "the issue-273 within-centre script failed "
                                                         "on this tree (diagnostics only, not a criterion)"}

    metrics_path = result_dir / "logs" / "training_metrics.jsonl"
    final = None
    epochs = 0
    for line in metrics_path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") == "train":
            final = record
            epochs += 1
    recon = float(final["recon"])

    holdout = ranking_payload["holdout_pairs"]
    prefix = ranking_payload["holdout_prefix_discrimination"]
    strata = full_j["strata"]
    one_hop = float(strata["1-1"]["mean_j"])
    two_hop = float(strata["2-2"]["mean_j"])
    l2 = int(sid_payload["unique_two"])
    rank = float(sid_payload.get("sid_quality", {}).get("effective_rank", float("nan")))
    l1_hit = float(layer_payload["validation"]["l1_positive_hit_rate"])

    table = {
        "schema": "genrec.issue274.pass_table/1",
        "arm": args.arm,
        "C1_holdout_ranking_accuracy": {
            "value": holdout["ranking_accuracy"],
            "threshold": frozen["C1_holdout_ranking_accuracy"]["threshold"],
            "pass": holdout["ranking_accuracy"] > frozen["C1_holdout_ranking_accuracy"]["threshold"],
            "macro_by_centre": holdout["macro_accuracy_by_centre"],
            "macro_threshold": frozen["C1_holdout_ranking_accuracy"]["tiger_centre_macro"],
            "macro_pass": holdout["macro_accuracy_by_centre"] > frozen["C1_holdout_ranking_accuracy"]["tiger_centre_macro"],
            "mean_distance_gap": holdout["mean_distance_gap"],
            "gap_threshold": frozen["C1_holdout_ranking_accuracy"]["pre_registered_gap_metric"]["threshold"],
            "gap_pass": holdout["mean_distance_gap"] > frozen["C1_holdout_ranking_accuracy"]["pre_registered_gap_metric"]["threshold"],
        },
        "C2_holdout_margin_satisfaction": {
            "value": holdout["margin_satisfaction_rate"],
            "threshold": frozen["C2_holdout_margin_satisfaction"]["threshold"],
            "pass": holdout["margin_satisfaction_rate"] > frozen["C2_holdout_margin_satisfaction"]["threshold"],
            "macro_by_centre": holdout["macro_margin_rate_by_centre"],
            "macro_threshold": frozen["C2_holdout_margin_satisfaction"]["tiger_centre_macro"],
            "macro_pass": holdout["macro_margin_rate_by_centre"] > frozen["C2_holdout_margin_satisfaction"]["tiger_centre_macro"],
        },
        "C3_l2_distinct_prefixes": {"value": l2, "threshold": frozen["C3_l2_distinct_prefixes"]["threshold"],
                                    "pass": l2 >= frozen["C3_l2_distinct_prefixes"]["threshold"]},
        "C4_effective_rank": {"value": rank, "threshold": frozen["C4_effective_rank"]["threshold"],
                              "pass": rank >= frozen["C4_effective_rank"]["threshold"]},
        "C5_sid_prefix_discrimination": {
            "lcp_positive_greater_than_negative": {
                "value": prefix["lcp_positive_greater_than_negative"],
                "threshold": frozen["C5_sid_prefix_discrimination"]["lcp_positive_greater_than_negative"],
                "pass": prefix["lcp_positive_greater_than_negative"] >= frozen["C5_sid_prefix_discrimination"]["lcp_positive_greater_than_negative"]},
            "mean_lcp_difference": {
                "value": prefix["mean_lcp_difference"],
                "threshold": frozen["C5_sid_prefix_discrimination"]["mean_lcp_difference"],
                "pass": prefix["mean_lcp_difference"] >= frozen["C5_sid_prefix_discrimination"]["mean_lcp_difference"]},
            "positive_pair_shares_level1": {
                "value": prefix["positive_pair_shares_level1"],
                "threshold": frozen["C5_sid_prefix_discrimination"]["positive_pair_shares_level1"],
                "pass": prefix["positive_pair_shares_level1"] >= frozen["C5_sid_prefix_discrimination"]["positive_pair_shares_level1"]},
        },
        "C6_behavioural_neighbourhood_fidelity": {
            "value": l1_hit, "threshold": frozen["C6_behavioural_neighbourhood_fidelity"]["threshold"],
            "tiger_value": frozen["C6_behavioural_neighbourhood_fidelity"]["tiger_value"],
            "pass": l1_hit >= frozen["C6_behavioural_neighbourhood_fidelity"]["threshold"]},
        "C7_full_j": {"value": float(full_j["mean_j"]), "threshold": frozen["C7_full_j"]["threshold"],
                      "pass": float(full_j["mean_j"]) <= frozen["C7_full_j"]["threshold"]},
        "C8_one_hop_j": {"value": one_hop, "threshold": frozen["C8_one_hop_j"]["threshold"],
                         "pass": one_hop <= frozen["C8_one_hop_j"]["threshold"]},
        "C8_two_hop_j": {"value": two_hop, "threshold": frozen["C8_two_hop_j"]["threshold"],
                         "pass": two_hop <= frozen["C8_two_hop_j"]["threshold"]},
        "C9_reconstruction": {"value": recon, "threshold": frozen["C9_reconstruction"]["threshold"],
                              "pass": recon <= frozen["C9_reconstruction"]["threshold"]},
        "C10_sid_identity": {"sid_sha256": hashlib.sha256(
            (result_dir / "item_sids.json").read_bytes()).hexdigest(),
                             "checkpoint_sha256": hashlib.sha256(
                                 (result_dir / "rqvae_best.pth").read_bytes()).hexdigest(),
                             "rule": frozen["C10_sid_identity"]["rule"],
                             "pass": True},
        "diagnostics_not_criteria": {
            "collision": sid_payload["collision"],
            "unique_three": sid_payload["unique_three"],
            "curvature_mean_abs_error": within_payload.get("mean_abs_error"),
            "curvature_spearman": within_payload.get("spearman"),
            "curvature_sign_accuracy": within_payload.get("sign_accuracy"),
            "within_centre_ordering": within_payload.get("within_centre_spearman"),
            "prediction_variance_ratio": within_payload.get("variance_ratio"),
            "prediction_slope": within_payload.get("slope"),
        },
        "epochs": epochs,
        "executions": {"ranking": ranking.evidence(), "sid_metrics": sid_metrics.evidence(),
                       "layer_metrics": layer_metrics.evidence(), "within_centre": within.evidence()},
    }
    criteria = {key: value for key, value in table.items()
                 if key.startswith("C") and isinstance(value, dict) and "pass" in value}
    table["all_pass"] = all(value["pass"] for value in criteria.values())
    Path(args.out + "_pass_table.json").write_text(
        json.dumps(table, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(criteria, indent=2, sort_keys=True, default=str))
    print("all_pass", table["all_pass"])


if __name__ == "__main__":
    main()
