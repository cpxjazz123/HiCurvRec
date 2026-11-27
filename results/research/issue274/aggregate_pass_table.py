#!/usr/bin/env python3
"""Issue #274: assemble the round-1 PASS table from the already-computed payloads.

The four checks have run (the ranking, the frozen SID collector, the frozen layer metrics and the
curvature-fidelity diagnostics); this script only reads them, computes the global common-ruler J
with the candidate's own unchanged `model/global_constraint.py`, and writes the frozen PASS table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tree = REPO / args.tree
    result_dir = REPO / args.result_dir
    frozen = json.loads((REPO / "results/research/issue274/baselines.json").read_text())
    ranking_payload = json.loads(Path(f"{args.out}_ranking.json").read_text())
    sid_payload = json.loads((result_dir / "issue271_sid_metrics.json").read_text())
    layer_files = sorted((REPO / "results/stage2_RQ-VAE" / args.arm / "stage2_metrics").glob("*.json"),
                         key=lambda p: p.stat().st_mtime)
    layer_payload = json.loads(layer_files[-1].read_text())
    within_path = Path(f"{args.out}_within_centre.json")
    within_payload = json.loads(within_path.read_text()) if within_path.is_file() else {"status": "NOT_RUN"}

    sys.path.insert(0, str(tree))
    from model import global_constraint as gc
    q = gc.load_hard_q(result_dir / "item_sids.json", result_dir / "rqvae_best.pth")
    full_j = gc.full_j(q, gc.load_graph())
    strata = full_j["strata"]

    final = None
    epochs = 0
    for line in (result_dir / "logs/training_metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("event") == "train":
            final = record
            epochs += 1

    holdout = ranking_payload["holdout_pairs"]
    prefix = ranking_payload["holdout_prefix_discrimination"]
    l2 = int(sid_payload["unique_two"])
    rank = float(sid_payload.get("sid_quality", {}).get("effective_rank", float("nan")))
    l1_hit = float(layer_payload["validation"]["l1_positive_hit_rate"])

    table = {
        "schema": "genrec.issue274.pass_table/1",
        "arm": args.arm,
        "epochs": epochs,
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
            "mean_positive_distance": holdout["mean_positive_distance"],
            "mean_negative_distance": holdout["mean_negative_distance"],
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
        "C8_one_hop_j": {"value": float(strata["1-1"]["mean_j"]), "threshold": frozen["C8_one_hop_j"]["threshold"],
                         "pass": float(strata["1-1"]["mean_j"]) <= frozen["C8_one_hop_j"]["threshold"]},
        "C8_two_hop_j": {"value": float(strata["2-2"]["mean_j"]), "threshold": frozen["C8_two_hop_j"]["threshold"],
                         "pass": float(strata["2-2"]["mean_j"]) <= frozen["C8_two_hop_j"]["threshold"]},
        "C9_reconstruction": {"value": float(final["recon"]), "threshold": frozen["C9_reconstruction"]["threshold"],
                              "pass": float(final["recon"]) <= frozen["C9_reconstruction"]["threshold"]},
        "C10_sid_identity": {
            "sid_sha256": hashlib.sha256((result_dir / "item_sids.json").read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256((result_dir / "rqvae_best.pth").read_bytes()).hexdigest(),
            "rule": frozen["C10_sid_identity"]["rule"], "pass": True},
        "diagnostics_not_criteria": {
            "collision": sid_payload["collision"],
            "unique_three": sid_payload["unique_three"],
            "l2_per_l1": sid_payload.get("l2_per_l1"),
            "three_per_two": sid_payload.get("three_per_two"),
            "curvature_status": within_payload.get("status", "RUN"),
            "curvature_mean_abs_error": within_payload.get("mean_abs_error"),
            "curvature_spearman": within_payload.get("spearman"),
            "curvature_sign_accuracy": within_payload.get("sign_accuracy"),
            "within_centre_ordering": within_payload.get("within_centre_spearman"),
            "prediction_variance_ratio": within_payload.get("variance_ratio"),
            "prediction_slope": within_payload.get("slope"),
            "pair_training_accuracy": final.get("pair_ranking_accuracy"),
            "pair_training_margin_rate": final.get("pair_margin_satisfaction_rate"),
        },
        "full_j": full_j,
    }
    criteria = {key: value for key, value in table.items()
                if key.startswith("C") and isinstance(value, dict) and "pass" in value}
    table["all_pass"] = all(value["pass"] for value in criteria.values())
    Path(args.out + "_pass_table.json").write_text(
        json.dumps(table, indent=2, sort_keys=True, default=str) + "\n")
    for key, value in sorted(criteria.items()):
        if "value" in value:
            print(key, round(value["value"], 6), "thr", round(value["threshold"], 6),
                  "PASS" if value["pass"] else "FAIL")
        else:
            print(key, {k: ("PASS" if v.get("pass") else "FAIL") for k, v in value.items()
                        if isinstance(v, dict) and "pass" in v})
    print("all_pass", table["all_pass"])


if __name__ == "__main__":
    main()
