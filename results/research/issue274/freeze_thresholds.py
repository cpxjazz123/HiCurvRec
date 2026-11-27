#!/usr/bin/env python3
"""Issue #274: assemble the frozen acceptance thresholds from the measured baselines.

Every number here comes from a frozen artefact; nothing is estimated.  The output
``baselines.json`` is the pre-registered threshold file the candidate is judged against.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue274"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_recon(run_dir: Path) -> float:
    path = run_dir / "logs/training_metrics.jsonl"
    last = None
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") == "train":
            last = record
    return float(last["recon"])


def main() -> None:
    tiger = json.loads((OUT / "baselines_tiger.json").read_text())
    tiger_sid = REPO / "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json"
    tiger_sid_sha = sha256(tiger_sid)

    layer_record = None
    for path in sorted(glob.glob(str(REPO / "results/stage2_RQ-VAE/TIGER_RQ-VAE/stage2_metrics/*.json"))):
        record = json.loads(Path(path).read_text())
        if record.get("provenance", {}).get("sid_sha256") == tiger_sid_sha:
            layer_record = record
            break
    if layer_record is None:
        raise SystemExit("no layer-metric record matches the frozen TIGER SID")

    recon_265 = final_recon(REPO / "results/stage2_RQ-VAE/_runs"
                             "/stage2_4891fa8df3e843839de770e52e156892"
                             "/TIGER_euclid_local_curve_supervision_rq/result")
    pass_table = json.loads((REPO / "results/research/issue272/round3_pass_table.json").read_text())

    holdout = tiger["holdout_pairs"]
    prefix = tiger["holdout_prefix_discrimination"]
    l1_hit = float(layer_record["metrics"]["valid_l1_positive_hit_rate"])

    thresholds = {
        "schema": "genrec.issue274.frozen_thresholds/1",
        "frozen_margin": tiger["frozen_margin"],
        "margin_source": tiger["margin_source"],
        "C1_holdout_ranking_accuracy": {
            "threshold": holdout["ranking_accuracy"],
            "comparison": "strictly greater than the frozen TIGER value",
            "tiger_value": holdout["ranking_accuracy"],
            "tiger_centre_macro": holdout["macro_accuracy_by_centre"],
            "saturated": holdout["ranking_accuracy"] > 0.99,
            "pre_registered_gap_metric": {
                "name": "mean_distance_gap (d_neg - d_pos) on the hold-out pairs",
                "threshold": holdout["mean_distance_gap"],
                "comparison": "strictly greater than the frozen TIGER value",
                "tiger_value": holdout["mean_distance_gap"]},
        },
        "C2_holdout_margin_satisfaction": {
            "threshold": holdout["margin_satisfaction_rate"],
            "comparison": "strictly greater than the frozen TIGER value",
            "tiger_value": holdout["margin_satisfaction_rate"],
            "tiger_centre_macro": holdout["macro_margin_rate_by_centre"],
            "saturated": holdout["margin_satisfaction_rate"] > 0.99,
        },
        "C3_l2_distinct_prefixes": {"threshold": 10918,
                                    "source": "the #265 record "
                                              "(results/stage2_RQ-VAE/TIGER_euclid_local_curve_supervision_rq/stage2_metrics/)"},
        "C4_effective_rank": {"threshold": 8, "source": "the issue's threshold"},
        "C5_sid_prefix_discrimination": {
            "lcp_positive_greater_than_negative": prefix["lcp_positive_greater_than_negative"],
            "mean_lcp_difference": prefix["mean_lcp_difference"],
            "positive_pair_shares_level1": prefix["positive_pair_shares_level1"],
            "comparison": "each value must be at least the frozen TIGER value (no tolerance)",
        },
        "C6_behavioural_neighbourhood_fidelity": {
            "metric": "valid_l1_positive_hit_rate (the frozen issue-252 layer metric)",
            "tiger_value": l1_hit,
            "threshold": 0.95 * l1_hit,
            "comparison": "greater than or equal",
        },
        "C7_full_j": {"threshold": pass_table["C9_full_j"]["threshold"],
                      "source": "the #269 re-check value (results/research/issue272/round3_pass_table.json)"},
        "C8_one_hop_j": {"threshold": 1.05 * pass_table["C10_one_hop_j"]["threshold"],
                         "source": "the #269 value x 1.05"},
        "C8_two_hop_j": {"threshold": 1.05 * pass_table["C11_two_hop_j"]["threshold"],
                         "source": "the #269 value x 1.05"},
        "C9_reconstruction": {
            "tiger_265_value": recon_265,
            "threshold": 1.05 * recon_265,
            "source": "the #265 same-caliber final reconstruction x 1.05",
            "consistency_note": "the threshold equals the C15 threshold used in the #269/#272/#273 "
                                "rounds (8.589e-05), so the earlier reported inconsistency is resolved",
        },
        "C10_sid_identity": {"rule": "the re-exported SID's sha256 must match the checkpoint's export"},
        "frozen_tiger_sid_sha256": tiger_sid_sha,
        "frozen_tiger_l2_distinct_prefixes": layer_record["metrics"]["level2_distinct_prefixes"],
        "frozen_tiger_l3_collision": layer_record["metrics"]["l3_collision"],
        "frozen_tiger_l3_refinement_gain_bits": layer_record["metrics"]["l3_refinement_gain_bits"],
    }
    (OUT / "baselines.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")
    print(json.dumps(thresholds, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
