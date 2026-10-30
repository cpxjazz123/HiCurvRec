#!/usr/bin/env python3
"""One table: every metric the rounds measured, on the 107 confirmed pairs.

The analysis set is the WHOLE collection -- no split, no subset, no extra pass
condition. Each metric was computed independently, on the same 107 pairs, and the
table reports its Spearman with Recall@10, its bootstrap interval and its
p-value. The comparators (the Euclidean control, the permuted control) are in the
same table, so the reading is one look.

The exploration caveat applies: many metrics were tried on this collection and
the best of them is reported, so the numbers are exploratory evidence of an
association, not a validated generalisation.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue126" / "metric_round" / "metric_comparison_107.csv"

#: the families, so the table reads as a finding rather than a list
FAMILY = {
    "hbaa": "behaviour contrast (the family that works)",
    "q_h_level1": "behaviour contrast (the family that works)",
    "hbaa_anchor": "behaviour contrast, control of the query point",
    "hbaa_pair_accuracy": "behaviour contrast, control of the reduction",
    "hbaa_euclid": "behaviour contrast, Euclidean control",
    "hbaa_flat": "behaviour contrast, nearly flat control",
    "hbaa_midpoint": "behaviour contrast, control of the query point",
    "hbaa_weighted": "behaviour contrast, control of the levels",
    "hbaa_size_matched": "behaviour contrast, control of the negatives",
    "hbaa_validation": "behaviour contrast, the held-out probe",
    "hbaa_offcell_expected_rank": "behaviour contrast, the removal variant",
    "hbaa_fullneg_rank_excess": "behaviour contrast, the global ranking variant",
    "behaviour_prefix_preservation": "behaviour relation (the behaviour half of direction 3)",
    "geometry_reconstruction_fidelity": "geometry half of direction 3",
    "joint_behaviour_over_geometry": "direction 3, the two halves together",
    "delta_four_point": "the delta-guided four-point consistency at c_delta",
    "delta_four_point_half": "the same at 0.5 c_delta",
    "delta_four_point_double": "the same at 2 c_delta",
    "delta_four_point_flat": "the same at a nearly flat curvature",
    "delta_four_point_euclidean": "the four-point consistency, Euclidean control",
    "delta_four_point_permuted": "the four-point consistency, permuted control",
    "gromov_prefix_alignment": "the Gromov-Prefix alignment (corrected)",
    "gromov_prefix_alignment_flat": "the same at a nearly flat curvature",
    "gromov_prefix_alignment_euclidean": "the same with a Euclidean depth",
    "hyperbolic_tree_distortion": "the prefix tree against the geodesic distance",
    "gromov_delta_tree_agreement": "the Gromov delta as a three-point tree test",
    "behaviour_hierarchy_agreement": "direction 2, the behaviour hierarchy",
    "hyp_cell_expected_rank": "the true cell's tempered mass",
    "hyp_cell_rank_lift": "the true cell's rank",
    "hyp_adj_r2_l1": "partition geometry",
    "hyp_bucket_reach_lift_l1": "partition geometry",
    "hyp_frechet_cover_radius_l1": "partition geometry",
    "hyp_voronoi_margin_l1": "partition geometry",
    "hyp_hsoftmax_path": "rewarding an answer already in hand",
    "hyp_hsoftmax_path_null": "rewarding an answer already in hand",
    "hbaa_emission_hit10": "rewarding an answer already in hand",
    "hbaa_size_matched_kept": "a diagnostic, not a metric",
    "hbaa_teacher_curvature": "behaviour contrast, the teacher's own curvature",
    "hbaa_recency": "behaviour contrast, control of the query point",
    "hbaa_cell_retrieval_difficulty": "rewarding an answer already in hand, negated",
    "hbaa_nextitem": "behaviour contrast, a next-item teacher",
    "hbaa_size_matched": "behaviour contrast, control of the negatives",
    "hbaa_levels": "behaviour contrast, control of the query point",
    "hyperbolic_cone_consistency": "behaviour contrast, control of the query point",
}


def main() -> int:
    best = {}
    for path in sorted((REPO / "results" / "research" / "issue126").glob("metric_bench*.json")):
        try:
            report = json.loads(path.read_text())
        except ValueError:
            continue
        for result in report.get("results", []):
            name = result["metric"]
            entry = {"spearman": result["spearman"], "p": result["p"],
                     "ci95": result["bootstrap_ci95"], "eligible": result.get("eligible"),
                     "pairs": report.get("pairs")}
            # the largest sample a metric was measured on wins
            if name not in best or (entry["pairs"] or 0) >= (best[name]["pairs"] or 0):
                best[name] = entry
    rows = []
    for name, entry in sorted(best.items(), key=lambda item: -item[1]["spearman"]):
        rows.append({
            "metric": name,
            "family": FAMILY.get(name, "unclassified"),
            "spearman": round(entry["spearman"], 4),
            "p": entry["p"],
            "ci95_low": round(entry["ci95"][0], 4),
            "ci95_high": round(entry["ci95"][1], 4),
            "eligible": bool(entry["eligible"]),
            "pairs": entry["pairs"],
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} metrics on the {rows[0]['pairs']} confirmed pairs\n")
    print(f"{'metric':34s} {'rho':>8s} {'p':>10s}  CI")
    for row in rows:
        print(f"{row['metric']:34s} {row['spearman']:+8.4f} {row['p']:10.3g}  "
              f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]"
              f"{'  ELIGIBLE' if row['eligible'] else ''}")
    print(f"\nwritten: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
