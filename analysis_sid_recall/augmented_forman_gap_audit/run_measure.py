"""Issue #137 audit measurement: corrected semantics over the 108 SID arms.

Reuses #136's frozen graph/curvature primitives, adds the corrected
edge-negative/group-negative features, the triangle-increment decomposition and
the null sensitivity, and never writes into the #136 directory.

Outputs: `features_by_level.csv|json`, `support.csv`,
`feature_semantics_audit.csv`, `new_vs_old.csv`, `timing.json`,
`completeness.json`.
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE),
             str(REPO / "analysis_sid_recall" / "layerwise_curvature"),
             str(REPO / "analysis_sid_recall" / "full_census_geometry"),
             str(REPO / "analysis_sid_recall" / "intra_inter_geometry"),
             str(REPO / "analysis_sid_recall" / "augmented_forman_gap")):
    if path not in sys.path:
        sys.path.insert(0, path)

import af3  # noqa: E402
import audit_features  # noqa: E402
import features_lw as features  # noqa: E402
import geom  # noqa: E402

SEED = 20261103
DESIGNS = ("global", "degree_stratified")
REPLICATES = 20
OLD = REPO / "analysis_sid_recall" / "augmented_forman_gap" / "features_by_level.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps({
        "seed": SEED, "null_replicates": REPLICATES, "null_designs": list(DESIGNS),
        "source": "reuses the #136 TRAIN graph builder, item mapping and frozen curvature cache",
        "semantics": {"group_mean_negative_fraction": "groups with MEAN AF3 < 0 / eligible groups",
                      "edge_negative_fraction": "negative AF3 edges / all eligible induced edges",
                      "mean_group_negative_edge_fraction": "unweighted mean of per-group edge fractions"},
        "triangle_identity": "gap_af3 - gap_forman == 3*(T_within_mean - T_cross_mean)"},
        indent=1) + "\n", encoding="utf-8")

    pairs = features.load_pairs()
    excluded = [pair for pair in pairs if pair["tier"] == "HASH_MISMATCH"]
    pairs = [pair for pair in pairs if pair["tier"] != "HASH_MISMATCH"]
    if args.limit:
        pairs = pairs[:args.limit]
    nodes, index, weighted, _ = features.behaviour_graph()
    binary = geom.binary_graph(weighted)
    degree_b = af3.degree_of(binary)
    started = time.time()
    curvature = af3.edge_curvatures(binary)
    global_seconds = time.time() - started

    old_rows = {}
    if OLD.is_file():
        for raw in csv.DictReader(OLD.open()):
            old_rows[raw["sid_sha256"]] = raw
    checkpoint = out / "_checkpoint.jsonl"
    done, records = set(), []
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text().splitlines():
            try:
                record = json.loads(line)
            except Exception:
                continue
            records.append(record)
            done.add(record["sid_sha256"])
    support_path = out / "_support_checkpoint.csv"
    support_rows = []
    if args.resume and support_path.is_file():
        support_rows = list(csv.DictReader(support_path.open()))

    timings = {"per_sid": [], "global_curvature_seconds": round(global_seconds, 2),
               "phases": {"intra": 0.0, "quotient": 0.0, "gap": 0.0, "nulls": 0.0}}
    for number, pair in enumerate(pairs):
        if pair["sid_sha256"] in done:
            continue
        table = json.loads(pair["payload"])
        record = {"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                  "series": pair["series"], "provenance_tier": pair["tier"],
                  "commit_tier": pair.get("commit_tier"),
                  "recall": float(pair["test_recall@10"])}
        sid_started = time.time()
        for level in (1, 2, 3):
            labels = features.labels_for(table, level, index, nodes)
            phase = time.time()
            intra = audit_features.intra_features(binary, labels)
            timings["phases"]["intra"] += time.time() - phase
            for key in ("groups", "eligible_groups", "singleton_groups", "no_edge_groups",
                        "edges", "edge_counts_consistent", "group_mean_negative_fraction",
                        "edge_negative_fraction", "mean_group_negative_edge_fraction",
                        "forman_group_mean_negative_fraction",
                        "forman_edge_negative_fraction",
                        "forman_mean_group_negative_edge_fraction",
                        "af3_mean", "af3_median", "af3_p10", "af3_p90",
                        "af3_edge_weighted", "forman_mean", "triangles_mean",
                        "af3_minus_forman", "status"):
                record[f"intra_{key}_l{level}"] = intra.get(key)
            record[f"occupied_l{level}"] = int(np.unique(labels[labels >= 0]).size)
            record[f"singleton_share_l{level}"] = (
                intra["singleton_groups"] / intra["groups"] if intra["groups"] else None)
            phase = time.time()
            matrix = geom.quotient(binary, labels)
            inter = audit_features.quotient_features(matrix)
            timings["phases"]["quotient"] += time.time() - phase
            for key, value in inter.items():
                record[f"inter_{key}_l{level}"] = value
            record[f"quotient_nodes_l{level}"] = int(matrix.shape[0]) if matrix.nnz else 0
            phase = time.time()
            gap = audit_features.gap_block(curvature, labels)
            timings["phases"]["gap"] += time.time() - phase
            for key in ("status", "gap_af3", "gap_forman", "within_mean", "cross_mean",
                        "within_median", "cross_median", "within_negative_fraction",
                        "cross_negative_fraction", "forman_within_mean",
                        "forman_cross_mean", "triangles_within_mean",
                        "triangles_cross_mean", "triangle_increment", "edge_retention",
                        "edges_total", "edges_unlabelled", "edges_eligible",
                        "edges_within", "edges_cross"):
                record[f"gap_{key}_l{level}"] = gap.get(key)
            phase = time.time()
            for position, design in enumerate(DESIGNS):
                null = af3.gap_null(curvature["rows"], curvature["cols"],
                                    curvature["af3"], labels, degree_b, design,
                                    REPLICATES,
                                    SEED + 1000 * level + 100 * position + number)
                record[f"gap_af3_nullmean_{design}_l{level}"] = null.get("mean")
                record[f"gap_af3_nullsd_{design}_l{level}"] = null.get("sd")
                real = gap.get("gap_af3")
                if (real is None or not np.isfinite(real)
                        or null.get("status") != "DEFINED"):
                    record[f"gap_af3_z_{design}_l{level}"] = None
                    record[f"gap_af3_centered_{design}_l{level}"] = None
                else:
                    record[f"gap_af3_centered_{design}_l{level}"] = real - null["mean"]
                    record[f"gap_af3_z_{design}_l{level}"] = (
                        (real - null["mean"]) / null["sd"] if null["sd"] > 0 else None)
            timings["phases"]["nulls"] += time.time() - phase
            support_rows.append({"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                                 "level": level, "occupied": record[f"occupied_l{level}"],
                                 "groups": intra["groups"],
                                 "eligible_groups": intra["eligible_groups"],
                                 "singleton_groups": intra["singleton_groups"],
                                 "no_edge_groups": intra["no_edge_groups"],
                                 "intra_edges": intra["edges"],
                                 "edge_counts_consistent": intra["edge_counts_consistent"],
                                 "quotient_nodes": record[f"quotient_nodes_l{level}"],
                                 "quotient_edges": inter.get("edges", 0),
                                 "gap_edges_total": gap["edges_total"],
                                 "gap_edges_within": gap["edges_within"],
                                 "gap_edges_cross": gap["edges_cross"],
                                 "gap_edges_unlabelled": gap["edges_unlabelled"],
                                 "gap_status": gap["status"]})
        timings["per_sid"].append(round(time.time() - sid_started, 2))
        records.append(record)
        with checkpoint.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        new_support = [row for row in support_rows if row["sid_sha256"] == pair["sid_sha256"]]
        fields = []
        for row in new_support:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with support_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if support_path.stat().st_size == 0:
                writer.writeheader()
            for row in new_support:
                writer.writerow(row)
        print(f"  {pair['variant']} {timings['per_sid'][-1]:.1f}s", flush=True)

    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with (out / "features_by_level.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    (out / "features_by_level.json").write_text(json.dumps(records, indent=1) + "\n",
                                                encoding="utf-8")
    support_fields = []
    for row in support_rows:
        for key in row:
            if key not in support_fields:
                support_fields.append(key)
    with (out / "support.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=support_fields, extrasaction="ignore")
        writer.writeheader()
        for row in support_rows:
            writer.writerow(row)

    semantics = ["sid_sha256,level,eligible_groups,edges,af3_mean,"
                 "group_mean_negative_fraction,edge_negative_fraction,"
                 "mean_group_negative_edge_fraction,"
                 "forman_group_mean_negative_fraction,forman_edge_negative_fraction"]
    for record in records:
        for level in (1, 2, 3):
            semantics.append(",".join([
                str(record["sid_sha256"]), str(level),
                str(record.get(f"intra_eligible_groups_l{level}", "")),
                str(record.get(f"intra_edges_l{level}", "")),
                str(record.get(f"intra_af3_mean_l{level}", "")),
                str(record.get(f"intra_group_mean_negative_fraction_l{level}", "")),
                str(record.get(f"intra_edge_negative_fraction_l{level}", "")),
                str(record.get(f"intra_mean_group_negative_edge_fraction_l{level}", "")),
                str(record.get(f"intra_forman_group_mean_negative_fraction_l{level}", "")),
                str(record.get(f"intra_forman_edge_negative_fraction_l{level}", ""))]))
    (out / "feature_semantics_audit.csv").write_text("\n".join(semantics) + "\n",
                                                     encoding="utf-8")

    comparison = ["feature,level,old_value,new_value,changed"]
    shared = [("gap_af3", "gap_af3"), ("gap_forman", "gap_forman"),
              ("gap_edges_within", "edges_within"), ("gap_edges_cross", "edges_cross"),
              ("intra_af3_mean", "intra_af3_mean"),
              ("intra_forman_mean", "intra_forman_mean"),
              ("inter_af3_mean", "inter_af3_mean"),
              ("inter_forman_mean", "inter_forman_mean")]
    for record in records:
        old = old_rows.get(record["sid_sha256"])
        for level in (1, 2, 3):
            for old_key, new_key in shared:
                old_value = old.get(f"{old_key}_l{level}") if old else None
                new_value = record.get(f"{new_key}_l{level}")
                try:
                    changed = (abs(float(old_value) - float(new_value)) > 1e-9
                               if old_value not in (None, "") and new_value is not None
                               else True)
                except (TypeError, ValueError):
                    changed = True
                comparison.append(f"{old_key},L{level},{old_value},{new_value},{changed}")
    (out / "new_vs_old.csv").write_text("\n".join(comparison) + "\n", encoding="utf-8")

    per_sid = np.asarray(timings["per_sid"], dtype=float)
    timing = {"per_sid": timings["per_sid"],
              "global_curvature_seconds": timings["global_curvature_seconds"],
              "phases_seconds": {key: round(value, 2)
                                 for key, value in timings["phases"].items()},
              "median_sid_seconds": float(np.median(per_sid)) if per_sid.size else None,
              "max_sid_seconds": float(per_sid.max()) if per_sid.size else None,
              "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)}
    (out / "timing.json").write_text(json.dumps(timing, indent=1) + "\n", encoding="utf-8")
    completeness = {"rows": len(records), "excluded_hash_mismatch": len(excluded),
                    "global_edges": int(curvature["rows"].size),
                    "tiers": {tier: sum(1 for pair in pairs if pair["tier"] == tier)
                              for tier in ("HASH_PROVEN", "RECORD_HASH_MATCH", "MISSING")},
                    "levels": [1, 2, 3],
                    "nulls": {"replicates": REPLICATES, "designs": list(DESIGNS)},
                    "identity": "gap_af3-gap_forman=3*(T_within-T_cross) checked per row",
                    "source_comparison": "new_vs_old.csv compares shared fields with #136"}
    (out / "completeness.json").write_text(json.dumps(completeness, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"written {len(records)} rows; median {timing['median_sid_seconds']}s/SID",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
