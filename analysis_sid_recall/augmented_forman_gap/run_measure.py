"""Issue #136 runner: AF3 on induced, quotient and original-edge graphs.

One global original-graph curvature pass is cached; every SID/level reuses it and
only its prefix membership changes. Checkpoint/resume per SID, deterministic nulls.

Outputs: `features_by_level.csv|json`, `support.csv`, `timing.json`,
`completeness.json`, `_checkpoint.jsonl`.
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
             str(REPO / "analysis_sid_recall" / "intra_inter_geometry")):
    if path not in sys.path:
        sys.path.insert(0, path)

import af3  # noqa: E402
import features_lw as features  # noqa: E402
import geom  # noqa: E402

SEED = 20261103
DESIGNS = ("global", "degree_stratified")
REPLICATES = 20


def summarise_curvature(rows: np.ndarray, curvature: dict) -> dict:
    af3_stats = af3._summarise(curvature["af3"])
    f_stats = af3._summarise(curvature["forman"])
    degree = curvature["degree"]
    nodes = int(degree.size)
    edges = int(rows.size)
    wedges = float((degree.astype(np.float64) * (degree - 1.0)).sum() / 2.0)
    triangles = float(curvature["triangles"].sum() / 3.0)
    return {"af3_mean": af3_stats["mean"], "af3_median": af3_stats["median"],
            "af3_p10": af3_stats["p10"], "af3_p90": af3_stats["p90"],
            "af3_negative_fraction": af3_stats["negative_fraction"],
            "forman_mean": f_stats["mean"], "forman_median": f_stats["median"],
            "forman_negative_fraction": f_stats["negative_fraction"],
            "triangles": triangles, "edges": edges, "nodes": nodes,
            "density": edges / (nodes * (nodes - 1) / 2.0) if nodes > 1 else None,
            "transitivity": (3.0 * triangles / wedges) if wedges > 0 else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {"seed": SEED, "null_replicates": REPLICATES, "null_designs": list(DESIGNS),
                "convention": json.loads((HERE / "definitions.json").read_text())
                if (HERE / "definitions.json").is_file() else None}
    (out / "settings.json").write_text(json.dumps(settings, indent=1) + "\n", encoding="utf-8")

    pairs = features.load_pairs()
    excluded = [pair for pair in pairs if pair["tier"] == "HASH_MISMATCH"]
    pairs = [pair for pair in pairs if pair["tier"] != "HASH_MISMATCH"]
    if args.limit:
        pairs = pairs[:args.limit]
    nodes, index, weighted, _ = features.behaviour_graph()
    binary = geom.binary_graph(weighted)
    degree_b = af3.degree_of(binary)
    started = time.time()
    global_curvature = af3.edge_curvatures(binary)
    global_seconds = time.time() - started

    checkpoint = out / "_checkpoint.jsonl"
    done = set()
    records = []
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
            groups = af3.group_af3(binary, labels)
            timings["phases"]["intra"] += time.time() - phase
            for key in ("groups", "eligible_groups", "singleton_groups", "no_edge_groups",
                        "edges", "af3_mean", "af3_median", "af3_p10", "af3_p90",
                        "af3_negative_fraction", "af3_edge_weighted", "forman_mean",
                        "group_coverage", "status"):
                record[f"intra_{key}_l{level}"] = groups.get(key)
            phase = time.time()
            matrix = geom.quotient(binary, labels)
            quotient = af3.edge_curvatures(matrix) if matrix.nnz else None
            timings["phases"]["quotient"] += time.time() - phase
            if quotient is not None:
                block = summarise_curvature(quotient["rows"], quotient)
                for key, value in block.items():
                    record[f"inter_{key}_l{level}"] = value
                record[f"inter_status_l{level}"] = "DEFINED"
            else:
                record[f"inter_status_l{level}"] = "UNDEFINED_EMPTY_QUOTIENT"
            phase = time.time()
            gap = af3.gap_metrics(global_curvature["rows"], global_curvature["cols"],
                                  global_curvature["af3"], labels)
            gap_f = af3.gap_metrics(global_curvature["rows"], global_curvature["cols"],
                                    global_curvature["forman"], labels)
            timings["phases"]["gap"] += time.time() - phase
            record[f"gap_af3_l{level}"] = gap.get("gap")
            record[f"gap_forman_l{level}"] = gap_f.get("gap")
            for key in ("status", "within_mean", "cross_mean", "within_median",
                        "cross_median", "within_negative_fraction",
                        "cross_negative_fraction", "edges_total", "edges_unlabelled",
                        "edges_eligible", "edges_within", "edges_cross"):
                record[f"gap_{key}_l{level}"] = gap.get(key)
            record[f"edge_retention_l{level}"] = (
                gap["edges_within"] / gap["edges_eligible"]
                if gap["edges_eligible"] else None)
            record[f"gap_forman_within_l{level}"] = gap_f.get("within_mean")
            record[f"gap_forman_cross_l{level}"] = gap_f.get("cross_mean")
            phase = time.time()
            for design in DESIGNS:
                null = af3.gap_null(global_curvature["rows"], global_curvature["cols"],
                                    global_curvature["af3"], labels, degree_b, design,
                                    REPLICATES,
                                    SEED + 1000 * level + 100 * DESIGN_INDEX[design]
                                    + number)
                record[f"gap_af3_nullmean_{design}_l{level}"] = null.get("mean")
                record[f"gap_af3_nullsd_{design}_l{level}"] = null.get("sd")
                real = gap.get("gap")
                if (real is None or not np.isfinite(real)
                        or null.get("status") != "DEFINED"):
                    record[f"gap_af3_z_{design}_l{level}"] = None
                elif null["sd"] > 0:
                    record[f"gap_af3_z_{design}_l{level}"] = (real - null["mean"]) / null["sd"]
                else:
                    record[f"gap_af3_z_{design}_l{level}"] = None
            timings["phases"]["nulls"] += time.time() - phase
            support_rows.append({"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                                 "level": level, "items": int((labels >= 0).sum()),
                                 "groups": groups.get("groups"),
                                 "eligible_groups": groups.get("eligible_groups"),
                                 "singleton_groups": groups.get("singleton_groups"),
                                 "no_edge_groups": groups.get("no_edge_groups"),
                                 "intra_edges": groups.get("edges"),
                                 "quotient_nodes": int(matrix.shape[0]) if matrix.nnz else 0,
                                 "quotient_edges": int(matrix.nnz // 2),
                                 "edges_total": gap["edges_total"],
                                 "edges_unlabelled": gap["edges_unlabelled"],
                                 "edges_within": gap["edges_within"],
                                 "edges_cross": gap["edges_cross"]})
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
    support_rows = [row for row in support_rows]
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
                    "global_edges": int(global_curvature["rows"].size),
                    "tiers": {tier: sum(1 for pair in pairs if pair["tier"] == tier)
                              for tier in ("HASH_PROVEN", "RECORD_HASH_MATCH", "MISSING")},
                    "exactness": {
                        "intra": "every eligible induced edge, per group; zero-edge groups "
                                 "are UNDEFINED_NO_EDGES and counted",
                        "inter": "every quotient edge",
                        "gap": "every original-graph edge classified once; unlabelled edges "
                               "counted separately and never cross",
                        "af3": "triangle-only convention; edges without triangles are 0"},
                    "nulls": {"replicates": REPLICATES, "designs": list(DESIGNS)},
                    "settings": settings}
    (out / "completeness.json").write_text(json.dumps(completeness, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"written {len(records)} rows; median {timing['median_sid_seconds']}s/SID",
          flush=True)
    return 0


DESIGN_INDEX = {name: index for index, name in enumerate(DESIGNS)}

if __name__ == "__main__":
    raise SystemExit(main())
