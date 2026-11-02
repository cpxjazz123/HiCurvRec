"""Issue #133 measurement runner: corrected intra-group and new inter-group geometry.

Per (SID, level): binary TRAIN-only graph B, every occupied prefix group (no cap),
singleton groups kept in the census but excluded from the intra-group averages,
the quotient graph Q (one node per occupied prefix, crossing edges only),
quotient topology and combinatorial Forman curvature on **every** quotient edge,
and Ollivier-Ricci on a declared sampled edge set (labelled SAMPLED_ESTIMATE with
its coverage). Checkpoint/resume is per SID and idempotent.

Outputs: `features_by_level.csv|json`, `support_L1_L2_L3.csv`, `timing.json`,
`completeness.json`, `_checkpoint.jsonl`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE),
             str(REPO / "analysis_sid_recall" / "layerwise_curvature"),
             str(REPO / "analysis_sid_recall" / "full_census_geometry")):
    if path not in sys.path:
        sys.path.insert(0, path)

import features_lw as features  # noqa: E402
import geom  # noqa: E402

SETTINGS = {
    "intra_eligibility": "groups with >= 2 items enter the averages; singletons are "
                         "counted in the census and the quotient only",
    "triangle_method": "exact sparse (A@A).multiply(A).sum()//6, int32 accumulator",
    "triangle_nnz_max": geom.TRIANGLE_NNZ_MAX,
    "conductance": "binary cut / min(binary volume of S, volume of complement); "
                   "UNDEFINED_ZERO_VOLUME when the denominator is zero",
    "quotient": "one node per occupied prefix, edge iff a source edge crosses, no "
                "self-loops, unweighted",
    "forman": "4 - deg(u) - deg(v) on every quotient edge",
    "orc": {"lazy": 0.5, "ground_metric": "0/1/2-hop (exact transport by max flow)",
            "edges": geom.ORC_EDGES, "label": "SAMPLED_ESTIMATE"},
    "gromov": {"exact_max_component": geom.GROMOV_MAX,
               "above": "BLOCKED_NOT_FULL_CENSUS"},
    "nulls": "not used: #133 asks for the raw Spearman study",
    "seed": 20261102,
}


def support_checkpoint_fields(rows: list[dict]) -> list[str]:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _aggregate(metrics: list[dict], sizes: np.ndarray, key: str) -> dict:
    values = np.asarray([entry[key] if entry.get(key) is not None else np.nan
                         for entry in metrics], dtype=float)
    finite = np.isfinite(values)
    return {"groups": int(finite.sum()),
            "group_coverage": float(finite.mean()) if values.size else 0.0,
            "group_mean": float(values[finite].mean()) if finite.any() else None,
            "item_weighted": float(np.average(values[finite], weights=sizes[finite]))
            if finite.any() else None,
            "median": float(np.median(values[finite])) if finite.any() else None,
            "p90": float(np.quantile(values[finite], 0.9)) if finite.any() else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(SETTINGS, indent=1) + "\n", encoding="utf-8")

    pairs = features.load_pairs()
    excluded = [pair for pair in pairs if pair["tier"] == "HASH_MISMATCH"]
    pairs = [pair for pair in pairs if pair["tier"] != "HASH_MISMATCH"]
    if args.limit:
        pairs = pairs[:args.limit]
    nodes, index, weighted, _ = features.behaviour_graph()
    binary = geom.binary_graph(weighted)
    degree_b = np.asarray(binary.sum(axis=1)).ravel()
    total_volume = float(degree_b.sum())
    checkpoint = out / "_checkpoint.jsonl"
    done = set()
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text().splitlines():
            try:
                done.add(json.loads(line)["sid_sha256"])
            except Exception:
                continue

    support_checkpoint = out / "_support_checkpoint.csv"
    support_rows = []
    if args.resume and support_checkpoint.is_file():
        support_rows = list(csv.DictReader(support_checkpoint.open()))
    timings = {"per_sid": [], "phases": {"intra": 0.0, "quotient": 0.0, "orc": 0.0}}
    for number, pair in enumerate(pairs):
        if pair["sid_sha256"] in done:
            continue
        table = json.loads(pair["payload"])
        record = {"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                  "series": pair["series"], "provenance_tier": pair["tier"],
                  "recall": float(pair["test_recall@10"])}
        started = time.time()
        for level in (1, 2, 3):
            labels = features.labels_for(table, level, index, nodes)
            groups = geom.groups_of(labels)
            sizes = np.asarray([group.size for group in groups], dtype=float)
            phase = time.time()
            intra = [geom.intra_metrics(binary, degree_b, group, total_volume)
                     if group.size >= 2 else
                     geom.singleton_metrics(binary, degree_b, int(group[0]), total_volume)
                     for group in groups]
            timings["phases"]["intra"] += time.time() - phase
            multi = [entry for entry in intra if entry["size"] >= 2]
            multi_sizes = np.asarray([entry["size"] for entry in multi], dtype=float)
            support_rows.append({"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                                 "level": level,
                                 "occupied": int(sizes.size),
                                 "singleton": int((sizes == 1).sum()),
                                 "multi": int((sizes >= 2).sum()),
                                 "items": int(sizes.sum()),
                                 "multi_items": int(sizes[sizes >= 2].sum()),
                                 "multi_item_coverage": float(sizes[sizes >= 2].sum() / sizes.sum())})
            for key in ("transitivity", "cycle_rank_normalised",
                        "largest_component_fraction", "conductance",
                        "internal_edges", "boundary_edges", "triangles",
                        "components"):
                block = _aggregate(multi, multi_sizes, key)
                record[f"intra_{key}_l{level}"] = block["group_mean"]
                record[f"intra_{key}_item_l{level}"] = block["item_weighted"]
                record[f"intra_{key}_p90_l{level}"] = block["p90"]
                record[f"intra_{key}_coverage_l{level}"] = block["group_coverage"]
            record[f"intra_groups_l{level}"] = len(multi)
            record[f"intra_singleton_excluded_l{level}"] = int((sizes == 1).sum())
            record[f"intra_multi_item_coverage_l{level}"] = float(
                sizes[sizes >= 2].sum() / sizes.sum())
            phase = time.time()
            matrix = geom.quotient(binary, labels)
            metrics = geom.quotient_metrics(matrix)
            timings["phases"]["quotient"] += time.time() - phase
            for key in ("nodes", "edges", "components", "largest_component_fraction",
                        "density", "degree_median", "degree_p90", "degree_max",
                        "triangles", "transitivity", "cycle_rank_per_node",
                        "forman_mean", "forman_median", "forman_negative_fraction"):
                record[f"quot_{key}_l{level}"] = metrics.get(key)
            record[f"quot_triangle_status_l{level}"] = metrics.get("triangle_status")
            record[f"quot_singleton_nodes_l{level}"] = int((sizes == 1).sum()) / max(sizes.size, 1)
            gromov = geom.quotient_gromov(matrix)
            record[f"quot_gromov_status_l{level}"] = gromov["status"]
            record[f"quot_gromov_delta_l{level}"] = gromov.get("delta_max")
            record[f"quot_gromov_components_blocked_l{level}"] = gromov.get("components_blocked")
            phase = time.time()
            orc = geom.orc_sampled(matrix, edges=geom.ORC_EDGES,
                                   seed=SETTINGS["seed"] + 97 * level)
            timings["phases"]["orc"] += time.time() - phase
            for key in ("mean", "median", "sd", "negative_fraction", "edge_coverage",
                        "edges_total", "edges_measured", "status"):
                record[f"orc_{key}_l{level}"] = orc.get(key)
        timings["per_sid"].append(round(time.time() - started, 2))
        records.append(record)
        with checkpoint.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        new_support = [row for row in support_rows
                       if row["sid_sha256"] == pair["sid_sha256"]]
        with support_checkpoint.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(support_checkpoint_fields(new_support)),
                                    extrasaction="ignore")
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerows(new_support)
        print(f"  {number + 1}/{len(pairs)} {pair['variant']} {timings['per_sid'][-1]:.1f}s",
              flush=True)

    if args.resume and checkpoint.is_file():
        by_sha = {}
        for line in checkpoint.read_text().splitlines():
            try:
                item = json.loads(line)
            except Exception:
                continue
            by_sha[item["sid_sha256"]] = item
        records = list(by_sha.values())

    import csv
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with (out / "features_by_level.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (out / "features_by_level.json").write_text(json.dumps(records, indent=1) + "\n",
                                                encoding="utf-8")
    support_fields = []
    for row in support_rows:
        for key in row:
            if key not in support_fields:
                support_fields.append(key)
    with (out / "support_L1_L2_L3.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=support_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(support_rows)
    timings["phases"] = {key: round(value, 1) for key, value in timings["phases"].items()}
    timings["median_sid_seconds"] = float(np.median(timings["per_sid"])) if timings["per_sid"] else None
    timings["max_sid_seconds"] = float(max(timings["per_sid"])) if timings["per_sid"] else None
    (out / "timing.json").write_text(json.dumps(timings, indent=1) + "\n", encoding="utf-8")
    completeness = {
        "rows": len(records), "excluded_hash_mismatch": len(excluded),
        "caps": {"group_count": None, "group_size": None, "item_count": None},
        "statuses": {
            "intra": "exact for every group; singletons counted, not averaged",
            "triangles": f"exact sparse; BLOCKED above nnz {geom.TRIANGLE_NNZ_MAX}",
            "quotient": "complete, no sampling; Forman on every edge",
            "orc": f"SAMPLED_ESTIMATE with {geom.ORC_EDGES} edges",
            "gromov": f"exact on components <= {geom.GROMOV_MAX}, else BLOCKED"},
    }
    (out / "completeness.json").write_text(json.dumps(completeness, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"written {len(records)} rows; median {timings['median_sid_seconds']}s/SID", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
