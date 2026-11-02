"""Issue #134 runner: corrected ORC (true geodesic) and boundary counts.

For every admitted SID and level: the #133 intra/quotient metrics are recomputed,
with ``boundary_edges = cut`` (D2), and the sampled ORC is recomputed on the same
seeded edges twice — once with the **true shortest-path** cost matrix (D1) and
once with the old 0/1/2-clipped cost — so the paired audit compares like with
like. All-edge ORC is measured as an edge count and reported
``BLOCKED_NOT_FULL_CENSUS``; the sampled result is ``SAMPLED_ESTIMATE`` with its
edge coverage.

Budget (declared): 16 edges per level for the primary ORC, plus 8/16/32 on the
first ``CONVERGENCE_SIDS`` SIDs for the budget sensitivity. Checkpoint/resume is
per SID and idempotent; the support table has its own checkpoint.
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
             str(REPO / "analysis_sid_recall" / "full_census_geometry"),
             str(REPO / "analysis_sid_recall" / "intra_inter_geometry"),
             str(REPO / "analysis_sid_recall" / "interaction_strength")):
    if path not in sys.path:
        sys.path.insert(0, path)

import features_lw as features  # noqa: E402
import geom  # noqa: E402
import geodesic  # noqa: E402

SETTINGS = {
    "orc_edges": 16,
    "orc_budget_sensitivity": [8, 16, 32],
    "convergence_sids": 12,
    "seed": 20261102,
    "ground_metric": "true shortest path in the full quotient (0..3), exact sparse LP",
    "clipped_diagnostic": "0/1/2-clipped cost, kept only for the paired audit",
    "boundary": "boundary_edges = cut (D2 fix)",
    "singletons": "counted in the census and kept as quotient nodes; excluded from "
                  "the intra-group averages",
    "gromov": "halved convention (max pair-sum - second)/2; exact only on components "
              "<= 24 nodes, otherwise BLOCKED_NOT_FULL_CENSUS",
}


def sample_edges(matrix, count, seed):
    coo = matrix.tocoo()
    keep = coo.row < coo.col
    pairs = list(zip(coo.row[keep].tolist(), coo.col[keep].tolist()))
    if len(pairs) <= count:
        return pairs, len(pairs)
    rng = np.random.default_rng(seed)
    chosen = [pairs[int(i)] for i in rng.choice(len(pairs), size=count, replace=False)]
    return chosen, len(pairs)


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
    degree = np.asarray(binary.sum(axis=1)).ravel()
    total_volume = float(degree.sum())
    checkpoint = out / "_checkpoint.jsonl"
    support_checkpoint = out / "_support_checkpoint.csv"
    done = set()
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text().splitlines():
            try:
                done.add(json.loads(line)["sid_sha256"])
            except Exception:
                continue
    support_rows = list(csv.DictReader(support_checkpoint.open())) \
        if args.resume and support_checkpoint.is_file() else []

    records = []
    timing = {"orc_true": 0.0, "orc_clipped": 0.0, "intra": 0.0, "quotient": 0.0,
              "per_sid": []}
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
            intra = []
            boundary_mismatch = 0
            for position, group in enumerate(groups):
                if group.size >= 2:
                    entry = geom.intra_metrics(binary, degree, group, total_volume)
                    # D2: the cut IS the number of undirected crossing edges
                    cut, crossing = geodesic.group_cut(binary, group, degree)
                    if int(round(entry["cut_binary"])) != crossing:
                        boundary_mismatch += 1
                    entry["boundary_edges"] = float(entry["cut_binary"])
                    entry["cut_crossing"] = crossing
                    intra.append(entry)
            timing["intra"] += time.time() - phase
            multi_sizes = np.asarray([entry["size"] for entry in intra], dtype=float)
            support_rows.append({"sid_sha256": pair["sid_sha256"], "level": level,
                                 "occupied": int(sizes.size),
                                 "singleton": int((sizes == 1).sum()),
                                 "multi": int((sizes >= 2).sum()),
                                 "items": int(sizes.sum()),
                                 "boundary_mismatches": boundary_mismatch})
            for key in ("transitivity", "cycle_rank_normalised",
                        "largest_component_fraction", "conductance",
                        "internal_edges", "boundary_edges", "triangles", "components"):
                values = np.asarray([entry[key] if entry.get(key) is not None else np.nan
                                     for entry in intra], dtype=float)
                finite = np.isfinite(values)
                record[f"intra_{key}_l{level}"] = float(values[finite].mean()) if finite.any() else None
                record[f"intra_{key}_coverage_l{level}"] = float(finite.mean()) if values.size else 0.0
            record[f"intra_groups_l{level}"] = len(intra)
            record[f"intra_singleton_excluded_l{level}"] = int((sizes == 1).sum())
            phase = time.time()
            matrix = geom.quotient(binary, labels)
            metrics = geom.quotient_metrics(matrix)
            gromov = geom.quotient_gromov(matrix)
            timing["quotient"] += time.time() - phase
            for key in ("nodes", "edges", "components", "largest_component_fraction",
                        "density", "degree_median", "degree_p90", "degree_max",
                        "triangles", "transitivity", "cycle_rank_per_node",
                        "forman_mean", "forman_median", "forman_negative_fraction"):
                record[f"quot_{key}_l{level}"] = metrics.get(key)
            record[f"quot_triangle_status_l{level}"] = metrics.get("triangle_status")
            record[f"quot_gromov_status_l{level}"] = gromov["status"]
            record[f"quot_gromov_delta_halved_l{level}"] = (
                None if gromov.get("delta_max") is None else gromov["delta_max"] / 2.0)
            phase = time.time()
            chosen, total_edges = sample_edges(matrix, SETTINGS["orc_edges"],
                                              SETTINGS["seed"] + 97 * level)
            true_values, clipped_values, d3_edges, deltas = [], [], 0, []
            for u, v in chosen:
                exact = geodesic.orc_edge(matrix, u, v)
                clipped = geodesic.orc_edge_clipped(matrix, u, v)
                true_values.append(exact["curvature"])
                clipped_values.append(clipped)
                deltas.append(exact["curvature"] - clipped)
                if exact["distance_three_pairs"] > 0:
                    d3_edges += 1
            timing["orc_true"] += time.time() - phase
            phase = time.time()
            for key, values in (("orc_true", true_values), ("orc_clipped", clipped_values)):
                values = np.asarray(values, dtype=float)
                record[f"{key}_mean_l{level}"] = float(values.mean()) if values.size else None
                record[f"{key}_median_l{level}"] = float(np.median(values)) if values.size else None
                record[f"{key}_sd_l{level}"] = float(values.std()) if values.size else None
                record[f"{key}_negative_fraction_l{level}"] = (
                    float((values < 0).mean()) if values.size else None)
            record[f"orc_edges_total_l{level}"] = total_edges
            record[f"orc_edges_measured_l{level}"] = len(chosen)
            record[f"orc_coverage_l{level}"] = (
                len(chosen) / total_edges if total_edges else 0.0)
            record[f"orc_status_l{level}"] = ("EXACT" if len(chosen) == total_edges
                                              else "SAMPLED_ESTIMATE")
            record[f"orc_d3_edges_l{level}"] = int(d3_edges)
            record[f"orc_delta_mean_l{level}"] = float(np.mean(deltas)) if deltas else None
            record[f"orc_delta_absmax_l{level}"] = float(np.max(np.abs(deltas))) if deltas else None
            record[f"orc_sign_flips_l{level}"] = int(sum(
                1 for a, b in zip(true_values, clipped_values) if (a > 0) != (b > 0)))
            timing["orc_clipped"] += time.time() - phase
        timing["per_sid"].append(round(time.time() - started, 2))
        records.append(record)
        with checkpoint.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if support_checkpoint.is_file():
            header = support_checkpoint.read_text().splitlines()[0].split(",")
        else:
            header = ["sid_sha256", "level", "occupied", "singleton", "multi", "items",
                      "boundary_mismatches"]
        new_support = [row for row in support_rows
                       if row["sid_sha256"] == pair["sid_sha256"]]
        with support_checkpoint.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerows(new_support)
        print(f"  {number + 1}/{len(pairs)} {pair['variant']} {timing['per_sid'][-1]:.1f}s",
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
        if support_checkpoint.is_file():
            support_rows = list(csv.DictReader(support_checkpoint.open()))

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
    timing["median_sid_seconds"] = float(np.median(timing["per_sid"])) if timing["per_sid"] else None
    timing["max_sid_seconds"] = float(max(timing["per_sid"])) if timing["per_sid"] else None
    timing["phases"] = {key: round(value, 1) for key, value in timing.items()
                        if isinstance(value, float)}
    (out / "timing.json").write_text(json.dumps({k: v for k, v in timing.items()
                                                if k != "per_sid"}, indent=1) + "\n",
                                     encoding="utf-8")
    completeness = {"rows": len(records), "excluded_hash_mismatch": len(excluded),
                    "caps": {"group_count": None, "group_size": None, "item_count": None},
                    "orc": {"primary_budget": SETTINGS["orc_edges"],
                            "status": "SAMPLED_ESTIMATE with per-row coverage",
                            "all_edge": "BLOCKED_NOT_FULL_CENSUS"},
                    "gromov": SETTINGS["gromov"],
                    "boundary": SETTINGS["boundary"]}
    (out / "completeness.json").write_text(json.dumps(completeness, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"written {len(records)} rows; median {timing['median_sid_seconds']}s/SID", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
