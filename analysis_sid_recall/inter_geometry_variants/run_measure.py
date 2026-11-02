"""Issue #135 runner: aggregation/conditioning/normalisation variants of Q_l.

Per (SID, level): the complete binary inter-prefix quotient Q_l (no caps), the
frozen A/B/C/D candidate measurements from ``candidates.py``, three null designs
with 20 deterministic replicates each, and the D sensitivity subgraph. Exact
structural metrics only; the ORC of #134 is not imported here.

Outputs: `features_by_level.csv|json`, `group_parent_support.csv`, `bench.json`,
`timing.json`, `completeness.json`, `_checkpoint.jsonl`.
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

import candidates as registry  # noqa: E402
import features_lw as features  # noqa: E402
import geom  # noqa: E402
import variants as variants  # noqa: E402

SEED = registry.SEED
DESIGNS = registry.NULL_DESIGNS
NULL_FEATURES = registry.NULL_FEATURES
REPLICATES = registry.NULL_REPLICATES


def _null_feature_record(block: dict, sibling: dict | None, level: int) -> dict:
    record = {"density": block["base"]["density"],
              "transitivity": block["base"]["transitivity"],
              "cycle_rank_per_node": block["cycle_rank_per_node"],
              "degree_median": block["degree_median"],
              "forman_mean": block["forman_mean"],
              "triangles_per_node_mean": block["triangles_per_node_mean"],
              "local_transitivity_mean": block["local_transitivity_mean"]}
    if sibling is not None:
        record["sibling_density_mean"] = sibling["aggregates"]["sibling_density_mean"]
        record["sibling_transitivity_mean"] = sibling["aggregates"][
            "sibling_transitivity_mean"]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {"seed": SEED, "null_replicates": REPLICATES, "null_designs": list(DESIGNS),
                "quotient": "complete Q_l, one node per occupied prefix (singletons kept), "
                            "edge iff a TRAIN item edge crosses, unweighted, no self-loops",
                "no_caps": "no item, group, node, edge or sampled-group limit",
                "registry_sha256": (HERE / "candidate_registry.sha256").read_text().strip()
                if (HERE / "candidate_registry.sha256").is_file() else None}
    (out / "settings.json").write_text(json.dumps(settings, indent=1) + "\n", encoding="utf-8")
    registry.freeze()

    pairs = features.load_pairs()
    excluded = [pair for pair in pairs if pair["tier"] == "HASH_MISMATCH"]
    pairs = [pair for pair in pairs if pair["tier"] != "HASH_MISMATCH"]
    if args.limit:
        pairs = pairs[:args.limit]
    indexed = list(enumerate(pairs))
    if args.shards > 1:
        import zlib
        indexed = [(number, pair) for number, pair in indexed
                   if zlib.crc32(pair["sid_sha256"].encode()) % args.shards == args.shard]
    nodes, index, weighted, _ = features.behaviour_graph()
    binary = geom.binary_graph(weighted)
    degree_b = np.asarray(binary.sum(axis=1)).ravel()

    checkpoint = out / "_checkpoint.jsonl"
    done = set()
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text().splitlines():
            try:
                done.add(json.loads(line)["sid_sha256"])
            except Exception:
                continue
    records = []
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text().splitlines():
            try:
                record = json.loads(line)
            except Exception:
                continue
            if record["sid_sha256"] not in {entry["sid_sha256"] for entry in records}:
                records.append(record)
    support_path = out / "_support_checkpoint.csv"
    support_rows = []
    if args.resume and support_path.is_file():
        support_rows = list(csv.DictReader(support_path.open()))

    timings = {"per_sid": [], "phases": {"quotient": 0.0, "sibling": 0.0,
                                        "multi_only": 0.0, "nulls": 0.0}}
    for number, pair in indexed:
        if pair["sid_sha256"] in done:
            continue
        table = json.loads(pair["payload"])
        record = {"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                  "series": pair["series"], "provenance_tier": pair["tier"],
                  "commit_tier": pair.get("commit_tier"),
                  "recall": float(pair["test_recall@10"])}
        started = time.time()
        for level in (1, 2, 3):
            labels = features.labels_for(table, level, index, nodes)
            occupied = int(np.unique(labels[labels >= 0]).size)
            child_items = np.bincount(labels[labels >= 0],
                                     minlength=occupied).astype(float)
            phase = time.time()
            matrix = geom.quotient(binary, labels)
            blocks = variants.quotient_blocks(matrix)
            timings["phases"]["quotient"] += time.time() - phase
            for key, value in blocks["base"].items():
                record[f"base_{key}_l{level}"] = value
            for key, value in blocks.items():
                if key == "base":
                    continue
                record[f"A_{key}_l{level}"] = value
            record[f"base_nodes_l{level}"] = int(matrix.shape[0])
            record[f"base_edges_l{level}"] = int(matrix.nnz // 2)
            record[f"base_singleton_share_l{level}"] = (
                float((child_items == 1).sum()) / max(occupied, 1))
            support = {"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                       "level": level, "occupied": occupied,
                       "singleton": int((child_items == 1).sum()),
                       "multi": int((child_items >= 2).sum()),
                       "items": int(child_items.sum()),
                       "nodes": int(matrix.shape[0]), "edges": int(matrix.nnz // 2)}

            sibling = None
            child_parent_all = None
            if level >= 2:
                parent_labels = features.labels_for(table, level - 1, index, nodes)
                child_parent_all = np.full(occupied, -1, dtype=np.int64)
                labelled = labels >= 0
                child_parent_all[labels[labelled]] = parent_labels[labelled]
                if (child_parent_all < 0).any() or (parent_labels[labelled] < 0).any():
                    raise RuntimeError("child prefix without a parent prefix")
                phase = time.time()
                sibling = variants.sibling_blocks(matrix, child_parent_all, child_items)
                timings["phases"]["sibling"] += time.time() - phase
                aggregates = sibling["aggregates"]
                for key in ("sibling_density_mean", "sibling_density_median_parent",
                            "sibling_density_child_weighted", "sibling_density_item_weighted",
                            "sibling_transitivity_mean",
                            "sibling_transitivity_child_weighted",
                            "sibling_cycle_rank_per_node_mean",
                            "sibling_degree_median_mean", "sibling_degree_mean_mean",
                            "sibling_degree_p90_mean", "sibling_forman_mean",
                            "sibling_forman_negative_fraction", "sibling_edge_share"):
                    record[f"B_{key}_l{level}"] = aggregates.get(key)
                support.update({"parents": aggregates["parents_total"],
                                "parents_multi": aggregates["parents_multi_child"],
                                "parents_edges": aggregates["parents_with_sibling_edge"],
                                "sibling_edges": aggregates["sibling_edges"],
                                "cross_parent_edges": aggregates["cross_parent_edges"],
                                "child_coverage": aggregates["child_item_coverage"]})
            phase = time.time()
            keep = child_items >= 2
            induced = variants.induced_subgraph(matrix, keep)
            induced_blocks = variants.quotient_blocks(induced)
            timings["phases"]["multi_only"] += time.time() - phase
            record[f"D_multi_only_density_l{level}"] = induced_blocks["base"]["density"]
            record[f"D_multi_only_transitivity_l{level}"] = induced_blocks["base"]["transitivity"]
            record[f"D_multi_only_cycle_rank_per_node_l{level}"] = induced_blocks[
                "cycle_rank_per_node"]
            record[f"D_multi_only_degree_median_l{level}"] = induced_blocks["degree_median"]
            record[f"D_multi_only_forman_mean_l{level}"] = induced_blocks["forman_mean"]
            record[f"D_multi_only_forman_negative_fraction_l{level}"] = induced_blocks[
                "forman_negative_fraction"]
            record[f"D_lost_nodes_l{level}"] = int((child_items == 1).sum())
            record[f"D_lost_edges_l{level}"] = int(matrix.nnz // 2) - int(induced.nnz // 2)
            record[f"D_lost_items_l{level}"] = float(child_items[child_items == 1].sum())
            support.update({"induced_nodes": int(induced.shape[0]),
                            "induced_edges": int(induced.nnz // 2),
                            "induced_items": float(child_items[keep].sum())})

            real = _null_feature_record(blocks, sibling, level)
            phase = time.time()
            for position, design in enumerate(DESIGNS):
                if design == "parent_conditioned" and level == 1:
                    continue
                parent_of_node = None
                if design == "parent_conditioned":
                    if level == 1:
                        continue
                    parent_of_node = np.full(len(nodes), -1, dtype=np.int64)
                    parent_of_node[labels >= 0] = child_parent_all[labels[labels >= 0]]

                def build(candidate_labels, level=level,
                          child_parent_all=child_parent_all):
                    candidate = geom.quotient(binary, candidate_labels)
                    block = variants.quotient_blocks(candidate)
                    sibling_block = None
                    if level >= 2:
                        local_items = np.bincount(
                            candidate_labels[candidate_labels >= 0],
                            minlength=int(candidate.shape[0])).astype(float)
                        sibling_block = variants.sibling_blocks(
                            candidate, child_parent_all, local_items)
                    return _null_feature_record(block, sibling_block, level)

                nulls = variants.null_blocks(
                    labels, degree_b, design, build, parent_of_node,
                    replicates=REPLICATES,
                    seed=SEED + 1000 * level + 100 * position + number)
                relative = variants.null_relative(real, nulls, NULL_FEATURES)
                for feature, entry in relative.items():
                    if entry is None:
                        record[f"C_{feature}_{design}_delta_l{level}"] = None
                        record[f"C_{feature}_{design}_z_l{level}"] = None
                        record[f"C_{feature}_{design}_ratio_l{level}"] = None
                        continue
                    record[f"C_{feature}_{design}_delta_l{level}"] = entry["delta"]
                    record[f"C_{feature}_{design}_z_l{level}"] = entry["z"]
                    record[f"C_{feature}_{design}_ratio_l{level}"] = entry["ratio"]
                    record[f"C_{feature}_{design}_nullmean_l{level}"] = entry["null_mean"]
                    record[f"C_{feature}_{design}_nullsd_l{level}"] = entry["null_sd"]
                    record[f"C_{feature}_{design}_nullmcse_l{level}"] = entry["null_mc_se"]
            timings["phases"]["nulls"] += time.time() - phase
            support_rows.append(support)
        timings["per_sid"].append(round(time.time() - started, 2))
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
        print(f"  {pair['variant']} "
              f"{timings['per_sid'][-1]:.1f}s", flush=True)

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
    (out / "group_parent_support.csv").write_text(
        (out / "_support_checkpoint.csv").read_text(encoding="utf-8"), encoding="utf-8")
    per_sid = np.asarray(timings["per_sid"], dtype=float)
    benchmarks = {"sids": int(per_sid.size), "shard": args.shard, "shards": args.shards,
                  "median_sid_seconds": float(np.median(per_sid)) if per_sid.size else None,
                  "max_sid_seconds": float(per_sid.max()) if per_sid.size else None,
                  "phases_seconds": {key: round(value, 2)
                                     for key, value in timings["phases"].items()},
                  "null_cost": {"designs": list(DESIGNS), "replicates": REPLICATES,
                                "level1_designs": ["global", "degree_stratified"],
                                "parent_conditioned_levels": [2, 3],
                                "applications": sum(1 for _ in records)
                                * (2 + 2 * 2)},
                  "peak_rss_mb": round(resource.getrusage(
                      resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)}
    (out / "timing.json").write_text(json.dumps(timings, indent=1) + "\n", encoding="utf-8")
    (out / "bench.json").write_text(json.dumps(benchmarks, indent=1) + "\n", encoding="utf-8")
    completeness = {
        "rows": len(records), "excluded_hash_mismatch": len(excluded),
        "tiers": {tier: sum(1 for pair in pairs if pair["tier"] == tier)
                  for tier in ("HASH_PROVEN", "RECORD_HASH_MATCH", "MISSING")},
        "levels": [1, 2, 3],
        "nulls": {"replicates": REPLICATES, "designs": list(DESIGNS),
                  "level1_designs": ["global", "degree_stratified"],
                  "parent_conditioned": "L2/L3 only, each parent keeps its item count "
                                       "and child-count histogram"},
        "status": {"A": "EXACT for every eligible node/edge; undefined handled per metric",
                   "B": "EXACT per parent; parents with <2 children/wedges/edges excluded "
                        "from the corresponding aggregate and counted",
                   "C": "MONTE_CARLO nulls; delta/z/ratio only when the metric is defined; "
                        "z requires sd>0",
                   "D": "SECONDARY induced subgraph; lost nodes/edges/items recorded"},
        "caps": "none",
        "settings": settings}
    (out / "completeness.json").write_text(json.dumps(completeness, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"written {len(records)} rows; median {benchmarks['median_sid_seconds']}s/SID",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
