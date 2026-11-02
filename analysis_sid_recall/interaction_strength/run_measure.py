"""Issue #130 runner: interaction strength vs SID organisation and Recall.

Declared construction (frozen in `config.json` before any Recall correlation):

* unit = one training context (a row's history plus its target), a pair counted
  once per context; `f(i)` = the number of contexts containing `i`;
* primary strength `s = c / sqrt(f_i f_j)` (popularity adjusted), with
  `log1p(c)` as the declared alternative and the distinct-user count as a
  declared unit sensitivity;
* weighted ORC uses the *unweighted* hop ground metric; weights enter only the
  lazy random-walk measures; the quotient aggregation is `sum`, with the
  pre-declared `exposure` normalisation (`/ sqrt(M_a M_b)`) as the primary and
  `raw` reported alongside;
* budget: 24 sampled quotient edges and 2 size-preserving null replicates per
  (arm, level) for the full set, declared as measured; a 48-edge / 4-replicate
  sensitivity runs on the first 12 arms.

No Stage 2/Stage 3 job is started; the Stage 3 recalls are the frozen historical
outcome, joined by SID content hash only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE), str(REPO / "analysis_sid_recall" / "layerwise_curvature")):
    if path not in sys.path:
        sys.path.insert(0, path)

import analysis_lw as stats  # noqa: E402
import features_lw as features  # noqa: E402
import strength  # noqa: E402
from tools.research.science.quotient_curvature import symmetrise  # noqa: E402

CONFIG = {
    "unit": "training context (row: history + target), one count per context",
    "deduplication": "unordered consecutive pair, counted once per context",
    "primary_strength": "popularity: c / sqrt(f_i f_j)",
    "alternative_strength": "intensity: log1p(c)",
    "unit_sensitivity": "distinct-user count per pair",
    "ground_metric": "unweighted hop distance of the behaviour graph",
    "quotient_aggregation": "sum over item edges",
    "primary_normalisation": "exposure (cell / sqrt(M_a M_b))",
    "reported_alternative_normalisation": "raw sum",
    "orc_budget": {"edges": 24, "null_replicates": 2},
    "orc_sensitivity": "NOT_RUN (the measured budget is the declared one)",
    "nulls": {"labelled_edges": 60, "activity_stratified": 40, "gap": 60},
    "seed": 20261102,
    "stage3_use": "read-only frozen historical recall; no tuning against it",
}


def labels_for(table: dict, level: int, index: dict, nodes: list) -> np.ndarray:
    return features.labels_for(table, level, index, nodes)


def null_assignments(labels: np.ndarray, sizes: np.ndarray, rng, count: int):
    for _ in range(count):
        order = rng.permutation(len(labels))
        shuffled = np.full_like(labels, -1)
        cursor = 0
        for number, size in enumerate(sizes.tolist()):
            shuffled[order[cursor:cursor + size]] = number
            cursor += size
        yield shuffled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = features.load_pairs()
    if args.limit:
        pairs = pairs[:args.limit]
    nodes, index, adjacency, _ = features.behaviour_graph()
    degree = np.diff(adjacency.indptr)

    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["user", "history", "target"])
    graph = strength.reconstruct(frame)
    # the reconstructed edge set must be the audited #128 graph
    recon = sp.coo_matrix((np.ones(graph["pairs"]), (graph["rows"], graph["cols"])),
                          shape=(len(graph["nodes"]), len(graph["nodes"]))).tocsr()
    recon = recon.maximum(recon.T)
    same_nodes = graph["nodes"] == nodes
    same_edges = ((recon != 0) != (adjacency != 0)).nnz == 0
    print(f"graph match: nodes {same_nodes} edges {same_edges} "
          f"({graph['pairs']} pairs, {int(adjacency.nnz // 2)} undirected)", flush=True)
    if not same_nodes or not same_edges:
        raise SystemExit("the reconstructed strength graph does not match the audited #128 graph")

    popularity = strength.strengths(graph["counts"], graph["activity"],
                                    graph["rows"], graph["cols"], "popularity")
    intensity = strength.strengths(graph["counts"], graph["activity"],
                                   graph["rows"], graph["cols"], "intensity")
    audit = {"popularity": strength.audit_block(graph, popularity),
             "intensity": strength.audit_block(graph, intensity),
             "node_order_matches_hash": bool(same_nodes and same_edges),
             "config": CONFIG}
    (out / "weight_audit.json").write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps(CONFIG, indent=1) + "\n", encoding="utf-8")

    rows_u, cols_u = graph["rows"], graph["cols"]
    reference = {row["sid_sha256"]: row for row in
                 csv.DictReader((REPO / "analysis_sid_recall" / "layerwise_curvature"
                                 / "experiment_table.csv").open())}
    records = []
    rng = np.random.default_rng(CONFIG["seed"])
    for number, pair in enumerate(pairs):
        if pair["payload"] is None:
            continue
        table = json.loads(pair["payload"])
        base = reference.get(pair["sid_sha256"], {})
        record = {"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                  "series": pair["series"], "provenance_tier": pair["tier"],
                  "recall": float(pair["test_recall@10"]),
                  "n_eval": int(pair["n_eval"]) if pair["n_eval"] else None,
                  "collision": float(base.get("collision") or "nan"),
                  "codes_used_l1": float(base.get("codes_used_l1") or "nan"),
                  "codes_used_l2": float(base.get("codes_used_l2") or "nan"),
                  "codes_used_l3": float(base.get("codes_used_l3") or "nan"),
                  "edges_l1": float(base.get("edges_l1") or "nan"),
                  "edges_l2": float(base.get("edges_l2") or "nan"),
                  "edges_l3": float(base.get("edges_l3") or "nan"),
                  "A1": float(base.get("A1") or "nan"), "A2": float(base.get("A2") or "nan"),
                  "A3": float(base.get("A3") or "nan"),
                  "g1": float(base.get("g1") or "nan"), "g2": float(base.get("g2") or "nan"),
                  "g3": float(base.get("g3") or "nan"),
                  "z1": float(base.get("z1") or "nan"), "z2": float(base.get("z2") or "nan"),
                  "z3": float(base.get("z3") or "nan")}
        sizes_by_level = {}
        for level in (1, 2, 3):
            labels = labels_for(table, level, index, nodes)
            usable = labels >= 0
            counts = np.bincount(labels[usable]) if usable.any() else np.array([], dtype=int)
            sizes = counts[counts > 0]
            sizes_by_level[level] = sizes
            record[f"prefix_groups_l{level}"] = int(sizes.size)
            # weighted retention and its two nulls
            retention = strength.weighted_retention(popularity, labels, rows_u, cols_u)
            global_null = strength.retention_excess(popularity, labels, rows_u, cols_u,
                                                    trials=CONFIG["nulls"]["labelled_edges"],
                                                    seed=CONFIG["seed"] + 11 * level)
            strata = np.minimum((np.log2(np.maximum(graph["activity"], 1.0))).astype(int), 8)
            stratified = strength.retention_excess(popularity, labels, rows_u, cols_u,
                                                   trials=CONFIG["nulls"]["activity_stratified"],
                                                   strata=strata,
                                                   seed=CONFIG["seed"] + 29 * level)
            record[f"R_l{level}_retention"] = retention.get("retention")
            record[f"R_l{level}_status"] = retention["status"]
            record[f"R_l{level}_excess"] = global_null.get("excess")
            record[f"R_l{level}_excess_p"] = global_null.get("p")
            record[f"R_l{level}_excess_stratified"] = stratified.get("excess")
            record[f"R_l{level}_excess_stratified_p"] = stratified.get("p")
            record[f"R_l{level}_mass"] = retention.get("mass")
            bins = strength.strength_bins(popularity, labels, rows_u, cols_u, bins=3)
            record[f"bins_l{level}_status"] = bins["status"]
            if bins["status"] == "MEASURED":
                for position, block in enumerate(bins["bins"]):
                    record[f"bin{position}_l{level}_same"] = block["same_prefix"]
                    record[f"bin{position}_l{level}_n"] = block["n"]
                record[f"bins_l{level}_trend"] = bins["trend_spearman"]
            # the strength gap, with its label-shuffle null
            gap = strength.gap_excess(popularity, labels, rows_u, cols_u,
                                      trials=CONFIG["nulls"]["gap"],
                                      seed=CONFIG["seed"] + 53 * level)
            record[f"gap_l{level}_status"] = gap["status"]
            record[f"gap_l{level}"] = gap.get("gap")
            record[f"gap_l{level}_p"] = gap.get("p")
            record[f"gap_l{level}_n_within"] = gap.get("n_within")
            record[f"gap_l{level}_n_between"] = gap.get("n_between")
            record[f"gap_l{level}_null_mean"] = gap.get("null_mean")
            # weighted ORC, exposure normalisation, with size-preserving nulls
            for normalisation in ("exposure", "raw"):
                try:
                    measured = strength.weighted_orc(
                        strength.quotient_weights(rows_u, cols_u, popularity, labels,
                                                  int(sizes.size), normalisation),
                        edges=CONFIG["orc_budget"]["edges"],
                        seed=CONFIG["seed"] + 71 * level)
                    nulls = []
                    for shuffled in null_assignments(labels, sizes, rng,
                                                     CONFIG["orc_budget"]["null_replicates"]):
                        try:
                            nulls.append(strength.weighted_orc(
                                strength.quotient_weights(rows_u, cols_u, popularity, shuffled,
                                                          int(sizes.size), normalisation),
                                edges=CONFIG["orc_budget"]["edges"],
                                seed=CONFIG["seed"] + 71 * level)["mean"])
                        except ValueError:
                            continue
                    record[f"gw_l{level}_{normalisation}"] = measured["mean"]
                    record[f"gw_l{level}_{normalisation}_null_mean"] = (
                        float(np.mean(nulls)) if nulls else None)
                    record[f"gw_l{level}_{normalisation}_null_sd"] = (
                        float(np.std(nulls)) if len(nulls) > 1 else None)
                    record[f"gw_l{level}_{normalisation}_z"] = (
                        (measured["mean"] - float(np.mean(nulls))) / float(np.std(nulls))
                        if len(nulls) > 1 and float(np.std(nulls)) > 1e-12 else None)
                    record[f"gw_l{level}_{normalisation}_edges"] = measured["edges"]
                except ValueError as error:
                    record[f"gw_l{level}_{normalisation}_status"] = f"NOT_MEASURED:{error}"
            print(f"  {number + 1}/{len(pairs)} {pair['variant']}", flush=True)
        records.append(record)

    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with (out / "experiment_by_level.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (out / "experiment_by_level.json").write_text(json.dumps(records, indent=1) + "\n",
                                                  encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(
        {"rows": len(records), "config": CONFIG, "audit": audit,
         "graph": {"items": len(nodes), "undirected_edges": int(adjacency.nnz // 2)}},
        indent=1) + "\n", encoding="utf-8")
    print(f"written: {len(records)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
