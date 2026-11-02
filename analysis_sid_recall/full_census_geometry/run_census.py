"""Issue #132 runner: full-census geometry with checkpointing and resume.

For every admitted SID (excluding the pairs whose content digest does not match
their recorded hash) and every level, the census enumerates every occupied
prefix group and computes the exact structural metrics of its induced subgraph.
The four-point deviation is exact on every connected component up to
``census.FOUR_POINT_MAX`` nodes and ``BLOCKED_NOT_FULL_CENSUS`` above it; the
detour is exact over every connected pair. Nulls (global, activity-stratified,
parent-conditioned) recompute the **structural** census on every group of every
level with 3 replicates; the four-point and detour nulls are declared
``BLOCKED_NOT_FULL_CENSUS`` rather than estimated on a sample.

Outputs: `census_by_level.csv|json`, `group_support_L1_L2_L3.csv`,
`settings.json`, `completeness.json`, and an append-only checkpoint
`_checkpoint.jsonl` used by `--resume`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE),
             str(REPO / "analysis_sid_recall" / "layerwise_curvature"),
             str(REPO / "analysis_sid_recall" / "local_prefix_geometry")):
    if path not in sys.path:
        sys.path.insert(0, path)

import census as census_lib  # noqa: E402
import features_lw as features  # noqa: E402
import geometry  # noqa: E402

SETTINGS = {
    "max_group_size": None,          # no cap
    "groups_per_level": None,        # no cap
    "four_point_max_component": census_lib.FOUR_POINT_MAX,
    "four_point_sampling": False,    # exact enumeration or BLOCKED
    "detour_sampling": "BLOCKED_NOT_FULL_CENSUS",
    "detour_reason": "exact all-pairs needs one full-graph BFS per item "
                     "(~2.7 million BFS for this table); the metric is not reported",
    "null_replicates": 3,
    "null_kinds": ["global", "stratified", "parent"],
    "null_detour": "BLOCKED_NOT_FULL_CENSUS",
    "null_four_point": "BLOCKED_NOT_FULL_CENSUS",
    "seed": 20261102,
}


def summarise(per_group: list[dict], support_block: dict) -> dict:
    return {"support": support_block,
            "aggregates": census_lib.aggregates(per_group, support_block)}


def deltas(real: dict, null: dict) -> dict:
    keys = ("transitivity", "cycle_rank_normalised", "conductance",
            "largest_component_fraction", "internal_edges", "boundary_edges",
            "triangles")
    out = {}
    for key in keys:
        a = (real.get("aggregates", {}).get(key) or {}).get("group_mean")
        b = (null.get("aggregates", {}).get(key) or {}).get("group_mean")
        out[key] = None if a is None or b is None else float(a - b)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--per-group-out", default=None)
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(SETTINGS, indent=1) + "\n", encoding="utf-8")

    pairs = features.load_pairs()
    mismatched = [pair for pair in pairs if pair["tier"] == "HASH_MISMATCH"]
    pairs = [pair for pair in pairs if pair["tier"] != "HASH_MISMATCH"]
    if args.limit:
        pairs = pairs[:args.limit]
    nodes, index, adjacency, _ = features.behaviour_graph()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    strata = np.minimum(np.log2(np.maximum(degree, 1.0)).astype(int), 8)
    rng = np.random.default_rng(SETTINGS["seed"])
    checkpoint = out / "_checkpoint.jsonl"
    done = set()
    if args.resume and checkpoint.is_file():
        for line in checkpoint.read_text().splitlines():
            try:
                done.add(json.loads(line)["sid_sha256"])
            except Exception:
                continue

    records, support_rows = [], []
    per_group_handle = None
    if args.per_group_out:
        per_group_handle = open(Path(args.per_group_out), "w")
        per_group_handle.write("sid_sha256,level,group_index,size,internal_edges,components,"
                               "triangles,transitivity,cycle_rank,conductance,boundary_edges,"
                               "four_point_status,detour_status,detour_ratio\n")
    for number, pair in enumerate(pairs):
        if pair["sid_sha256"] in done:
            continue
        table = json.loads(pair["payload"])
        record = {"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                  "series": pair["series"], "provenance_tier": pair["tier"],
                  "recall": float(pair["test_recall@10"])}
        parents = None
        for level in (1, 2, 3):
            labels = features.labels_for(table, level, index, nodes)
            measured = census_lib.census(adjacency, degree, labels)
            support_block = measured["support"]
            aggregates = census_lib.aggregates(measured["groups"], support_block)
            support_rows.append({"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                                 "level": level, **{k: v for k, v in support_block.items()
                                                    if not isinstance(v, dict)},
                                 "eligible_transitivity": support_block["eligible"]["transitivity"],
                                 "measured_transitivity": support_block["measured"]["transitivity"],
                                 "eligible_four_point": support_block["eligible"]["four_point"],
                                 "measured_four_point": support_block["measured"]["four_point"],
                                 "eligible_detour": support_block["eligible"]["detour"],
                                 "measured_detour": support_block["measured"]["detour"],
                                 "size_histogram": json.dumps(support_block["size_histogram"])})
            record[f"occupied_l{level}"] = support_block["occupied"]
            record[f"singleton_l{level}"] = support_block["singleton"]
            record[f"multi_l{level}"] = support_block["multi"]
            record[f"ge4_l{level}"] = support_block["ge4"]
            record[f"items_l{level}"] = support_block["items"]
            record[f"item_coverage_multi_l{level}"] = support_block["item_coverage_multi"]
            record[f"item_coverage_ge4_l{level}"] = support_block["item_coverage_ge4"]
            record[f"coverage_transitivity_l{level}"] = support_block["measured_item_coverage"]["transitivity"]
            record[f"coverage_four_point_l{level}"] = support_block["measured_item_coverage"]["four_point"]
            record[f"coverage_detour_l{level}"] = support_block["measured_item_coverage"]["detour"]
            for key in ("transitivity", "cycle_rank_normalised", "conductance",
                        "largest_component_fraction", "internal_edges", "boundary_edges",
                        "triangles", "components"):
                block = aggregates[key]
                record[f"{key}_l{level}"] = block["group_mean"]
                record[f"{key}_item_l{level}"] = block["item_weighted"]
                record[f"{key}_p90_l{level}"] = block["p90"]
                record[f"{key}_group_coverage_l{level}"] = block["group_coverage"]
            record[f"four_point_delta_l{level}"] = aggregates["four_point"]["mean_delta_max"]
            record[f"four_point_status_l{level}"] = aggregates["four_point"]["status"]
            record[f"four_point_groups_l{level}"] = aggregates["four_point"]["groups_exact"]
            record[f"four_point_quadruples_l{level}"] = aggregates["four_point"]["quadruples"]
            record[f"detour_l{level}"] = aggregates["detour"]["mean_ratio"]
            record[f"detour_pairs_l{level}"] = aggregates["detour"]["pairs"]
            record[f"detour_unavailable_l{level}"] = aggregates["detour"]["pairs_unavailable"]
            for kind in SETTINGS["null_kinds"]:
                null_aggregates = []
                for _ in range(SETTINGS["null_replicates"]):
                    if kind == "global":
                        shuffled = geometry.global_shuffle(labels, rng)
                    elif kind == "parent" and parents is not None:
                        shuffled = geometry.parent_conditioned_shuffle(labels, parents, rng)
                    elif kind == "stratified":
                        shuffled = geometry.activity_stratified_shuffle(labels, strata, rng)
                    else:
                        continue
                    null_census = census_lib.census(
                        adjacency, degree, shuffled, detour_enabled=False,
                        four_point_enabled=False)
                    null_aggregates.append(summarise(null_census["groups"],
                                                     null_census["support"]))
                if not null_aggregates:
                    continue
                for key in ("transitivity", "cycle_rank_normalised", "conductance",
                            "largest_component_fraction", "internal_edges",
                            "boundary_edges", "triangles"):
                    a = aggregates[key]["group_mean"]
                    values = [(block["aggregates"][key] or {}).get("group_mean")
                              for block in null_aggregates]
                    values = [value for value in values if value is not None]
                    if a is None or not values:
                        record[f"{key}_null_{kind}_l{level}"] = None
                        record[f"{key}_delta_{kind}_l{level}"] = None
                        continue
                    mean = float(np.mean(values))
                    record[f"{key}_null_{kind}_l{level}"] = mean
                    record[f"{key}_delta_{kind}_l{level}"] = float(a - mean)
                    if len(values) > 1:
                        spread = float(np.std(values))
                        record[f"{key}_z_{kind}_l{level}"] = (
                            float((a - mean) / spread) if spread >= 1e-3 else None)
            if per_group_handle is not None:
                for group_index, entry in enumerate(measured["groups"]):
                    per_group_handle.write(
                        f"{pair['sid_sha256']},{level},{group_index},{entry['size']},"
                        f"{entry['internal_edges']},{entry['components']},"
                        f"{entry['triangles']},{entry['transitivity']},"
                        f"{entry['cycle_rank']},{entry['conductance']},"
                        f"{entry['boundary_edges']},{entry['four_point']['status']},"
                        f"{entry['detour']['status']},{entry['detour']['mean_ratio']}\n")
            parents = labels
        record["hierarchy"] = "see group_support_L1_L2_L3.csv"
        records.append(record)
        with checkpoint.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"  {number + 1}/{len(pairs)} {pair['variant']}", flush=True)
    if per_group_handle is not None:
        per_group_handle.close()

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
    with (out / "census_by_level.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (out / "census_by_level.json").write_text(json.dumps(records, indent=1) + "\n",
                                             encoding="utf-8")
    support_fields = []
    for row in support_rows:
        for key in row:
            if key not in support_fields:
                support_fields.append(key)
    with (out / "group_support_L1_L2_L3.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=support_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(support_rows)
    completeness = {
        "rows": len(records), "excluded_hash_mismatch": len(mismatched),
        "excluded_sids": [pair["sid_sha256"] for pair in mismatched],
        "levels": {level: {"sids_measured": len(records)} for level in (1, 2, 3)},
        "caps": {"max_group_size": None, "groups_per_level": None},
        "four_point": "EXACT up to %d nodes per component, else BLOCKED" % census_lib.FOUR_POINT_MAX,
        "detour": "BLOCKED_NOT_FULL_CENSUS (exact all-pairs; not reported)",
        "nulls": {"replicates": SETTINGS["null_replicates"],
                   "kinds": SETTINGS["null_kinds"],
                   "four_point": "BLOCKED_NOT_FULL_CENSUS",
                   "detour": "BLOCKED_NOT_FULL_CENSUS",
                   "note": "the null arm recomputes the full structural census on "
                           "every group; four-point and detour are not estimated on "
                           "samples, so their nulls are declared blocked"}}
    (out / "completeness.json").write_text(json.dumps(completeness, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"written {len(records)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
