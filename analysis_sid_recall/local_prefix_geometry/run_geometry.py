"""Issue #131 runner: singleton-aware local geometry of the prefix groups.

Writes, under `analysis_sid_recall/local_prefix_geometry/`:

* `group_support_L1_L2_L3.csv` — per SID and level: occupied, singleton,
  multi-item, >=4, size histogram/quantiles and item coverage;
* `geometry_by_level.csv|json` — per SID and level: the sampled induced-graph
  metrics (group- and item-weighted), their null deltas/standardised effects for
  the global, parent-conditioned and activity-stratified shuffles, detour
  coverage, and the hierarchy-change summary;
* `settings.json` — the frozen eligibility rule, sample sizes and seeds.

Budget (declared): at most 300 eligible groups are sampled per (SID, level) for
the geometry metrics, the same rule for every null; two null replicates per
shuffle kind. The four-point metric is sampled inside one connected component
with >=4 nodes per group; the detour ratio uses 2 pairs per sampled group.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO), str(HERE), str(REPO / "analysis_sid_recall" / "layerwise_curvature")):
    if path not in sys.path:
        sys.path.insert(0, path)

import features_lw as features  # noqa: E402
import geometry  # noqa: E402

SETTINGS = {
    "min_group_size": 2,
    "max_group_size": 64,
    "groups_per_level": 300,
    "null_replicates": 2,
    "quadruples_per_group": 8,
    "detour_pairs_per_group": 2,
    "detour_groups": 25,
    "parent_samples": 100,
    "seed": 20261102,
    "eligible_rule": "2 <= |G| <= 64 (declared); larger groups reported, not measured",
}


def sample(groups: list[np.ndarray], count: int, rng) -> list[int]:
    if len(groups) <= count:
        return list(range(len(groups)))
    return sorted(rng.choice(len(groups), size=count, replace=False).tolist())


def measure_level(adjacency, degree, labels: np.ndarray, *, rng, settings) -> dict:
    groups = geometry.groups_from_labels(labels)
    audit = geometry.support_audit(groups)
    eligible = [index for index, group in enumerate(groups)
                if settings["min_group_size"] <= group.size <= settings["max_group_size"]]
    chosen = sample([groups[index] for index in eligible], settings["groups_per_level"], rng)
    members = [groups[eligible[position]] for position in chosen]
    per_group = []
    for number, group in enumerate(members):
        per_group.append(geometry.local_metrics(
            adjacency, group, degree,
            quadruples=settings["quadruples_per_group"],
            seed=settings["seed"] + 17 * number))
    detour_subset = sample(members, settings["detour_groups"], rng) if members else []
    detours = [geometry.path_detour(adjacency, members[position],
                                    pairs=settings["detour_pairs_per_group"],
                                    seed=settings["seed"] + 31 * position)
               for position in detour_subset]
    usable = [entry for entry in detours if entry["n"] > 0]
    totals = sum(entry["n"] + entry["unavailable"] for entry in detours)
    aggregate = geometry.aggregate_level(per_group, [int(group.size) for group in members])
    aggregate["detour"] = {
        "groups_sampled": len(detours), "pairs_measured": int(sum(entry["n"] for entry in detours)),
        "pairs_unavailable": int(sum(entry["unavailable"] for entry in detours)),
        "mean_ratio": float(np.mean([entry["detour"] for entry in usable])) if usable else None,
        "unavailable_fraction": (sum(entry["unavailable"] for entry in detours) / totals
                                 if totals else None)}
    aggregate["eligible_groups"] = len(eligible)
    aggregate["sampled_groups"] = len(members)
    aggregate["eligible_item_coverage"] = (float(sum(groups[index].size for index in eligible)
                                                 / max(audit.get("item_total", 1), 1)))
    aggregate["support"] = audit
    return {"audit": audit, "aggregate": aggregate, "groups": groups}


def null_level(adjacency, degree, labels: np.ndarray, parents: np.ndarray | None,
               strata: np.ndarray, reference: dict, *, rng, settings, kind: str) -> dict:
    deltas = []
    for _ in range(settings["null_replicates"]):
        if kind == "global":
            shuffled = geometry.global_shuffle(labels, rng)
        elif kind == "parent" and parents is not None:
            shuffled = geometry.parent_conditioned_shuffle(labels, parents, rng)
        elif kind == "stratified":
            shuffled = geometry.activity_stratified_shuffle(labels, strata, rng)
        else:
            return {"status": "UNDEFINED"}
        groups = geometry.groups_from_labels(shuffled)
        eligible = [index for index, group in enumerate(groups)
                    if settings["min_group_size"] <= group.size <= settings["max_group_size"]]
        chosen = sample([groups[index] for index in eligible], settings["groups_per_level"], rng)
        per_group = [geometry.local_metrics(adjacency, groups[eligible[position]], degree,
                                            quadruples=settings["quadruples_per_group"],
                                            seed=settings["seed"] + 7 * number)
                     for number, position in enumerate(chosen)]
        aggregate = geometry.aggregate_level(per_group,
                                             [int(groups[eligible[position]].size)
                                              for position in chosen])
        deltas.append({key: aggregate.get("group_weighted", {}).get(key, {}).get("group_mean")
                       for key in ("four_point", "transitivity", "cycle_surplus_normalised",
                                   "conductance", "largest_component_fraction")})
    out = {}
    real = {key: reference.get("group_weighted", {}).get(key, {}).get("group_mean")
            for key in ("four_point", "transitivity", "cycle_surplus_normalised",
                        "conductance", "largest_component_fraction")}
    for key in real:
        values = np.asarray([delta[key] for delta in deltas
                             if delta.get(key) is not None], dtype=float)
        if values.size == 0 or real.get(key) is None:
            out[key] = {"null_mean": None, "delta": None, "z": None}
            continue
        spread = float(values.std()) if values.size > 1 else None
        out[key] = {"null_mean": float(values.mean()), "null_sd": spread,
                    "delta": float(real[key] - values.mean()),
                    # a null spread below 1e-3 makes the z uninterpretable: report
                    # the raw delta and refuse the z instead of printing 1e5
                    "z": (float((real[key] - values.mean()) / spread)
                          if spread is not None and spread >= 1e-3 else None)}
        if out[key]["z"] is None:
            out[key]["note"] = "null spread below 1e-3: z not interpretable"
    return {"status": "MEASURED", "replicates": len(deltas), "features": out}


def hierarchy(table: dict, index: dict, adjacency, degree, *, rng, settings) -> dict:
    nodes = list(range(adjacency.shape[0]))
    prefixes = {}
    for level in (1, 2, 3):
        prefixes[level] = features.labels_for(table, level, index, nodes)
    absent = (prefixes[2] >= 0) & (prefixes[1] < 0)
    out = {"l2_without_l1": int(absent.sum())}
    categories = {"split_multi": 0, "one_multi_rest_singletons": 0, "only_singletons": 0}
    changes = {}
    for level_from, level_to in ((1, 2), (2, 3)):
        parent_of = prefixes[level_from]
        child_of = prefixes[level_to]
        children: dict[int, dict[int, int]] = {}
        for position in np.nonzero((parent_of >= 0) & (child_of >= 0))[0].tolist():
            children.setdefault(int(parent_of[position]), {})
            children[int(parent_of[position])][int(child_of[position])] = \
                children[int(parent_of[position])].get(int(child_of[position]), 0) + 1
        block = {"parents": len(children), "split_multi": 0,
                 "one_multi_rest_singletons": 0, "only_singletons": 0}
        sample_groups = []
        for parent, kids in children.items():
            sizes = np.asarray(list(kids.values()))
            multi = int((sizes >= 2).sum())
            if multi >= 2:
                block["split_multi"] += 1
            elif multi == 1 and int((sizes == 1).sum()) > 0:
                block["one_multi_rest_singletons"] += 1
            else:
                block["only_singletons"] += 1
            if multi >= 1 and len(sample_groups) < settings["parent_samples"]:
                sample_groups.append(parent)
        changes[f"L{level_from}_to_L{level_to}"] = block
    out["categories"] = changes
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(SETTINGS, indent=1) + "\n", encoding="utf-8")

    pairs = features.load_pairs()
    if args.limit:
        pairs = pairs[:args.limit]
    nodes, index, adjacency, _ = features.behaviour_graph()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    strata = np.minimum(np.log2(np.maximum(degree, 1.0)).astype(int), 8)
    rng = np.random.default_rng(SETTINGS["seed"])

    support_rows, records = [], []
    for number, pair in enumerate(pairs):
        if pair["payload"] is None:
            continue
        table = json.loads(pair["payload"])
        record = {"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                  "series": pair["series"], "provenance_tier": pair["tier"],
                  "recall": float(pair["test_recall@10"])}
        parents = None
        for level in (1, 2, 3):
            labels = features.labels_for(table, level, index, nodes)
            measured = measure_level(adjacency, degree, labels, rng=rng, settings=SETTINGS)
            audit = measured["audit"]
            aggregate = measured["aggregate"]
            support_rows.append({"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                                 "level": level, **{f"support_{key}": value
                                                    for key, value in audit.items()
                                                    if not isinstance(value, dict)},
                                 "size_histogram": json.dumps(audit.get("size_histogram", {}))})
            record[f"occupied_l{level}"] = audit.get("occupied")
            record[f"singleton_l{level}"] = audit.get("singleton")
            record[f"multi_l{level}"] = audit.get("multi")
            record[f"ge4_l{level}"] = audit.get("ge4")
            record[f"item_coverage_multi_l{level}"] = audit.get("item_coverage_multi")
            record[f"item_coverage_ge4_l{level}"] = audit.get("item_coverage_ge4")
            record[f"size_median_l{level}"] = audit.get("size_median")
            record[f"eligible_groups_l{level}"] = aggregate.get("eligible_groups")
            record[f"sampled_groups_l{level}"] = aggregate.get("sampled_groups")
            record[f"eligible_item_coverage_l{level}"] = aggregate.get("eligible_item_coverage")
            for key in ("transitivity", "cycle_surplus_normalised",
                        "conductance", "largest_component_fraction", "components",
                        "boundary_edges"):
                block = aggregate.get("group_weighted", {}).get(key, {})
                record[f"{key}_l{level}"] = block.get("group_mean")
                record[f"{key}_item_l{level}"] = block.get("item_weighted")
                record[f"{key}_p90_l{level}"] = block.get("p90")
                record[f"{key}_coverage_l{level}"] = block.get("coverage_group")
            four = aggregate.get("four_point", {})
            record[f"four_point_dev_l{level}"] = four.get("mean_deviation")
            record[f"four_point_norm_l{level}"] = four.get("mean_normalised")
            record[f"four_point_samples_l{level}"] = four.get("samples")
            record[f"four_point_groups_l{level}"] = four.get("groups")
            detour = aggregate.get("detour", {})
            record[f"detour_l{level}"] = detour.get("mean_ratio")
            record[f"detour_unavailable_l{level}"] = detour.get("unavailable_fraction")
            for kind in ("global", "stratified"):
                null = null_level(adjacency, degree, labels, None, strata, aggregate,
                                  rng=rng, settings=SETTINGS, kind=kind)
                for key, block in (null.get("features") or {}).items():
                    record[f"{key}_null_{kind}_l{level}"] = block.get("null_mean")
                    record[f"{key}_delta_{kind}_l{level}"] = block.get("delta")
                    record[f"{key}_z_{kind}_l{level}"] = block.get("z")
            if level > 1:
                null = null_level(adjacency, degree, labels, parents, strata, aggregate,
                                  rng=rng, settings=SETTINGS, kind="parent")
                for key, block in (null.get("features") or {}).items():
                    record[f"{key}_delta_parent_l{level}"] = block.get("delta")
                    record[f"{key}_z_parent_l{level}"] = block.get("z")
            parents = labels
        record["hierarchy"] = hierarchy(table, index, adjacency, degree, rng=rng,
                                        settings=SETTINGS)
        for key, value in record["hierarchy"].get("categories", {}).items():
            for name, count in value.items():
                record[f"{key}_{name}"] = count
        records.append(record)
        print(f"  {number + 1}/{len(pairs)} {pair['variant']}", flush=True)

    import csv
    support_fields = []
    for row in support_rows:
        for key in row:
            if key not in support_fields:
                support_fields.append(key)
    with (out / "group_support_L1_L2_L3.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=support_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(support_rows)
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with (out / "geometry_by_level.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (out / "geometry_by_level.json").write_text(json.dumps(records, indent=1) + "\n",
                                                encoding="utf-8")
    print(f"written {len(records)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
