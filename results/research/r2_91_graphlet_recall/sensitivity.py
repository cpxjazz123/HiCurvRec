"""Issue #218 step 3b: subsets, controls, regression increment and the 67-arm cross-check."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
OLD = REPO / "analysis_sid_recall" / "results" / "structure_vs_recall.json"
TYPES = ("K3", "P3", "C4", "P4", "K1_3")
OLD_OF = {"K3": "triangle", "P3": "path3", "C4": "cycle4", "P4": "path4", "K1_3": "star4"}


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    out = np.empty(len(values), dtype=float)
    out[order] = np.arange(len(values))
    return out


def spearman(x, y) -> float:
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    rx -= rx.mean(); ry -= ry.mean()
    denominator = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denominator) if denominator else float("nan")


def permutation_p(x, y, rounds=5000, seed=218) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(spearman(x, y))
    null = np.array([abs(spearman(x, rng.permutation(y))) for _ in range(rounds)])
    return float((null >= observed).mean())


def loo_r2(features: np.ndarray, target: np.ndarray) -> float:
    """Leave-one-out R^2 of a linear model (features already include an intercept)."""
    predicted = np.empty(len(target))
    for index in range(len(target)):
        keep = np.arange(len(target)) != index
        beta = np.linalg.lstsq(features[keep], target[keep], rcond=None)[0]
        predicted[index] = features[index] @ beta
    residual = ((target - predicted) ** 2).sum()
    total = ((target - target.mean()) ** 2).sum()
    return float(1 - residual / total)


def load() -> tuple[dict, dict, dict]:
    meta, gains, globals_ = {}, defaultdict(dict), {}
    with (OUT / "paired_arms.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["recall"]:
                meta[row["number"]] = {"recall": float(row["recall"]), "arm": row["arm"],
                                       "sid": row["sid_sha256"], "collision": float(row["collision"])}
    with (OUT / "per_arm_retention.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["depth"] == "L1":
                gains[row["number"]][row["type"]] = {"gain_pair": float(row["gain_pair"]),
                                                     "gain_all": float(row["gain_all"])}
    with (OUT / "per_arm_global.csv").open() as handle:
        for row in csv.DictReader(handle):
            globals_[row["number"]] = {"L1_max_share": float(row["L1_max_share"]),
                                       "edge_L1": float(row["edge_L1"])}
    return meta, gains, globals_


def family(arm: str) -> str:
    for prefix in ("TIGER_graphlet_role", "TIGER_g4_closure", "TIGER_role_geom", "TIGER_Uniform_KNN",
                   "TIGER_graphlet_codebook", "TIGER_mixed_rq", "TIGER_RQ-VAE", "TIGER_af3",
                   "TIGER_behavior_edge", "TIGER_sinkhorn", "TIGER_c17", "TIGER_c1", "TIGER_curv"):
        if arm.startswith(prefix):
            return prefix
    return "other"


def main() -> None:
    meta, gains, globals_ = load()
    numbers = sorted(meta)

    def subset(predicate):
        return [n for n in numbers if predicate(n)]

    def correlate(selected, metric="gain_pair"):
        result = {}
        for graphlet in TYPES:
            rows = [n for n in selected if graphlet in gains.get(n, {})]
            if len(rows) < 5:
                continue
            x = [gains[n][graphlet][metric] for n in rows]
            y = [meta[n]["recall"] for n in rows]
            result[graphlet] = {"n": len(rows), "rho": spearman(x, y), "p": permutation_p(x, y)}
        return result

    groups = defaultdict(list)
    for number in numbers:
        groups[meta[number]["sid"]].append(number)
    dedup_values = {}
    for sid, members in groups.items():
        first = meta[members[0]]
        dedup_values[sid] = {"recall": float(np.mean([meta[m]["recall"] for m in members])),
                             "collision": float(np.mean([meta[m]["collision"] for m in members])),
                             "gain_pair": {t: float(np.mean([gains[m][t]["gain_pair"] for m in members]))
                                           for t in TYPES if all(t in gains.get(m, {}) for m in members)}}

    def correlate_dedup(metric="gain_pair"):
        result = {}
        for graphlet in TYPES:
            rows = [v for v in dedup_values.values() if graphlet in v["gain_pair"]]
            x = [v["gain_pair"][graphlet] for v in rows]
            y = [v["recall"] for v in rows]
            result[graphlet] = {"n": len(rows), "rho": spearman(x, y), "p": permutation_p(x, y)}
        return result

    families = sorted({family(meta[n]["arm"]) for n in numbers})
    sensitivity = {
        "all": correlate(numbers),
        "dedup_sid": correlate_dedup(),
        "collision_lt_090": correlate(subset(lambda n: meta[n]["collision"] < 0.90)),
        "max_share_le_050": correlate(subset(lambda n: globals_[n]["L1_max_share"] <= 0.50)),
        "numbers_le_68": correlate(subset(lambda n: int(n) <= 68)),
        "numbers_69_91": correlate(subset(lambda n: int(n) >= 69)),
        "leave_one_family_out": {fam: correlate([n for n in numbers if family(meta[n]["arm"]) != fam])
                                 for fam in families},
        "family": {fam: correlate([n for n in numbers if family(meta[n]["arm"]) == fam]) for fam in families},
    }

    regression = {}
    for graphlet in TYPES:
        rows = [n for n in numbers if graphlet in gains.get(n, {})]
        y = np.array([meta[n]["recall"] for n in rows])
        base = np.column_stack([np.ones(len(rows)), [meta[n]["collision"] for n in rows]])
        extended = np.column_stack([base, [gains[n][graphlet]["gain_pair"] for n in rows]])
        regression[graphlet] = {"n": len(rows), "loo_r2_collision": loo_r2(base, y),
                                "loo_r2_collision_plus_gain": loo_r2(extended, y)}
        regression[graphlet]["delta"] = (regression[graphlet]["loo_r2_collision_plus_gain"]
                                         - regression[graphlet]["loo_r2_collision"])

    old = json.loads(OLD.read_text())
    old_numbers = [row["number"] for row in old["rows"]]
    historical = {"n_old": len(old_numbers), "rows": {}}
    for graphlet in TYPES:
        rows = [n for n in old_numbers if n in meta and graphlet in gains.get(n, {})]
        x = [gains[n][graphlet]["gain_pair"] for n in rows]
        y = [meta[n]["recall"] for n in rows]
        old_block = (old.get("all", {}).get("structures", {}).get(OLD_OF[graphlet], {}) or {}).get("gain", {})
        historical["rows"][graphlet] = {
            "n": len(rows),
            "new_metric_rho": spearman(x, y),
            "old_metric_rho": old_block.get("rho"),
            "old_n": old_block.get("n"),
        }

    payload = {"sensitivity": sensitivity, "regression_loo": regression, "historical_67": historical,
               "families": families}
    (OUT / "sensitivity.json").write_text(json.dumps(payload, indent=1))
    for graphlet in TYPES:
        all_row = sensitivity["all"][graphlet]
        dedup_row = sensitivity["dedup_sid"][graphlet]
        reg = regression[graphlet]
        hist = historical["rows"][graphlet]
        print(f"{graphlet:5s} all rho={all_row['rho']:+.3f} (n={all_row['n']}) | "
              f"dedup rho={dedup_row['rho']:+.3f} (n={dedup_row['n']}) | "
              f"dR2={reg['delta']:+.4f} | old67 rho_new={hist['new_metric_rho']:+.3f} "
              f"rho_old={hist['old_metric_rho']:+.3f}")


if __name__ == "__main__":
    main()
