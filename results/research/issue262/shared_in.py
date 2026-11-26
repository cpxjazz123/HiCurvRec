"""Issue #262: directed shared-in/out diagnosis and corrected held-out fidelity.

Read-only. Uses the directed TRAIN graph for relation labels and validation only for
the next-item population; the recommendation test is never read. All negative
candidates are degree/popularity-matched and excluded from the row's history.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue262"
RAW = OUT / "_raw"
sys.path.insert(0, str(ROOT))

from tools.graph_infra import assets  # noqa: E402
from tools.issue253.evaluation_reference import fixed_three_paired_numpy  # noqa: E402
from tools.iteration import analysis as ianalysis  # noqa: E402

ARMS = {
    100: "TIGER_gnn_attention_sectional_rq",
    101: "TIGER_three_expert_attention_rq",
    102: "TIGER_module_budget_prefix_rq",
    103: "TIGER_periodic_budget_prefix_rq",
    104: "TIGER_role_conditioned_l1_rq",
    105: "TIGER_role_conditioned_l3_rq",
    106: "TIGER_role_conditioned_l2_rq",
    107: "TIGER_topology_residual_l2_rq",
}
ROUND = {102: "r01", 103: "r02", 104: "r03", 105: "r04"}
SEED = 42


def row_sets(csr: sp.csr_matrix, node: int) -> set[int]:
    start, end = csr.indptr[int(node)], csr.indptr[int(node) + 1]
    return set(int(x) for x in csr.indices[start:end])


def jaccard(left: set, right: set) -> float | None:
    union = left | right
    return len(left & right) / len(union) if union else None


def prefix_hit(sid3: np.ndarray, anchor: int, item: int, level: int) -> bool:
    return bool(np.array_equal(sid3[anchor, :level], sid3[item, :level]))


def binned(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, np.linspace(0, 1, count + 1))
    edges = np.unique(edges)
    if len(edges) <= 1:
        return np.zeros(len(values), dtype=np.int64)
    return np.clip(np.digitize(values, edges[1:-1], right=True), 0, len(edges) - 2).astype(np.int64)


def build_validation_pairs(valid: pd.DataFrame) -> tuple[list[dict], dict]:
    """One deterministic degree/popularity-matched negative per nonempty history."""
    rows = []
    for record in valid.to_dict("records"):
        history = np.asarray(record["history"], dtype=np.int64).reshape(-1)
        seen = np.asarray(record["seen_history"], dtype=np.int64).reshape(-1)
        if len(history) == 0:
            continue
        target = int(record["target"])
        anchor = int(history[-1])
        rows.append({"user": int(record["user"]), "anchor": anchor, "target": target,
                     "history": history, "seen": seen})
    return rows, {"rows": len(rows), "empty_history": int(len(valid) - len(rows))}


def negative_candidates(rows: list[dict], fwd: sp.csr_matrix, rev: sp.csr_matrix,
                        n_items: int) -> tuple[np.ndarray, dict]:
    in_degree = np.diff(rev.indptr)
    out_degree = np.diff(fwd.indptr)
    total = in_degree + out_degree
    total_bin = binned(total, 10)
    pop_bin = binned(in_degree, 8)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for item in range(n_items):
        buckets[(int(total_bin[item]), int(pop_bin[item]))].append(item)
    for bucket in buckets.values():
        bucket_array = np.asarray(bucket, dtype=np.int64)
        # fixed order; below we sample deterministically per row
        buckets[(int(total_bin[bucket_array[0]]), int(pop_bin[bucket_array[0]]))] = bucket_array
    rng = np.random.default_rng(SEED)
    negatives = np.full(len(rows), -1, dtype=np.int64)
    exact = fallback = 0
    for position, row in enumerate(rows):
        target = row["target"]
        key = (int(total_bin[target]), int(pop_bin[target]))
        forbidden = set(int(x) for x in row["seen"].tolist()) | {row["anchor"], target}
        candidate = buckets.get(key)
        pool = [int(x) for x in candidate] if candidate is not None else []
        pool = [x for x in pool if x not in forbidden]
        if not pool:
            # deterministic fallback: same total-degree decile, then any non-history item
            same_total = np.flatnonzero(total_bin == total_bin[target])
            pool = [int(x) for x in same_total if int(x) not in forbidden]
            fallback += 1
        else:
            exact += 1
        if not pool:
            continue
        negatives[position] = int(pool[int(rng.integers(0, len(pool)))])
    if np.any(negatives < 0):
        raise RuntimeError("negative matching failed for at least one row")
    return negatives, {"exact_degree_popularity_match": exact, "fallback": fallback,
                       "rows": len(rows), "seed": SEED,
                       "total_degree_bins": 10, "in_degree_bins": 8}


def relation_matched_negatives(rows: list[dict], fwd: sp.csr_matrix, rev: sp.csr_matrix,
                               n_items: int) -> tuple[np.ndarray, dict]:
    """Match the target's degree, popularity and direct/reverse/two-hop status."""
    in_degree = np.diff(rev.indptr)
    out_degree = np.diff(fwd.indptr)
    total = in_degree + out_degree
    total_bin = binned(total, 10)
    pop_bin = binned(in_degree, 8)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for item in range(n_items):
        buckets[(int(total_bin[item]), int(pop_bin[item]))].append(item)
    rng = np.random.default_rng(SEED + 1)
    negatives = np.full(len(rows), -1, dtype=np.int64)
    exact = fallback = 0
    for position, row in enumerate(rows):
        target = row["target"]
        anchor = row["anchor"]
        target_features = pair_features(anchor, target, fwd, rev)
        target_flags = (target_features["direct"], target_features["reverse"],
                        target_features["two_hop_hits"] > 0)
        forbidden = set(int(x) for x in row["seen"].tolist()) | {anchor, target}
        key = (int(total_bin[target]), int(pop_bin[target]))
        pool = [int(x) for x in buckets.get(key, []) if int(x) not in forbidden]
        # A deterministic bounded scan keeps the control feasible; the cap is
        # recorded, and it is not a full enumeration of the degree bucket.
        if len(pool) > 64:
            pool = [int(x) for x in rng.choice(np.asarray(pool, dtype=np.int64), 64, replace=False)]
        exact_pool = []
        for item in pool:
            feature = pair_features(anchor, item, fwd, rev)
            if (feature["direct"], feature["reverse"], feature["two_hop_hits"] > 0) == target_flags:
                exact_pool.append(item)
        if exact_pool:
            exact += 1
            pool = exact_pool
        else:
            fallback += 1
            # retain the degree/popularity match but not the relation match
        if not pool:
            same_total = np.flatnonzero(total_bin == total_bin[target])
            pool = [int(x) for x in same_total if int(x) not in forbidden]
        negatives[position] = int(pool[int(rng.integers(0, len(pool)))])
    if np.any(negatives < 0):
        raise RuntimeError("relation-matched negative failed")
    return negatives, {"exact_relation_degree_popularity_match": exact,
                       "fallback_relation": fallback, "rows": len(rows), "seed": SEED + 1,
                       "candidate_cap": 64}


def pair_features(anchor: int, item: int, fwd: sp.csr_matrix, rev: sp.csr_matrix) -> dict:
    pred_anchor = row_sets(rev, anchor)
    pred_item = row_sets(rev, item)
    succ_anchor = row_sets(fwd, anchor)
    succ_item = row_sets(fwd, item)
    j_in = jaccard(pred_anchor, pred_item)
    j_out = jaccard(succ_anchor, succ_item)
    two_hop = len(np.intersect1d(
        fwd.indices[fwd.indptr[anchor]:fwd.indptr[anchor + 1]],
        rev.indices[rev.indptr[item]:rev.indptr[item + 1]]))
    return {
        "j_in": j_in,
        "j_out": j_out,
        "common_in": bool(pred_anchor & pred_item),
        "common_out": bool(succ_anchor & succ_item),
        "direct": bool(fwd[anchor, item]),
        "reverse": bool(fwd[item, anchor]),
        "two_hop_hits": int(two_hop),
        "pred_anchor": len(pred_anchor),
        "pred_item": len(pred_item),
        "succ_anchor": len(succ_anchor),
        "succ_item": len(succ_item),
    }


def safe_corr(x, y, rank=False):
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3:
        return {"n": len(pairs), "value": None}
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    value = spearmanr(xs, ys).statistic if rank else pearsonr(xs, ys).statistic
    return {"n": len(pairs), "value": float(value) if value is not None and math.isfinite(float(value)) else None}


def bootstrap_diff(target: np.ndarray, negative: np.ndarray, seed: int = SEED,
                   draws: int = 2000) -> dict:
    target = np.asarray(target, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    values = target - negative
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"n": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = float(finite[rng.integers(0, len(finite), len(finite))].mean())
    return {"n": int(len(finite)), "mean": float(finite.mean()),
            "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975)),
            "draws": draws}


def bass_rates(values: np.ndarray, labels: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    out = {}
    for name, mask in (("all", np.ones(len(values), dtype=bool)),):
        selected = values[mask]
        out[name] = {"n": int(len(selected)), "mean": float(selected.mean()) if len(selected) else None,
                     "positive": float((selected > 0).mean()) if len(selected) else None}
    return out


def load_q(index: int) -> dict:
    path = RAW / "q" / f"{index}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def logistic_controls(rows: list[dict], negatives: np.ndarray, sid3: np.ndarray,
                      q: np.ndarray, scale: float, fwd: sp.csr_matrix,
                      rev: sp.csr_matrix) -> dict:
    """Descriptive logistic check; target=1 vs its matched negative=0."""
    try:
        import statsmodels.api as sm
    except Exception as exc:  # noqa: BLE001
        return {"status": "NA", "reason": repr(exc)}
    in_degree = np.diff(rev.indptr)
    out_degree = np.diff(fwd.indptr)
    anchors = np.asarray([r["anchor"] for r in rows], dtype=np.int64)
    targets = np.asarray([r["target"] for r in rows], dtype=np.int64)
    features = []
    labels = []
    clusters = []
    for i, row in enumerate(rows):
        anchor = row["anchor"]
        for label, item in ((1, targets[i]), (0, int(negatives[i]))):
            p = pair_features(anchor, item, fwd, rev)
            features.append([
                1.0,
                1.0 if p["common_in"] else 0.0,
                1.0 if p["common_out"] else 0.0,
                1.0 if p["direct"] else 0.0,
                1.0 if p["reverse"] else 0.0,
                float(p["j_in"] if p["j_in"] is not None else 0.0),
                float(p["j_out"] if p["j_out"] is not None else 0.0),
                float(prefix_hit(sid3, anchor, item, 1)),
                float(prefix_hit(sid3, anchor, item, 2)),
                float(prefix_hit(sid3, anchor, item, 3)),
                float(in_degree[item]), float(out_degree[item]),
                float(in_degree[anchor]), float(out_degree[anchor]),
                float(fixed_three_paired_numpy(q[[anchor]], q[[item]], scale=scale)[0]),
            ])
            labels.append(label)
            clusters.append(anchor)
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    # standardize continuous columns only; binary columns remain interpretable
    names = ["intercept", "common_in", "common_out", "direct", "reverse",
             "j_in", "j_out", "prefix_L1", "prefix_L2", "prefix_L3",
             "in_degree_item", "out_degree_item", "in_degree_anchor",
             "out_degree_anchor", "fixed3_distance"]
    for column in range(X.shape[1]):
        if column in (0, 1, 2, 3, 4, 7, 8, 9):
            continue
        std = X[:, column].std()
        if std > 0:
            X[:, column] = (X[:, column] - X[:, column].mean()) / std
    model = sm.Logit(y, X)
    keep = [0] + [i for i in range(1, X.shape[1]) if X[:, i].std() > 0]
    X = X[:, keep]
    names = [names[i] for i in keep]
    try:
        fit = model.fit(disp=False, maxiter=200, cov_type="cluster",
                        cov_kwds={"groups": np.asarray(clusters)})
        params = fit.params
        bse = fit.bse
        pvalues = fit.pvalues
        method = "cluster-robust logistic"
    except Exception:  # singular/perfect separation: keep a descriptive fit, no p-values
        fit = sm.Logit(y, X).fit_regularized(alpha=1e-3, L1_wt=0.0, maxiter=500,
                                             disp=False)
        params = fit.params
        bse = np.full(len(params), np.nan)
        pvalues = np.full(len(params), np.nan)
        method = "L2-regularized logistic fallback (singular Hessian)"
    return {"status": "PASS", "n_pairs": int(len(y)), "features": names,
            "coef": dict(zip(names, np.asarray(params).tolist())),
            "std_err": dict(zip(names, np.asarray(bse).tolist())),
            "p": dict(zip(names, np.asarray(pvalues).tolist())),
            "odds_ratio": {name: float(math.exp(value)) for name, value in zip(names, np.asarray(params).tolist())},
            "method": method,
            "scope": "descriptive paired target-vs-matched-negative logistic; cluster by anchor; not causal"}


def true_heldout_score(index: int, bank_name: str, pairs: np.ndarray,
                       true_j: np.ndarray, q: np.ndarray, scale: float) -> dict:
    """Within-arm Spearman(distance, true directed Jaccard) and a binary AUC."""
    valid = np.isfinite(true_j)
    if valid.sum() < 3:
        return {"n": int(valid.sum()), "spearman": None, "auc": None}
    source = pairs[valid]
    distance = fixed_three_paired_numpy(q[source[:, 0]], q[source[:, 1]], scale=scale)
    labels = true_j[valid]
    rho = spearmanr(labels, distance).statistic
    # AUC that a higher Jaccard has a smaller distance, using rank ordering.
    order = np.argsort(distance)
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    pos = labels > 0
    neg = labels <= 0
    auc = None
    if pos.any() and neg.any():
        # Mann-Whitney P(rank of positive > rank of negative); distance decreasing
        auc = float((ranks[pos].mean() - (pos.sum() - 1) / 2) / neg.sum())
    return {"n": int(valid.sum()), "spearman": float(rho), "auc": auc,
            "mean_distance": float(distance.mean()), "mean_true_j": float(labels.mean())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    directed = ROOT / "results/research/behavior_graph_directed"
    edge_table = pd.read_parquet(directed / "edges.parquet", columns=["source", "target", "weight"])
    fwd = sp.coo_matrix((edge_table["weight"].to_numpy(dtype=np.float64),
                         (edge_table["source"].to_numpy(), edge_table["target"].to_numpy())),
                        shape=(24587, 24587)).tocsr()
    rev = fwd.T.tocsr()
    valid = pd.read_parquet(ROOT / "results/stage0_build_parquet/valid.parquet")
    rows, population = build_validation_pairs(valid)
    negatives, matching = negative_candidates(rows, fwd, rev, 24587)
    relation_negatives, relation_matching = relation_matched_negatives(rows, fwd, rev, 24587)
    anchors = np.asarray([r["anchor"] for r in rows], dtype=np.int64)
    targets = np.asarray([r["target"] for r in rows], dtype=np.int64)
    print("population", population, "matching", matching,
          "relation_matching", relation_matching, flush=True)

    # relation features are arm-independent
    target_features = [pair_features(int(a), int(t), fwd, rev) for a, t in zip(anchors, targets)]
    negative_features = [pair_features(int(a), int(n), fwd, rev) for a, n in zip(anchors, negatives)]
    relation_negative_features = [pair_features(int(a), int(n), fwd, rev)
                                  for a, n in zip(anchors, relation_negatives)]
    common_in_target = np.asarray([x["common_in"] for x in target_features], dtype=bool)
    common_in_negative = np.asarray([x["common_in"] for x in negative_features], dtype=bool)
    common_in_relation = np.asarray([x["common_in"] for x in relation_negative_features], dtype=bool)
    common_out_target = np.asarray([x["common_out"] for x in target_features], dtype=bool)
    common_out_negative = np.asarray([x["common_out"] for x in negative_features], dtype=bool)
    common_out_relation = np.asarray([x["common_out"] for x in relation_negative_features], dtype=bool)
    j_in_target = np.asarray([x["j_in"] for x in target_features], dtype=object)
    j_in_negative = np.asarray([x["j_in"] for x in negative_features], dtype=object)

    # degree/popularity strata for the paired common-in result
    in_degree = np.diff(rev.indptr)
    out_degree = np.diff(fwd.indptr)
    total_degree = in_degree + out_degree
    degree_bins = binned(total_degree[anchors], 10)
    strata = {}
    for bin_id in np.unique(degree_bins):
        mask = degree_bins == bin_id
        pos = common_in_target[mask]
        neg = common_in_negative[mask]
        strata[str(int(bin_id))] = {
            "rows": int(mask.sum()),
            "target_rate": float(pos.mean()) if len(pos) else None,
            "negative_rate": float(neg.mean()) if len(neg) else None,
            "difference": float(pos.mean() - neg.mean()) if len(pos) else None,
        }

    arm_rows = {}
    for index, arm in ARMS.items():
        q = load_q(index)
        q_values = q["q"].astype(np.float64)
        scale = float(q["scale"])
        sid3 = q["sid3"].astype(np.int64)
        # per-arm prefix shares
        pref = {}
        for level in (1, 2, 3):
            pref[f"target_L{level}"] = np.asarray(
                [prefix_hit(sid3, int(a), int(t), level) for a, t in zip(anchors, targets)], dtype=bool)
            pref[f"negative_L{level}"] = np.asarray(
                [prefix_hit(sid3, int(a), int(n), level) for a, n in zip(anchors, negatives)], dtype=bool)
        # per-arm distance to the anchor
        distance_target = np.empty(len(rows), dtype=np.float64)
        distance_negative = np.empty(len(rows), dtype=np.float64)
        for start in range(0, len(rows), 2048):
            end = min(start + 2048, len(rows))
            distance_target[start:end] = fixed_three_paired_numpy(
                q_values[anchors[start:end]], q_values[targets[start:end]], scale=scale)
            distance_negative[start:end] = fixed_three_paired_numpy(
                q_values[anchors[start:end]], q_values[negatives[start:end]], scale=scale)
        arm_rows[index] = {"arm": arm, "distance_target": distance_target,
                           "distance_negative": distance_negative,
                           "prefix": pref, "q": q}
        # logistic model for this arm; not used as a causal effect
        try:
            lp = logistic_controls(rows, negatives, sid3, q_values, scale, fwd, rev)
        except Exception as exc:  # noqa: BLE001
            lp = {"status": "NA", "reason": repr(exc)}
        arm_rows[index]["logistic"] = lp

    # primary paired rates (arm-independent graph relation)
    primary = {
        "population": population,
        "negative_matching": matching,
        "common_in": {
            "target_rate": float(common_in_target.mean()),
            "negative_rate": float(common_in_negative.mean()),
            "paired_difference": bootstrap_diff(common_in_target.astype(float),
                                                common_in_negative.astype(float)),
            "target_count": int(common_in_target.sum()),
            "negative_count": int(common_in_negative.sum()),
            "n": int(len(rows)),
        },
        "common_out": {
            "target_rate": float(common_out_target.mean()),
            "negative_rate": float(common_out_negative.mean()),
            "paired_difference": bootstrap_diff(common_out_target.astype(float),
                                                common_out_negative.astype(float)),
            "target_count": int(common_out_target.sum()),
            "negative_count": int(common_out_negative.sum()),
        },
        "degree_strata": strata,
        "relation_matched": {
            "definition": "negative matched on degree/popularity and direct/reverse/two-hop status",
            "matching": relation_matching,
            "common_in": {
                "target_rate": float(common_in_target.mean()),
                "negative_rate": float(common_in_relation.mean()),
                "paired_difference": bootstrap_diff(common_in_target.astype(float),
                                                    common_in_relation.astype(float)),
                "target_count": int(common_in_target.sum()),
                "negative_count": int(common_in_relation.sum()),
            },
            "common_out": {
                "target_rate": float(common_out_target.mean()),
                "negative_rate": float(common_out_relation.mean()),
                "paired_difference": bootstrap_diff(common_out_target.astype(float),
                                                    common_out_relation.astype(float)),
            },
        },
        "no_direct_reverse_two_hop": {},
        "direct": {"target_rate": float(np.mean([x["direct"] for x in target_features])),
                   "negative_rate": float(np.mean([x["direct"] for x in negative_features]))},
        "reverse": {"target_rate": float(np.mean([x["reverse"] for x in target_features])),
                    "negative_rate": float(np.mean([x["reverse"] for x in negative_features]))},
        "two_hop_hits_mean": {
            "target": float(np.mean([x["two_hop_hits"] for x in target_features])),
            "negative": float(np.mean([x["two_hop_hits"] for x in negative_features])),
        },
    }
    # Jaccard continuous summaries (None retained in count)
    for name, values in (("target", j_in_target), ("negative", j_in_negative)):
        vals = np.asarray([float(x) for x in values if x is not None], dtype=np.float64)
        primary[f"j_in_{name}"] = {"n": int(len(vals)), "missing": int(len(values) - len(vals)),
                                    "mean": float(vals.mean()) if len(vals) else None,
                                    "median": float(np.median(vals)) if len(vals) else None,
                                    "positive_fraction": float((vals > 0).mean()) if len(vals) else None}
    absent = np.asarray([
        (not t["direct"] and not t["reverse"] and t["two_hop_hits"] == 0)
        and (not n["direct"] and not n["reverse"] and n["two_hop_hits"] == 0)
        for t, n in zip(target_features, negative_features)
    ], dtype=bool)
    primary["no_direct_reverse_two_hop"] = {
        "rows": int(absent.sum()),
        "target_common_in_rate": float(common_in_target[absent].mean()) if absent.any() else None,
        "negative_common_in_rate": float(common_in_negative[absent].mean()) if absent.any() else None,
        "paired_difference": bootstrap_diff(common_in_target[absent].astype(float),
                                            common_in_negative[absent].astype(float)),
        "scope": "both target and its degree/popularity negative have no direct edge, reverse edge or two-hop chain",
    }

    # Per-arm paired prefix/distance summaries and arm-level scores.
    arm_summary = {}
    for index, arm in ARMS.items():
        data = arm_rows[index]
        pref = data["prefix"]
        summary = {"arm": arm, "index": index,
                   "distance": {
                       "target_mean": float(data["distance_target"].mean()),
                       "negative_mean": float(data["distance_negative"].mean()),
                       "paired_difference": bootstrap_diff(data["distance_target"], data["distance_negative"]),
                   },
                   "prefix": {}}
        for level in (1, 2, 3):
            t = pref[f"target_L{level}"]
            n = pref[f"negative_L{level}"]
            summary["prefix"][f"L{level}"] = {
                "target_rate": float(t.mean()), "negative_rate": float(n.mean()),
                "paired_difference": bootstrap_diff(t.astype(float), n.astype(float)),
                "target_given_common_in": float(t[common_in_target].mean()) if common_in_target.any() else None,
                "negative_given_common_in": float(n[common_in_negative].mean()) if common_in_negative.any() else None,
            }
        summary["logistic"] = data["logistic"]
        arm_summary[str(index)] = summary

    # Correct held-out fidelity on the graph that generated the bank.
    graph_base = ROOT / "results/research/infra246/assets"
    heldout = {}
    for split in ("struct_test", "struct_val"):
        split_fwd = assets.load_sparse(graph_base / "graph" / f"{split}_forward.npz")
        split_rev = assets.load_sparse(graph_base / "graph" / f"{split}_reverse.npz")
        for group in ("shared_in", "shared_out"):
            bank = np.load(graph_base / "pairs" / f"strict_{split}_{group}.npz", allow_pickle=False)
            pairs = np.asarray(bank["pairs"], dtype=np.int64)
            true_j = []
            visible_j = []
            for u, v in pairs.tolist():
                true_j.append(jaccard(row_sets(split_rev, u), row_sets(split_rev, v))
                              if group == "shared_in" else
                              jaccard(row_sets(split_fwd, u), row_sets(split_fwd, v)))
                visible_j.append(float(bank[f"jaccard_{'in' if group == 'shared_in' else 'out'}"][
                    len(true_j) - 1]))
            true_j = np.asarray([np.nan if x is None else x for x in true_j], dtype=np.float64)
            visible_j = np.asarray(visible_j, dtype=np.float64)
            per_arm = {}
            for index, arm in ARMS.items():
                q = arm_rows[index]["q"]
                score = true_heldout_score(index, f"{split}_{group}", pairs, true_j,
                                           q["q"].astype(np.float64), float(q["scale"]))
                per_arm[str(index)] = score
            scores = [per_arm[str(i)]["spearman"] for i in ARMS]
            test_r10 = {}
            for i, arm in ARMS.items():
                final = sorted((ROOT / "results/stage3_T5Train" / arm).glob("**/test_final.json"))[0]
                test_r10[str(i)] = json.loads(final.read_text()).get("test_recall@10")
            heldout[f"{split}_{group}"] = {
                "bank": str(graph_base / "pairs" / f"strict_{split}_{group}.npz"),
                "pairs": int(len(pairs)),
                "true_feature_sha": "recomputed from " + split + " directed graph",
                "visible_feature_used_by_261": "bank_features.compute on visible graph",
                "true_j_summary": {"mean": float(np.nanmean(true_j)), "positive_fraction": float(np.nanmean(true_j > 0)),
                                   "missing": int(np.isnan(true_j).sum())},
                "visible_j_summary": {"mean": float(np.nanmean(visible_j)), "positive_fraction": float(np.nanmean(visible_j > 0)),
                                      "missing": int(np.isnan(visible_j).sum())},
                "per_arm": per_arm,
                "cross_arm_test_R10": {
                    "pearson": safe_corr(scores, [test_r10[str(i)] for i in ARMS])["value"],
                    "spearman": safe_corr(scores, [test_r10[str(i)] for i in ARMS], rank=True)["value"],
                    "scores": dict(zip([str(i) for i in ARMS], scores)),
                    "test_R10": test_r10,
                },
                "scope": "structure-test/validation population; not the recommendation test population",
            }
            # original #261 visible feature correlation to the same distance, for context
            visible_scores = {}
            for index, arm in ARMS.items():
                q = arm_rows[index]["q"]
                visible_scores[str(index)] = true_heldout_score(index, f"{split}_{group}", pairs,
                                                               visible_j, q["q"].astype(np.float64),
                                                               float(q["scale"]))["spearman"]
            heldout[f"{split}_{group}"]["visible_feature_cross_arm"] = {
                "pearson": safe_corr([visible_scores[str(i)] for i in ARMS],
                                     [test_r10[str(i)] for i in ARMS])["value"],
                "spearman": safe_corr([visible_scores[str(i)] for i in ARMS],
                                      [test_r10[str(i)] for i in ARMS], rank=True)["value"],
                "scores": visible_scores,
            }

    # Per-arm scatter artifact (large; ignored by .gitignore) and compact CSV.
    RAW.mkdir(parents=True, exist_ok=True)
    scatter_path = RAW / "sharedin_pairs.npz"
    np.savez_compressed(
        scatter_path,
        anchor=anchors.astype(np.int32), target=targets.astype(np.int32),
        negative=negatives.astype(np.int32),
        relation_negative=relation_negatives.astype(np.int32),
        common_in_target=common_in_target, common_in_negative=common_in_negative,
        common_in_relation=common_in_relation,
        common_out_target=common_out_target, common_out_negative=common_out_negative,
        common_out_relation=common_out_relation,
        j_in_target=np.asarray([np.nan if x is None else x for x in j_in_target], dtype=np.float64),
        j_in_negative=np.asarray([np.nan if x is None else x for x in j_in_negative], dtype=np.float64),
        **{f"distance_target_{i}": arm_rows[i]["distance_target"].astype(np.float32) for i in ARMS},
        **{f"distance_negative_{i}": arm_rows[i]["distance_negative"].astype(np.float32) for i in ARMS},
    )
    result = {
        "schema": "genrec.issue262.shared_in/1",
        "source": {
            "valid": str(ROOT / "results/stage0_build_parquet/valid.parquet"),
            "directed_graph": str(directed),
            "directed_metadata": json.loads((directed / "metadata.json").read_text()),
        },
        "negative_definition": "one item matched on directed total-degree decile and in-degree octile, excluding the row's seen/history/anchor/target",
        "primary": primary,
        "arms": arm_summary,
        "heldout_fidelity": heldout,
        "counts": {
            "validation_rows": len(rows),
            "with_common_in_target": int(common_in_target.sum()),
            "with_common_in_negative": int(common_in_negative.sum()),
            "j_in_target_missing": int(np.isnan(np.asarray([np.nan if x is None else x for x in j_in_target], dtype=np.float64)).sum()),
            "j_in_negative_missing": int(np.isnan(np.asarray([np.nan if x is None else x for x in j_in_negative], dtype=np.float64)).sum()),
        },
    }
    (OUT / "SHARED_IN.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"primary_common_in": primary["common_in"],
                      "common_out": primary["common_out"],
                      "heldout": {k: {"true": v["true_j_summary"],
                                      "visible": v["visible_j_summary"],
                                      "cross": v["cross_arm_test_R10"]} for k, v in heldout.items()},
                      "scatter": str(scatter_path)}, indent=2))


if __name__ == "__main__":
    main()
