"""Graphlet Role x relation x SID prefix x predictive value -- the descriptive analysis.

Reads the edge table, the item table, and the frozen T5 scores; writes `analysis.json` and prints
the Markdown tables used by `REPORT.md`.  No training, no parameter update, no SID rewrite.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/graphlet_relation_sid/analyse.py

Three questions, in the same order as the owner's plan:

  1. what relation does an edge inside a given local structure carry (semantic / title / co-occurrence
     / directed transition / future behaviour)?
  2. how do those relations map onto the existing SID prefix (LCP 0..3), after matching degree,
     popularity and semantic similarity?
  3. does the frozen T5 checkpoint predict the target better for one relation class than another
     (recall@10, MRR@10, teacher-forced log-probability), again after matching popularity / history
     length / semantic similarity?
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

pd.set_option("future.no_silent_downcasting", True)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EDGES = HERE / "edge_features.parquet"
ITEMS = HERE / "item_features.parquet"
SCORES = HERE / "test_scores.parquet"
SCORE_META = HERE / "test_scores_meta.json"
GRAPHLETS = HERE / "graphlet_names.json"
STATS = HERE / "graph_stats.json"
TEST = REPO / "results" / "stage0_build_parquet" / "test.parquet"
ITEM_TABLE = REPO / "results" / "stage0_build_parquet" / "items.parquet"
EMB = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = HERE / "analysis.json"
MARKDOWN = HERE / "tables.md"

STOPWORDS = set("""
a an the and or for with without of to in on at by from as is are was were be been being this that
these those it its his her their our your my you we they he she them him us i not no nor but if then
than so such into over under between within per each other more most less least very can could may
might will would should do does did done has have had having use used using new set pack piece pieces
""".split())

#: controls used for the T5 adjustment; semantic similarity is removed from the list when the
#: grouping variable itself is semantic similarity
TEST_CONTROLS = [
    "log_target_pop", "log_last_pop", "log_target_degree", "log_last_degree",
    "history_length", "sem_cos", "title_jaccard",
]


def tokenize(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(title).lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 1}


def title_jaccard(titles, left, right):
    sets = [tokenize(title) for title in titles]
    sizes = np.array([len(value) for value in sets], dtype=np.float64)
    out = np.zeros(len(left), dtype=np.float64)
    for index, (a, b) in enumerate(zip(left.tolist(), right.tolist())):
        first, second = sets[a], sets[b]
        union = sizes[a] + sizes[b] - len(first & second)
        out[index] = len(first & second) / union if union else 0.0
    return out


def semantic_cosine(embeddings, left, right):
    norms = np.linalg.norm(embeddings, axis=1)
    unit = embeddings / np.clip(norms, 1e-12, None)[:, None]
    out = np.zeros(len(left), dtype=np.float64)
    step = 50_000
    for start in range(0, len(left), step):
        stop = min(start + step, len(left))
        out[start:stop] = np.einsum("ij,ij->i", unit[left[start:stop]], unit[right[start:stop]])
    return out


def decile(series: pd.Series, q: int = 10) -> pd.Series:
    return pd.qcut(series.rank(method="first"), q=q, labels=False, duplicates="drop")


def bin_by_edges(series: pd.Series, edges: np.ndarray) -> pd.Series:
    """Assign a continuous column to fixed quantile bins learned on the TRAIN edges."""
    return pd.cut(series, bins=edges, labels=False, include_lowest=True)


def prop_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


STRATA_CACHE: dict[tuple, np.ndarray] = {}


def _strata_codes(frame: pd.DataFrame, strata: list[str]) -> np.ndarray:
    """One integer cell code per row, cached because every group reuses the same strata."""
    key = (id(frame), tuple(strata))
    if key not in STRATA_CACHE:
        values = frame[strata].to_numpy()
        base_keep = ~pd.isna(values).any(axis=1)
        _, codes = np.unique(values[base_keep], axis=0, return_inverse=True)
        full = np.full(len(frame), -1, dtype=np.int64)
        full[base_keep] = codes
        STRATA_CACHE[key] = full
    return STRATA_CACHE[key]


def matched_diff(frame: pd.DataFrame, mask: np.ndarray, value: str, strata: list[str]) -> dict:
    """Weighted within-stratum difference between the masked edges and the rest.

    Cells where either side has fewer than five rows are dropped; the weight is the harmonic-like
    n1*n0/(n1+n0), so a cell contributes only as much as its smaller side can support.
    """
    values = frame[value].to_numpy(dtype=np.float64)
    full_codes = _strata_codes(frame, strata)
    keep = np.isfinite(values) & (full_codes >= 0)
    codes = full_codes[keep]
    values = values[keep]
    group = mask[keep].astype(np.float64)
    n_cells = int(codes.max()) + 1 if len(codes) else 0
    count1 = np.bincount(codes, weights=group, minlength=n_cells)
    count0 = np.bincount(codes, weights=1.0 - group, minlength=n_cells)
    sum1 = np.bincount(codes, weights=values * group, minlength=n_cells)
    sum0 = np.bincount(codes, weights=values * (1.0 - group), minlength=n_cells)
    sum_sq1 = np.bincount(codes, weights=values * values * group, minlength=n_cells)
    sum_sq0 = np.bincount(codes, weights=values * values * (1.0 - group), minlength=n_cells)
    valid = (count1 >= 5) & (count0 >= 5)
    if not valid.any():
        return {"diff": float("nan"), "se": float("nan"), "cells": 0, "n_masked": int(mask.sum())}
    mean1 = np.divide(sum1, count1, out=np.zeros_like(sum1), where=count1 > 0)
    mean0 = np.divide(sum0, count0, out=np.zeros_like(sum0), where=count0 > 0)
    var1 = np.divide(sum_sq1, count1, out=np.zeros_like(sum1), where=count1 > 0) - mean1 ** 2
    var0 = np.divide(sum_sq0, count0, out=np.zeros_like(sum0), where=count0 > 0) - mean0 ** 2
    var1 = np.clip(var1, 0.0, None)
    var0 = np.clip(var0, 0.0, None)
    weight = count1 * count0 / (count1 + count0)
    total = weight[valid].sum()
    difference = (weight[valid] * (mean1[valid] - mean0[valid])).sum() / total
    cell_var = (
        np.divide(var1, count1, out=np.zeros_like(var1), where=count1 > 0)
        + np.divide(var0, count0, out=np.zeros_like(var0), where=count0 > 0)
    )
    se = math.sqrt((((weight[valid] / total) ** 2) * cell_var[valid]).sum())
    return {
        "diff": float(difference),
        "se": float(se),
        "cells": int(valid.sum()),
        "n_masked": int(mask.sum()),
    }


def residual_delta(frame: pd.DataFrame, mask: np.ndarray, outcome: str, controls: list[str]) -> dict:
    """Group-minus-rest difference after an OLS adjustment on the controls.

    A linear probability model for the binary recall outcome: this is a descriptive adjustment, not
    a causal estimate, and the report says so.
    """
    data = frame[[outcome, *controls]].dropna()
    keep_index = data.index
    mask = pd.Series(mask, index=frame.index).loc[keep_index].to_numpy(dtype=bool)
    y = data[outcome].to_numpy(dtype=np.float64)
    x = StandardScaler().fit_transform(data[controls].to_numpy(dtype=np.float64))
    model = LinearRegression().fit(x, y)
    residual = y - model.predict(x)
    if mask.sum() < 5 or (~mask).sum() < 5:
        return {"diff": float("nan"), "se": float("nan")}
    left = residual[mask]
    right = residual[~mask]
    se = math.sqrt(left.var(ddof=1) / max(len(left), 1) + right.var(ddof=1) / max(len(right), 1))
    return {"diff": float(left.mean() - right.mean()), "se": float(se)}


def summarize_edge(frame: pd.DataFrame, mask: np.ndarray, strata: list[str], exclude: tuple[str, ...] = ()) -> dict:
    subset = frame.loc[mask]
    row = {
        "n": int(mask.sum()),
        "share": float(mask.mean()),
        "sem_cos": float(subset["sem_cos"].mean()),
        "title_jaccard": float(subset["title_jaccard"].mean()),
        "cooc_users": float(subset["cooc_users"].mean()),
        "p_trans_max": float(subset["p_trans_max"].mean()),
        "same_community": float(subset["same_community"].mean()),
        "triangles": float(subset["triangles"].mean()),
        "future_hit": float(subset["future_hit"].mean()),
        "future_lift": float(subset["future_lift_max"].mean()),
        "lcp": float(subset["lcp"].mean()),
        "prefix1": float(subset["prefix1"].mean()),
        "prefix2": float(subset["prefix2"].mean()),
        "prefix3": float(subset["prefix3"].mean()),
        "q_l2": float(subset["q_l2"].mean()),
        "q_cos": float(subset["q_cos"].mean()),
        "mean_degree": float(((subset["degree_left"] + subset["degree_right"]) / 2).mean()),
        "overlap_share": float((subset["graphlet_count"] > 1).mean()),
        "roles": {
            f"{int(a)}-{int(b)}": int(value)
            for (a, b), value in subset.groupby(["role_pair_left", "role_pair_right"]).size().sort_values(ascending=False).head(3).items()
        },
    }
    row["matched"] = {
        value: matched_diff(frame, mask, value, strata)
        for value in ("sem_cos", "title_jaccard", "cooc_users", "p_trans_max", "future_hit", "future_lift_max", "lcp", "prefix1")
    }
    semantic_strata = strata + [
        column for column in ("sem_bin", "title_bin")
        if column in frame.columns and column not in exclude
    ]
    row["matched_semantic"] = {
        value: matched_diff(frame, mask, value, semantic_strata)
        for value in ("lcp", "prefix1")
    }
    return row


def summarize_test(frame: pd.DataFrame, mask: np.ndarray, controls: list[str], strata: list[str]) -> dict:
    subset = frame.loc[mask]
    n = int(mask.sum())
    hits = int(subset["hit10"].sum())
    low, high = prop_ci(hits, n)
    row = {
        "n": n,
        "recall10": float(hits / n) if n else float("nan"),
        "recall10_ci": [float(low), float(high)],
        "mrr10": float(subset["mrr10"].mean()),
        "mean_logprob": float(subset["logprob_total"].mean()),
        "mean_logprob_first": float(subset["logprob_sid1"].mean()),
        "mean_logprob_rest": float(subset["logprob_rest"].mean()),
        "median_rank": float(subset["rank_miss_filled"].median()),
        "mean_target_rank": float(subset["rank_miss_filled"].mean()),
        "mean_target_pop": float(subset["target_popularity"].mean()),
        "mean_history": float(subset["history_length"].mean()),
        "mean_sem_cos": float(subset["sem_cos"].mean()),
        "mean_title": float(subset["title_jaccard"].mean()),
        "prefix1": float(subset["prefix1"].mean()),
    }
    row["adjusted"] = {
        "hit10": residual_delta(frame, mask, "hit10", controls),
        "logprob": residual_delta(frame, mask, "logprob_total", controls),
        "logprob_rest": residual_delta(frame, mask, "logprob_rest", controls),
    }
    row["matched"] = {
        "hit10": matched_diff(frame, mask, "hit10", strata),
        "logprob": matched_diff(frame, mask, "logprob_total", strata),
        "logprob_rest": matched_diff(frame, mask, "logprob_rest", strata),
    }
    return row


def format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def format_pp(value: float) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{100 * value:+.2f}pp"


def format_float(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def format_matched(row: dict, key: str = "hit10") -> str:
    matched = row.get("matched", {}).get(key, {})
    value = matched.get("diff", float("nan"))
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100 * value:+.2f}pp (±{100 * matched['se']:.2f}, {matched['cells']}c)"


def main() -> int:
    edges = pd.read_parquet(EDGES)
    items = pd.read_parquet(ITEMS)
    scores = pd.read_parquet(SCORES)
    graphlet_meta = json.loads(GRAPHLETS.read_text())
    stats = json.loads(STATS.read_text())
    score_meta = json.loads(SCORE_META.read_text())

    names = graphlet_meta["order"]
    mask_values = edges["graphlet_mask"].to_numpy(dtype=np.int64)
    for bit, name in enumerate(names):
        edges[f"g_{name}"] = (((mask_values >> bit) & 1) != 0)
    edges["role_pair_left"] = np.minimum(edges["role_left"], edges["role_right"])
    edges["role_pair_right"] = np.maximum(edges["role_left"], edges["role_right"])
    edges["degree_sum"] = edges["degree_left"] + edges["degree_right"]
    edges["pop_sum"] = edges["popularity_left"] + edges["popularity_right"]
    edges["log_degree_sum"] = np.log1p(edges["degree_sum"])
    edges["log_pop_sum"] = np.log1p(edges["pop_sum"])

    edges["sem_bin"] = pd.qcut(edges["sem_cos"].rank(method="first"), 5, labels=False, duplicates="drop")
    edges["title_bin"] = pd.cut(edges["title_jaccard"], [-1e-9, 0.0, 0.05, 0.15, 1.0], labels=False)
    edges["cooc_bin"] = pd.cut(edges["cooc_users"], [0, 1, 3, 9, 10_000], labels=False)
    edges["trans_bin"] = pd.cut(edges["p_trans_max"], [-1e-9, 0.02, 0.06, 0.15, 1.0], labels=False)
    edges["tri_bin"] = pd.cut(edges["triangles"], [-1, 0, 1, 1_000], labels=False)

    edge_strata = []
    for column in ("degree_left", "degree_right", "popularity_left", "popularity_right"):
        edge_strata.append(f"{column}_dec")
        edges[f"{column}_dec"] = decile(edges[column])

    # fixed bin edges for the test rows: learned on TRAIN edges so the two sides share a scale
    test_sem_edges = np.quantile(edges["sem_cos"], np.linspace(0, 1, 6))
    test_sem_edges[0], test_sem_edges[-1] = -np.inf, np.inf
    test_title_edges = np.array([-np.inf, 0.0, 0.05, 0.15, np.inf])
    test_trans_edges = np.array([-np.inf, 0.02, 0.06, 0.15, np.inf])

    result: dict = {
        "inputs": {
            "edge_features": str(EDGES.relative_to(REPO)),
            "edge_features_sha256": hashlib.sha256(EDGES.read_bytes()).hexdigest(),
            "test_scores": str(SCORES.relative_to(REPO)),
            "test_scores_sha256": hashlib.sha256(SCORES.read_bytes()).hexdigest(),
            "graph_stats": stats,
            "score_meta": score_meta,
        },
        "graphlets": graphlet_meta,
        "experiment1": {},
        "experiment2": {},
        "experiment3": {},
    }

    # ---------------- experiment 1: relation profile of each local structure ----------------
    baseline = summarize_edge(edges, np.ones(len(edges), dtype=bool), edge_strata)
    result["experiment1"]["all_edges"] = baseline
    result["experiment1"]["graphlets"] = {}
    for name in names:
        mask = edges[f"g_{name}"].to_numpy(dtype=bool)
        result["experiment1"]["graphlets"][name] = summarize_edge(edges, mask, edge_strata)
    result["experiment1"]["structural_bins"] = {}
    for column, labels in (
        ("tri_bin", {0: "no triangle", 1: "one triangle", 2: "2+ triangles"}),
        ("same_community", {0: "different Louvain community", 1: "same Louvain community"}),
        ("is_bridge", {0: "non-bridge", 1: "bridge"}),
    ):
        for value, label in labels.items():
            mask = (edges[column] == value).to_numpy()
            result["experiment1"]["structural_bins"][label] = summarize_edge(edges, mask, edge_strata)

    # ---------------- experiment 2: relation -> SID prefix ----------------
    def relation_table(column: str, labels: dict | None = None) -> dict:
        table = {}
        for value in sorted(edges[column].dropna().unique().tolist()):
            mask = (edges[column] == value).to_numpy()
            label = labels.get(int(value), str(value)) if labels else str(value)
            row = summarize_edge(edges, mask, edge_strata, exclude=(column,))
            table[label] = row
        return table

    result["experiment2"]["semantic_quintiles"] = relation_table("sem_bin")
    result["experiment2"]["title_bins"] = relation_table("title_bin")
    result["experiment2"]["cooccurrence_bins"] = relation_table("cooc_bin")
    result["experiment2"]["transition_bins"] = relation_table("trans_bin")
    result["experiment2"]["triangle_bins"] = relation_table("tri_bin")

    # role pairs, ordered by frequency
    role_counts = edges.groupby(["role_pair_left", "role_pair_right"]).size().sort_values(ascending=False)
    result["experiment2"]["role_pairs"] = {}
    for (a, b), count in role_counts.head(12).items():
        mask = ((edges["role_pair_left"] == a) & (edges["role_pair_right"] == b)).to_numpy()
        result["experiment2"]["role_pairs"][f"role{int(a)}-role{int(b)}"] = summarize_edge(edges, mask, edge_strata)

    # graphlet x semantic-high: does a dense structure only share prefixes when it is also semantic?
    semantic_high = (edges["sem_bin"] >= 4).to_numpy()
    result["experiment2"]["graphlet_by_semantics"] = {}
    for name in names:
        inside = edges[f"g_{name}"].to_numpy(dtype=bool)
        result["experiment2"]["graphlet_by_semantics"][name] = {
            "high_semantic": summarize_edge(edges, inside & semantic_high, edge_strata),
            "not_high_semantic": summarize_edge(edges, inside & ~semantic_high, edge_strata),
        }

    # logistic/OLS view of prefix@1 on all relation columns, for a compact effect summary
    from sklearn.linear_model import LogisticRegression

    feature_columns = [
        "sem_cos", "title_jaccard", "log_degree_sum", "log_pop_sum", "same_community",
        "triangles", "cooc_users", "p_trans_max",
    ]
    model_frame = edges[feature_columns + ["prefix1"]].dropna()
    x = StandardScaler().fit_transform(model_frame[feature_columns].to_numpy(dtype=np.float64))
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(x, model_frame["prefix1"].to_numpy())
    result["experiment2"]["logistic_prefix1"] = {
        "n": int(len(model_frame)),
        "features": feature_columns,
        "coefficients_standardised": {name: float(value) for name, value in zip(feature_columns, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
    }

    # ---------------- experiment 3: relation -> T5 prediction ----------------
    test = pd.read_parquet(TEST, columns=["user", "seen_history", "target"]).reset_index(drop=True)
    test["global_row"] = np.arange(len(test))
    test["last_item"] = [int(history[-1]) if len(history) else -1 for history in test["seen_history"].to_numpy()]
    test["history_length"] = [len(history) for history in test["seen_history"].to_numpy()]
    score_columns = scores.drop(columns=["last_item", "history_length"], errors="ignore")
    test = test.merge(score_columns, on="global_row", how="left", validate="one_to_one")
    if test["target_item"].isna().any():
        raise SystemExit("test_scores does not cover every test row")
    test["target_item"] = test["target_item"].astype(np.int64)
    test["last_item"] = test["last_item"].astype(np.int64)

    pair_left = np.minimum(test["last_item"].to_numpy(), test["target_item"].to_numpy())
    pair_right = np.maximum(test["last_item"].to_numpy(), test["target_item"].to_numpy())
    test["pair_left"] = pair_left
    test["pair_right"] = pair_right
    merged = test.merge(
        edges, left_on=["pair_left", "pair_right"], right_on=["left", "right"], how="left", validate="many_to_one"
    )
    merged["seen_edge"] = merged["left"].notna()
    direction_forward = merged["last_item"].to_numpy() == merged["left"].to_numpy()
    merged["trans_dir"] = np.where(direction_forward, merged["trans_ab"], merged["trans_ba"])
    merged["p_trans_dir"] = np.where(direction_forward, merged["p_trans_ab"], merged["p_trans_ba"])
    for column in ("trans_dir", "p_trans_dir", "cooc_users", "triangles", "same_community"):
        merged.loc[~merged["seen_edge"], column] = 0
    for name in names:
        merged[f"g_{name}"] = merged[f"g_{name}"].fillna(False).infer_objects(copy=False).astype(bool)

    titles = pd.read_parquet(ITEM_TABLE, columns=["item_id", "title"]).sort_values("item_id")["title"].tolist()
    test_sem = semantic_cosine(
        np.load(EMB), merged["last_item"].to_numpy(), merged["target_item"].to_numpy()
    )
    test_title = title_jaccard(titles, merged["last_item"].to_numpy(), merged["target_item"].to_numpy())
    merged["test_sem_cos"] = test_sem
    merged["test_title_jaccard"] = test_title

    # semantic/title/prefix for every test row, independent of whether the edge is in TRAIN
    item_lookup = items.set_index("item_id")
    last_rows = item_lookup.loc[merged["last_item"].to_numpy()]
    target_rows = item_lookup.loc[merged["target_item"].to_numpy()]
    merged["last_popularity"] = last_rows["popularity"].to_numpy()
    merged["target_popularity"] = target_rows["popularity"].to_numpy()
    merged["last_degree"] = last_rows["degree"].to_numpy()
    merged["target_degree"] = target_rows["degree"].to_numpy()
    merged["last_role"] = last_rows["role"].to_numpy()
    merged["target_role"] = target_rows["role"].to_numpy()
    merged["same_community_test"] = (last_rows["community"].to_numpy() == target_rows["community"].to_numpy()).astype(int)
    last_sid = last_rows[["sid1", "sid2", "sid3"]].to_numpy()
    target_sid = target_rows[["sid1", "sid2", "sid3"]].to_numpy()
    equal = last_sid == target_sid
    merged["lcp"] = np.where(equal[:, 0], np.where(equal[:, 1], np.where(equal[:, 2], 3, 2), 1), 0)
    merged["prefix1"] = equal[:, 0].astype(int)
    merged["prefix2"] = (equal[:, 0] & equal[:, 1]).astype(int)
    merged["prefix3"] = (equal[:, 0] & equal[:, 1] & equal[:, 2]).astype(int)
    merged["role_pair_left"] = np.minimum(merged["last_role"], merged["target_role"])
    merged["role_pair_right"] = np.maximum(merged["last_role"], merged["target_role"])

    merged["rank_miss_filled"] = merged["target_rank_top10"].fillna(11.0)
    merged["hit10"] = (merged["target_rank_top10"] < 10).astype(int)
    merged["mrr10"] = np.where(merged["target_rank_top10"] < 10, 1.0 / (merged["target_rank_top10"] + 1.0), 0.0)
    merged["logprob_rest"] = merged["logprob_sid2"] + merged["logprob_sid3"] + merged["logprob_eos"]
    merged["log_target_pop"] = np.log1p(merged["target_popularity"])
    merged["log_last_pop"] = np.log1p(merged["last_popularity"])
    merged["log_target_degree"] = np.log1p(merged["target_degree"])
    merged["log_last_degree"] = np.log1p(merged["last_degree"])
    merged["sem_cos"] = merged["test_sem_cos"]
    merged["title_jaccard"] = merged["test_title_jaccard"]

    merged["sem_bin"] = bin_by_edges(merged["sem_cos"], test_sem_edges)
    merged["title_bin"] = pd.cut(merged["title_jaccard"], test_title_edges, labels=False)
    merged["trans_bin"] = bin_by_edges(merged["p_trans_dir"], test_trans_edges)
    merged["cooc_bin"] = pd.cut(merged["cooc_users"], [-1, 1, 3, 9, 10_000], labels=False)

    test_strata = []
    for column in ("target_popularity", "last_popularity", "history_length"):
        merged[f"{column}_dec"] = decile(merged[column])
        test_strata.append(f"{column}_dec")
    merged["sem_cos_dec"] = decile(merged["sem_cos"])
    prefix_strata = test_strata + ["sem_cos_dec"]

    result["experiment3"]["overall"] = summarize_test(merged, np.ones(len(merged), dtype=bool), TEST_CONTROLS, test_strata)
    groups: dict[str, dict] = {}
    for column, labels in (
        ("seen_edge", {False: "unseen TRAIN edge", True: "seen TRAIN edge"}),
        ("prefix1", {0: "different L1", 1: "same L1"}),
        ("prefix2", {0: "LCP < 2", 1: "same L1+L2"}),
        ("prefix3", {0: "LCP < 3", 1: "same L1+L2+L3"}),
        ("lcp", {0: "LCP 0", 1: "LCP 1", 2: "LCP 2", 3: "LCP 3"}),
        ("sem_bin", {0: "semantic Q1", 1: "semantic Q2", 2: "semantic Q3", 3: "semantic Q4", 4: "semantic Q5"}),
        ("title_bin", {0: "title Jaccard 0", 1: "0 < J <= .05", 2: ".05 < J <= .15", 3: "J > .15"}),
        ("cooc_bin", {0: "co-occur 1 user", 1: "2-3 users", 2: "4-9 users", 3: "10+ users"}),
        ("trans_bin", {0: "p_trans <= .02", 1: ".02-.06", 2: ".06-.15", 3: ">.15"}),
        ("tri_bin", {0: "no triangle", 1: "one triangle", 2: "2+ triangles"}),
        ("same_community_test", {0: "different community", 1: "same community"}),
    ):
        table = {}
        for value, label in labels.items():
            if column == "seen_edge":
                mask = merged[column].to_numpy(dtype=bool) == bool(value)
            else:
                mask = (merged[column] == value).to_numpy()
            if mask.sum() < 30:
                continue
            table_controls = TEST_CONTROLS
            if column in ("sem_bin",):
                table_strata = test_strata
            elif column == "title_bin":
                table_strata = test_strata + ["sem_cos_dec"]
            else:
                # prefix, co-occurrence, transition, triangle, community, seen-edge:
                # match on popularity/history and on semantic/title similarity
                table_strata = prefix_strata
            table[label] = summarize_test(merged, mask, table_controls, table_strata)
        groups[column] = table
    result["experiment3"]["groups"] = groups

    # graphlet groups are meaningful only on edges that exist in TRAIN
    seen = merged["seen_edge"].to_numpy(dtype=bool)
    result["experiment3"]["graphlets"] = {}
    for name in names:
        mask = seen & merged[f"g_{name}"].to_numpy(dtype=bool)
        if mask.sum() >= 30:
            result["experiment3"]["graphlets"][name] = summarize_test(merged, mask, TEST_CONTROLS, test_strata)
    result["experiment3"]["seen_edges_baseline"] = summarize_test(merged, seen, TEST_CONTROLS, test_strata)

    # the graphlet question is only meaningful inside TRAIN edges: compare each structure with the
    # OTHER seen edges, not with the unseen majority
    seen_frame = merged.loc[seen].reset_index(drop=True)
    result["experiment3"]["graphlets_vs_seen"] = {}
    for name in names:
        mask = seen_frame[f"g_{name}"].to_numpy(dtype=bool)
        if mask.sum() >= 30:
            result["experiment3"]["graphlets_vs_seen"][name] = summarize_test(
                seen_frame, mask, TEST_CONTROLS, prefix_strata
            )
    result["experiment3"]["seen_frame_baseline"] = summarize_test(
        seen_frame, np.ones(len(seen_frame), dtype=bool), TEST_CONTROLS, prefix_strata
    )

    # role pairs, top 10 by test frequency
    role_counts_test = merged.groupby(["role_pair_left", "role_pair_right"]).size().sort_values(ascending=False)
    result["experiment3"]["role_pairs"] = {}
    for (a, b), count in role_counts_test.head(10).items():
        mask = ((merged["role_pair_left"] == a) & (merged["role_pair_right"] == b)).to_numpy()
        result["experiment3"]["role_pairs"][f"role{int(a)}-role{int(b)}"] = summarize_test(
            merged, mask, TEST_CONTROLS, prefix_strata
        )
    seen_role_counts = seen_frame.groupby(["role_pair_left", "role_pair_right"]).size().sort_values(ascending=False)
    result["experiment3"]["role_pairs_seen"] = {}
    for (a, b), count in seen_role_counts.head(10).items():
        mask = ((seen_frame["role_pair_left"] == a) & (seen_frame["role_pair_right"] == b)).to_numpy()
        result["experiment3"]["role_pairs_seen"][f"role{int(a)}-role{int(b)}"] = summarize_test(
            seen_frame, mask, TEST_CONTROLS, prefix_strata
        )

    # the key 2x2: cross-prefix but still predictable
    cross = {}
    for name, mask in (
        ("all", np.ones(len(merged), dtype=bool)),
        ("same_L1", merged["prefix1"].to_numpy(dtype=bool)),
        ("different_L1", ~merged["prefix1"].to_numpy(dtype=bool)),
    ):
        cross[name] = {}
        for relation_name, relation_mask in (
            ("semantic_Q5", (merged["sem_bin"] == 4).to_numpy()),
            ("cooc_4plus", (merged["cooc_users"] >= 4).to_numpy()),
            ("trans_gt_.15", (merged["p_trans_dir"] > 0.15).to_numpy()),
            ("seen_edge", merged["seen_edge"].to_numpy(dtype=bool)),
        ):
            selected = mask & relation_mask
            if selected.sum() >= 30:
                cross[name][relation_name] = summarize_test(merged, selected, TEST_CONTROLS, test_strata)
    result["experiment3"]["cross_prefix"] = cross

    prefix_gap = {}
    for relation_name, relation_mask in (
        ("all rows", np.ones(len(merged), dtype=bool)),
        ("semantic_Q5", (merged["sem_bin"] == 4).to_numpy()),
        ("cooc_4plus", (merged["cooc_users"] >= 4).to_numpy()),
        ("trans_gt_.15", (merged["p_trans_dir"] > 0.15).to_numpy()),
        ("seen_edge", merged["seen_edge"].to_numpy(dtype=bool)),
    ):
        selected = relation_mask
        if selected.sum() < 60:
            continue
        subframe = merged.loc[selected].reset_index(drop=True)
        same_mask = subframe["prefix1"].to_numpy(dtype=bool)
        if same_mask.sum() < 30 or (~same_mask).sum() < 30:
            continue
        prefix_gap[relation_name] = {
            "same_L1": summarize_test(subframe, same_mask, TEST_CONTROLS, test_strata),
            "different_L1": summarize_test(subframe, ~same_mask, TEST_CONTROLS, test_strata),
            "gap": summarize_test(subframe, same_mask, TEST_CONTROLS, prefix_strata),
        }
    result["experiment3"]["prefix_gap_by_relation"] = prefix_gap

    # save the derived test table (without the wide edge-table columns) for reproducibility
    keep = [
        "global_row", "last_item", "target_item", "history_length", "target_rank_top10",
        "logprob_total", "logprob_rest", "sem_cos", "title_jaccard", "lcp", "prefix1", "prefix2", "prefix3",
        "seen_edge", "cooc_users", "trans_dir", "p_trans_dir", "triangles", "same_community_test",
        "target_popularity", "last_popularity", "target_degree", "last_degree",
        "role_pair_left", "role_pair_right",
    ]
    merged[keep].to_parquet(HERE / "test_relations.parquet", index=False)

    OUT.write_text(json.dumps(result, indent=2) + "\n")

    # ---------------- Markdown tables ----------------
    lines: list[str] = []
    lines.append("# Graphlet Role x relation x SID prefix tables\n")
    lines.append(
        f"TRAIN edges {len(edges):,}; TEST rows {len(merged):,}; "
        f"T5 checkpoint `{score_meta['checkpoint']}` (recall@10 {score_meta['recomputed_recall@10']:.5f}).\n"
    )
    lines.append("## Experiment 1 - relation profile by local structure\n")
    lines.append(
        "| structure | n edges | % of edges | sem cos | title Jaccard | co-occur users | max p(trans) | "
        "same community | future hit | future lift | mean LCP | prefix@1 | prefix@2 | prefix@3 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = [("all edges", result["experiment1"]["all_edges"])] + list(result["experiment1"]["graphlets"].items())
    for label, row in rows:
        lines.append(
            f"| {label} | {row['n']:,} | {format_pct(row['share'])} | {row['sem_cos']:.3f} | "
            f"{row['title_jaccard']:.3f} | {row['cooc_users']:.2f} | {row['p_trans_max']:.3f} | "
            f"{format_pct(row['same_community'])} | {format_pct(row['future_hit'])} | "
            f"{row['future_lift']:.3f} | {row['lcp']:.3f} | {format_pct(row['prefix1'])} | "
            f"{format_pct(row['prefix2'])} | {format_pct(row['prefix3'])} |\n"
        )
    lines.append("\nMatched differences (structure minus the rest, stratified by degree and popularity deciles):\n\n")
    lines.append("| structure | d sem cos | d title | d co-occur | d p(trans) | d future hit | d LCP | d prefix@1 | d prefix@1 (semantic-controlled) |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, row in rows[1:]:
        matched = row["matched"]
        controlled = row["matched_semantic"]["prefix1"]
        lines.append(
            f"| {label} | {matched['sem_cos']['diff']:+.3f} | {matched['title_jaccard']['diff']:+.3f} | "
            f"{matched['cooc_users']['diff']:+.2f} | {matched['p_trans_max']['diff']:+.3f} | "
            f"{100 * matched['future_hit']['diff']:+.2f}pp | {matched['lcp']['diff']:+.3f} | "
            f"{100 * matched['prefix1']['diff']:+.2f}pp | {100 * controlled['diff']:+.2f}pp |\n"
        )

    lines.append("\n### Exact structural bins\n\n")
    lines.append("| bin | n | future hit | co-occur users | max p(trans) | prefix@1 | prefix@2 | prefix@3 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, row in result["experiment1"]["structural_bins"].items():
        lines.append(
            f"| {label} | {row['n']:,} | {format_pct(row['future_hit'])} | {row['cooc_users']:.2f} | "
            f"{row['p_trans_max']:.3f} | {format_pct(row['prefix1'])} | {format_pct(row['prefix2'])} | "
            f"{format_pct(row['prefix3'])} |\n"
        )
    lines.append("\n### Role pairs (undirected, by endpoint ORCA role)\n\n")
    lines.append("| role pair | n | sem cos | co-occur users | max p(trans) | future hit | prefix@1 | prefix@2 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, row in result["experiment2"]["role_pairs"].items():
        lines.append(
            f"| {label} | {row['n']:,} | {row['sem_cos']:.3f} | {row['cooc_users']:.2f} | "
            f"{row['p_trans_max']:.3f} | {format_pct(row['future_hit'])} | {format_pct(row['prefix1'])} | "
            f"{format_pct(row['prefix2'])} |\n"
        )

    lines.append("\n## Experiment 2 - relation -> SID prefix\n")
    for title, table in (
        ("Semantic quintiles", result["experiment2"]["semantic_quintiles"]),
        ("Title-token Jaccard", result["experiment2"]["title_bins"]),
        ("Co-occurrence", result["experiment2"]["cooccurrence_bins"]),
        ("Directed transition strength", result["experiment2"]["transition_bins"]),
        ("Triangles per edge", result["experiment2"]["triangle_bins"]),
    ):
        lines.append(f"\n### {title}\n\n")
        show_control = title != "Semantic quintiles"
        if show_control:
            lines.append("| group | n | mean LCP | prefix@1 | prefix@2 | prefix@3 | quantised L2 | q cos | matched d prefix@1 | d prefix@1 (sem-controlled) |\n")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        else:
            lines.append("| group | n | mean LCP | prefix@1 | prefix@2 | prefix@3 | quantised L2 | q cos | matched d prefix@1 |\n")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for label, row in table.items():
            matched = row["matched"]["prefix1"]
            if show_control:
                controlled = row["matched_semantic"]["prefix1"]
                lines.append(
                    f"| {label} | {row['n']:,} | {row['lcp']:.3f} | {format_pct(row['prefix1'])} | "
                    f"{format_pct(row['prefix2'])} | {format_pct(row['prefix3'])} | {row['q_l2']:.3f} | "
                    f"{row['q_cos']:.3f} | {100 * matched['diff']:+.2f}pp | "
                    f"{100 * controlled['diff']:+.2f}pp |\n"
                )
            else:
                lines.append(
                    f"| {label} | {row['n']:,} | {row['lcp']:.3f} | {format_pct(row['prefix1'])} | "
                    f"{format_pct(row['prefix2'])} | {format_pct(row['prefix3'])} | {row['q_l2']:.3f} | "
                    f"{row['q_cos']:.3f} | {100 * matched['diff']:+.2f}pp |\n"
                )
    lines.append("\n### Graphlet x semantic-high\n\n")
    lines.append("| structure | high-sem n | high-sem prefix@1 | other n | other prefix@1 |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for name, row in result["experiment2"]["graphlet_by_semantics"].items():
        lines.append(
            f"| {name} | {row['high_semantic']['n']:,} | {format_pct(row['high_semantic']['prefix1'])} | "
            f"{row['not_high_semantic']['n']:,} | {format_pct(row['not_high_semantic']['prefix1'])} |\n"
        )

    lines.append("\n## Experiment 3 - T5 prediction by relation\n")
    lines.append(
        "\n`matched d recall` is the within-cell difference in recall@10: popularity and history-length "
        "deciles for every group, plus semantic-similarity and title bins for prefix, co-occurrence, "
        "transition, triangle, community, seen-edge and graphlet groups (never for the variable used as "
        "the grouping). Parentheses give the standard error and the number of cells with >=5 rows on "
        "both sides; `n/a` means no cell had enough support. `adj. d recall` is the OLS-residual "
        "difference on the same controls and is more stable, but it is linear.\n"
    )
    lines.append("\n### Overall and SID prefix\n\n")
    lines.append("| group | n | recall@10 | MRR@10 | mean log p | mean rest log p | median rank (11=miss) | adj. d recall | adj. d log p | matched d recall | matched d rest log p |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    header_rows = [("all test rows", result["experiment3"]["overall"])]
    for column in ("prefix1", "prefix2", "prefix3", "lcp"):
        header_rows.extend((f"{column}: {label}", row) for label, row in groups[column].items())
    for label, row in header_rows:
        adjusted = row["adjusted"]
        matched = row["matched"]
        lines.append(
            f"| {label} | {row['n']:,} | {format_pct(row['recall10'])} | {row['mrr10']:.4f} | "
            f"{row['mean_logprob']:.3f} | {row['mean_logprob_rest']:.3f} | {row['median_rank']:.0f} | "
            f"{format_pp(adjusted['hit10']['diff'])} | {format_float(adjusted['logprob']['diff'])} | "
            f"{format_matched(row)} | {format_float(matched['logprob_rest']['diff'])} |\n"
        )
    lines.append("\n### Relation classes\n\n")
    lines.append("| group | n | recall@10 | MRR@10 | mean log p | mean target pop | adj. d recall | matched d recall |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for column in ("sem_bin", "title_bin", "cooc_bin", "trans_bin", "tri_bin", "same_community_test", "seen_edge"):
        for label, row in groups[column].items():
            adjusted = row["adjusted"]
            matched = row["matched"]
            lines.append(
                f"| {column}: {label} | {row['n']:,} | {format_pct(row['recall10'])} | {row['mrr10']:.4f} | "
                f"{row['mean_logprob']:.3f} | {row['mean_target_pop']:.1f} | "
                f"{format_pp(adjusted['hit10']['diff'])} | {format_matched(row)} |\n"
            )
    lines.append("\n### Graphlets (seen TRAIN edges only, compared with the other seen edges)\n\n")
    lines.append("| structure | n | recall@10 | MRR@10 | mean log p | adj. d recall | matched d recall |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    baseline_seen = result["experiment3"]["seen_frame_baseline"]
    lines.append(
        f"| seen-edge baseline | {baseline_seen['n']:,} | {format_pct(baseline_seen['recall10'])} | "
        f"{baseline_seen['mrr10']:.4f} | {baseline_seen['mean_logprob']:.3f} | "
        f"{format_pp(baseline_seen['adjusted']['hit10']['diff'])} | "
        f"{format_matched(baseline_seen)} |\n"
    )
    for name, row in result["experiment3"]["graphlets_vs_seen"].items():
        lines.append(
            f"| {name} | {row['n']:,} | {format_pct(row['recall10'])} | {row['mrr10']:.4f} | "
            f"{row['mean_logprob']:.3f} | {format_pp(row['adjusted']['hit10']['diff'])} | "
            f"{format_matched(row)} |\n"
        )
    lines.append("\n### Role pairs on seen TRAIN edges\n\n")
    lines.append("| role pair | n | recall@10 | MRR@10 | mean log p | adj. d recall | matched d recall |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for label, row in result["experiment3"]["role_pairs_seen"].items():
        lines.append(
            f"| {label} | {row['n']:,} | {format_pct(row['recall10'])} | {row['mrr10']:.4f} | "
            f"{row['mean_logprob']:.3f} | {format_pp(row['adjusted']['hit10']['diff'])} | "
            f"{format_matched(row)} |\n"
        )
    lines.append("\n### Cross-prefix but still predictable\n\n")
    lines.append("| prefix | relation | n | recall@10 | MRR@10 | mean log p |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for prefix_label, table in result["experiment3"]["cross_prefix"].items():
        for relation_label, row in table.items():
            lines.append(
                f"| {prefix_label} | {relation_label} | {row['n']:,} | {format_pct(row['recall10'])} | "
                f"{row['mrr10']:.4f} | {row['mean_logprob']:.3f} |\n"
            )
    lines.append("\n### Is the L1 prefix worth anything inside a relation class?\n\n")
    lines.append("| relation | n same | recall same | n diff | recall diff | raw gap | matched gap | adj. gap |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for relation_name, entry in result["experiment3"]["prefix_gap_by_relation"].items():
        same = entry["same_L1"]
        different = entry["different_L1"]
        gap = entry["gap"]
        lines.append(
            f"| {relation_name} | {same['n']:,} | {format_pct(same['recall10'])} | "
            f"{different['n']:,} | {format_pct(different['recall10'])} | "
            f"{format_pp(same['recall10'] - different['recall10'])} | "
            f"{format_matched(gap)} | {format_pp(gap['adjusted']['hit10']['diff'])} |\n"
        )
    MARKDOWN.write_text("".join(lines) + "\n")
    print(f"wrote {OUT} and {MARKDOWN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
