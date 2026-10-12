from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr


def build_successors(pairs):
    out = defaultdict(set)
    for anchor, positive in zip(pairs["anchors"], pairs["positives"]):
        out[int(anchor)].add(int(positive))
    return dict(out)


def topk_composition(ranked, successors, negatives, k):
    chosen = np.asarray(ranked)[:k]
    succ_count = sum(int(item) in successors for item in chosen)
    neg_count = sum(int(item) in negatives for item in chosen)
    rest = len(chosen) - succ_count - neg_count
    denom = max(len(chosen), 1)
    return succ_count / denom, neg_count / denom, rest / denom


def _corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return None
    return float(pearsonr(x[keep], y[keep]).statistic)


def _adjacency_arrays(successors, n_items):
    out = np.zeros(n_items, dtype=np.int64)
    inc = np.zeros(n_items, dtype=np.int64)
    for a, targets in successors.items():
        if 0 <= a < n_items:
            out[a] = len(targets)
        for target in targets:
            if 0 <= target < n_items:
                inc[target] += 1
    return out, inc


def _mixing(neighbors, successors, frequencies, seed, anchor_filter=None):
    rng = np.random.default_rng(seed)
    allowed = set(successors) if anchor_filter is None else set(anchor_filter)
    predecessors = defaultdict(set)
    for anchor, targets in successors.items():
        if anchor in allowed:
            for target in targets:
                predecessors[target].add(anchor)
    pairs = set()
    for members in predecessors.values():
        members = sorted(members)
        if len(members) > 1:
            for i, a in enumerate(members):
                for b in members[i+1:min(len(members), i+11)]:
                    pairs.add((a, b))
                    if len(pairs) >= 20000:
                        break
                if len(pairs) >= 20000:
                    break
        if len(pairs) >= 20000:
            break
    if not pairs:
        return {str(k): {"related_anchor_jaccard": None, "popularity_matched_null_jaccard": None,
                         "n_related_pairs": 0, "n_null_pairs": 0} for k in (5, 10, 20, 50)}
    anchors = sorted(set(a for pair in pairs for a in pair) & allowed)
    freq_decile = np.zeros(len(frequencies), dtype=np.int64)
    order = np.argsort(frequencies, kind="stable")
    freq_decile[order] = np.minimum(np.arange(len(order)) * 10 // max(len(order), 1), 9)
    out = {}
    for k in (5, 10, 20, 50):
        related = []
        for a, b in pairs:
            related.append(len(set(neighbors[a, :k]) & set(neighbors[b, :k])) /
                           max(len(set(neighbors[a, :k]) | set(neighbors[b, :k])), 1))
        null = []
        for _ in range(len(related)):
            a = int(rng.choice(anchors))
            pool = np.flatnonzero(freq_decile == freq_decile[a])
            pool = pool[pool != a]
            if not len(pool):
                continue
            b = int(rng.choice(pool))
            null.append(len(set(neighbors[a, :k]) & set(neighbors[b, :k])) /
                        max(len(set(neighbors[a, :k]) | set(neighbors[b, :k])), 1))
        out[str(k)] = {"related_anchor_jaccard": float(np.mean(related)) if related else None,
                       "popularity_matched_null_jaccard": float(np.mean(null)) if null else None,
                       "n_related_pairs": len(related), "n_null_pairs": len(null)}
    return out


def analyze_neighborhoods(residuals, pairs, train_frequency=None, seed=42, train_graph=None):
    if train_frequency is None:
        train_frequency = np.zeros(next(iter(residuals.values())).shape[0], dtype=np.int64)
    n_items = len(train_frequency)
    successors = build_successors(pairs)
    degree_graph = train_graph if train_graph is not None else successors
    negatives_by_anchor = defaultdict(set)
    for a, negatives in zip(pairs["anchors"], pairs["negatives"]):
        negatives_by_anchor[int(a)].update(map(int, negatives))
    out_degree, in_degree = _adjacency_arrays(degree_graph, n_items)
    report, csv_rows = [], []
    for model, tensor in residuals.items():
        for level in range(3):
            vectors = np.asarray(tensor[:, level, :], dtype=np.float64)
            tree = cKDTree(vectors)
            distances, neighbor_ids = tree.query(vectors, k=min(51, n_items), workers=-1)
            if neighbor_ids.ndim == 1:
                neighbor_ids, distances = neighbor_ids[:, None], distances[:, None]
            neighbors = np.empty((n_items, 50), dtype=np.int64)
            neighbor_dists = np.empty((n_items, 50), dtype=np.float64)
            for item in range(n_items):
                keep = neighbor_ids[item] != item
                ids, ds = neighbor_ids[item][keep], distances[item][keep]
                neighbors[item, :min(50, len(ids))] = ids[:50]
                neighbor_dists[item, :min(50, len(ds))] = ds[:50]
            local_density = neighbor_dists.mean(axis=1)
            anchors = sorted(a for a in successors if 0 <= a < n_items and successors[a])
            per_anchor = []
            composition = {k: [] for k in (5, 10, 20, 50)}
            for anchor in anchors:
                succ = {x for x in successors[anchor] if 0 <= x < n_items and x != anchor}
                negs = {x for x in negatives_by_anchor[anchor] if 0 <= x < n_items and x not in succ and x != anchor}
                if not succ:
                    continue
                ranked = neighbors[anchor]
                for k in composition:
                    composition[k].append(topk_composition(ranked, succ, negs, k))
                succ_ds = np.linalg.norm(vectors[list(succ)] - vectors[anchor], axis=1)
                neg_ds = np.linalg.norm(vectors[list(negs)] - vectors[anchor], axis=1) if negs else np.array([])
                nearest_succ = float(succ_ds.min())
                nearest_neg = float(neg_ds.min()) if len(neg_ds) else np.nan
                per_anchor.append({"anchor": anchor, "multiplicity": len(succ), "nearest_successor": nearest_succ,
                                   "nearest_negative": nearest_neg, "margin": nearest_succ - nearest_neg,
                                   "intrusion": float(np.mean(neg_ds < nearest_succ)) if len(neg_ds) else np.nan,
                                   "inside_kth": {k: float(np.mean(neg_ds <= np.partition(succ_ds, min(k, len(succ_ds))-1)[min(k, len(succ_ds))-1])) if len(neg_ds) else np.nan
                                                  for k in (5,10,20,50)}})
            strata = {}
            for stratum, selected in (("multiplicity_1", [r for r in per_anchor if r["multiplicity"] == 1]),
                                      ("multiplicity_2plus", [r for r in per_anchor if r["multiplicity"] >= 2])):
                strata[stratum] = {"n_anchors": len(selected),
                    "constraint_satisfaction_rate": float(np.mean([r["nearest_successor"] < r["nearest_negative"] for r in selected if np.isfinite(r["nearest_negative"])])) if any(np.isfinite(r["nearest_negative"]) for r in selected) else None,
                    "mean_margin": float(np.nanmean([r["margin"] for r in selected])) if selected else None,
                    "hard_negative_intrusion_rate": float(np.nanmean([r["intrusion"] for r in selected])) if selected else None,
                    "inside_kth_successor_distance": {str(k): float(np.nanmean([r["inside_kth"][k] for r in selected])) if selected else None for k in (5,10,20,50)},
                    "topk_composition": {}}
                for k in composition:
                    vals = [topk_composition(neighbors[r["anchor"]], successors[r["anchor"]], negatives_by_anchor[r["anchor"]], k) for r in selected]
                    strata[stratum]["topk_composition"][str(k)] = {name: float(np.mean([v[i] for v in vals])) if vals else None for i,name in enumerate(("successor_share","matched_negative_share","neither_share"))}
                strata[stratum]["neighborhood_overlap"] = _mixing(
                    neighbors, successors, train_frequency, seed + level,
                    {r["anchor"] for r in selected})
            for k, values in composition.items():
                for i, label in enumerate(("successor_share", "matched_negative_share", "neither_share")):
                    csv_rows.append({"model": model, "level": f"L{level+1}", "metric": label, "K": k,
                                     "value": float(np.mean([x[i] for x in values])) if values else None,
                                     "n_anchors": len(values), "stratum": "all"})
            # Confusability: how often each item is a matched negative inside the nearest-successor radius.
            confusion = np.zeros(n_items, dtype=np.int64)
            for r in per_anchor:
                a = r["anchor"]
                if not np.isfinite(r["nearest_successor"]):
                    continue
                confusable = [x for x in negatives_by_anchor[a] if 0 <= x < n_items and x != a and
                              np.linalg.norm(vectors[a] - vectors[x]) < r["nearest_successor"]]
                confusion[confusable] += 1
            density = local_density
            confusability = {"train_frequency": _corr(confusion, train_frequency), "out_degree": _corr(confusion, out_degree),
                             "in_degree": _corr(confusion, in_degree), "local_density": _corr(confusion, density)}
            diversity = np.bincount([r["multiplicity"] for r in per_anchor]) if per_anchor else np.array([], dtype=int)
            diversity_report = {str(i): int(v) for i, v in enumerate(diversity) if i > 0 and v}
            report.append({"model": model, "level": f"L{level+1}", "n_anchors": len(per_anchor),
                           "successor_multiplicity_distribution": diversity_report, "stratified": strata,
                           "topk_composition": {str(k): {name: float(np.mean([v[i] for v in values])) if values else None
                                                         for i,name in enumerate(("successor_share","matched_negative_share","neither_share"))}
                                                for k,values in composition.items()},
                           "hard_negative_intrusion_rate": float(np.nanmean([r["intrusion"] for r in per_anchor])) if per_anchor else None,
                           "inside_kth_successor_distance": {str(k): float(np.nanmean([r["inside_kth"][k] for r in per_anchor])) if per_anchor else None for k in (5,10,20,50)},
                           "mean_margin": float(np.nanmean([r["margin"] for r in per_anchor])) if per_anchor else None,
                           "constraint_satisfaction_rate": float(np.mean([r["nearest_successor"] < r["nearest_negative"] for r in per_anchor if np.isfinite(r["nearest_negative"])])) if any(np.isfinite(r["nearest_negative"]) for r in per_anchor) else None,
                           "neighborhood_overlap": _mixing(neighbors, successors, train_frequency, seed + level),
                           "confusability_correlations": confusability,
                           "most_confusable_items": sorted([{"item_id": int(i), "confusability": int(confusion[i]), "train_frequency": int(train_frequency[i]), "out_degree": int(out_degree[i]), "in_degree": int(in_degree[i]), "local_density": float(density[i])} for i in np.argsort(confusion)[-20:][::-1]], key=lambda x: (-x["confusability"], x["item_id"]))})
    return report, csv_rows
