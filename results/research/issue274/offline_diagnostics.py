#!/usr/bin/env python3
"""Issue #274, step 4: the offline diagnostics the issue requires before any training.

Three questions from issue section 1:
  1. is the real curvature ordering consistent with the outer-pair relation inside one centre?
  2. under matched topology/direction conditions, does the curvature still carry selection
     information beyond the closed/open label (the issue asks us to record honestly if it does not)?
  3. what residual correlation do shared items create between the training and the hold-out pairs?

Only frozen assets and the frozen pair index are read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue274"
SAMPLE_CENTRES = 1500


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    index = np.load(OUT / "pairs_index.npz")
    train = ~index["is_holdout"]
    pos_c = index["pos_curvature"]
    neg_c = index["neg_curvature"]

    # 1. the curvature ordering against the relation type
    diff = pos_c[train] - neg_c[train]
    q = np.quantile(diff, [0.05, 0.25, 0.5, 0.75, 0.95])

    # 3. the residual correlation from shared items
    pairs = np.stack([index["pos_b"][train], index["pos_c"][train],
                      index["neg_d"][train], index["neg_e"][train]], axis=1)
    shared = np.any(pairs[:, :1] == pairs[:, 2:3], axis=1) | \
        np.any(pairs[:, :1] == pairs[:, 3:4], axis=1) | \
        np.any(pairs[:, 1:2] == pairs[:, 2:3], axis=1) | \
        np.any(pairs[:, 1:2] == pairs[:, 3:4], axis=1)
    item_counts = np.bincount(pairs.reshape(-1), minlength=24587)
    participation = item_counts[item_counts > 0]
    centres = index["centre"][train]
    centre_counts = np.bincount(centres, minlength=24587)

    # 2. the extra-information check on the matched subset: inside one centre and one structure
    #    class, does the curvature track an independent tightness signal of the outer pair
    #    (the number of common neighbours of the two outer items in the visible undirected graph)?
    undirected = np.load(REPO / "assets/train/visible_undirected.npz")
    indptr = undirected["indptr"]
    indices = undirected["indices"]
    adjacency = [set(indices[indptr[u]:indptr[u + 1]].tolist()) for u in range(len(indptr) - 1)]

    rng = np.random.default_rng(20261127)
    candidate_centres = np.unique(centres)
    sample = set(rng.choice(candidate_centres,
                             size=min(SAMPLE_CENTRES, candidate_centres.size),
                             replace=False).tolist())
    rows = {c: [] for c in sample}
    for shard_index in range(13):
        z = np.load(REPO / "assets/shards" / f"shard_{shard_index:03d}.npz")
        centre = z["centre"]
        keep = np.isin(centre, list(sample)) & z["valid"] & (z["bc_dist_sq"] == 1)
        if not keep.any():
            continue
        for c, b, cc, curv, klass in zip(centre[keep], z["b"][keep], z["c"][keep],
                                         z["curvature"][keep], z["structure_class"][keep]):
            rows[int(c)].append((int(b), int(cc), float(curv), int(klass)))
    per_group_correlations = []
    per_group_spread = []
    for c, entries in rows.items():
        if len(entries) < 4:
            continue
        entries.sort(key=lambda e: (e[3], e[0], e[1]))
        groups: dict = {}
        for b, cc, curv, klass in entries:
            groups.setdefault(klass, []).append((b, cc, curv))
        for klass, items in groups.items():
            if len(items) < 4:
                continue
            curv = np.array([it[2] for it in items])
            common = np.array([len(adjacency[it[0]] & adjacency[it[1]]) for it in items],
                               dtype=np.float64)
            if np.unique(common).size < 2:
                continue
            per_group_correlations.append(spearman(curv, common))
            per_group_spread.append(float(curv.max() - curv.min()))
    correlations = np.array([c for c in per_group_correlations if np.isfinite(c)])

    summary = {
        "schema": "genrec.issue274.offline_diagnostics/1",
        "question_1_curvature_vs_relation_type": {
            "train_groups": int(train.sum()),
            "fraction_positive_curvature_above_negative": float((diff > 0).mean()),
            "difference_quantiles": {"p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                                     "p75": float(q[3]), "p95": float(q[4])},
            "positive_curvature_median": float(np.median(pos_c[train])),
            "negative_curvature_median": float(np.median(neg_c[train])),
            "note": "the positive pair comes from a closed triangle and the negative from an open wedge, "
                    "so the sign separation reported here is a property of the two structural "
                    "populations, not an independent finding",
        },
        "question_2_extra_information_on_the_matched_subset": {
            "sampled_centres": len(sample),
            "matched_class_groups": len(per_group_correlations),
            "mean_spearman_curvature_vs_common_neighbours": float(correlations.mean()) if correlations.size else None,
            "median_spearman_curvature_vs_common_neighbours": float(np.median(correlations)) if correlations.size else None,
            "fraction_groups_with_positive_spearman": float((correlations > 0).mean()) if correlations.size else None,
            "mean_within_group_curvature_spread": float(np.mean(per_group_spread)) if per_group_spread else None,
            "interpretation": "inside one structure class the outer pairs all share the same graph "
                              "distance (1 for the closed class), so the curvature cannot be backed by "
                              "the graph distance; the common-neighbour count is the independent "
                              "higher-order signal tested here",
        },
        "question_3_residual_correlation_from_shared_items": {
            "train_groups_with_a_shared_item_between_the_two_pairs": int(shared.sum()),
            "fraction": float(shared.mean()),
            "distinct_items_used": int((item_counts > 0).sum()),
            "item_participation_mean": float(participation.mean()),
            "item_participation_p50": float(np.median(participation)),
            "item_participation_p99": float(np.quantile(participation, 0.99)),
            "centres_used": int((centre_counts > 0).sum()),
            "groups_per_centre_mean": float(train.sum() / max((centre_counts > 0).sum(), 1)),
        },
    }
    (OUT / "offline_diagnostics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
