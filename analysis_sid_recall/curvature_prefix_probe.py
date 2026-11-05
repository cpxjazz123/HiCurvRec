"""Does the behaviour curvature say anything about SID prefix sharing?

The design under test asks the SID hierarchy to respect a RELATIVE relation
about behaviour: two items whose behaviour neighbourhoods overlap heavily (a
local cohesion: many shared neighbours, many triangles) should share a COARSER
SID prefix with each other than a pair that is connected but whose surroundings
lean to different local structures (a local bridge).

That claim has two halves, and this probe measures both on the frozen
validation pairs, with the training split as the only source of behaviour:

1. the SIGNAL: is the positive's neighbourhood overlap really higher than the
   negatives'?
2. the HEADROOM: does the SID that exists today already put the high-overlap
   pairs together? If it does, a loss that enforces the relation has nothing
   left to gain; if it does not, the relation is exactly what a training
   signal could fix.

The overlap statistic is the triangle count through the edge, |N(A) & N(B)|:
the design's own definition of a cohesion edge, and it needs no transport, so
52,855 anchors x 11 partners are affordable.

Usage:
    python3 analysis_sid_recall/curvature_prefix_probe.py --arm TIGER_RQ-VAE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
PAIRS = REPO / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_valid_s42.npz"

#: The arms the probe reads the identifiers of.
ARMS = {
    "TIGER_RQ-VAE": REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json",
    "TIGER_BEHAVIOR_SAMPLE_RQ-VAE": (REPO / "results" / "stage2_RQ-VAE" /
                                     "TIGER_BEHAVIOR_SAMPLE_RQ-VAE" / "resample" /
                                     "item_sids.json"),
    "TIGER_pmi-geomargin-l1-rq": (REPO / "results" / "stage2_RQ-VAE" /
                                   "TIGER_pmi-geomargin-l1-rq" / "item_sids.json"),
}


def behaviour_graph(frame) -> tuple:
    """The training split's UNDIRECTED co-occurrence graph (train only)."""
    weights: dict[tuple[int, int], float] = {}
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first == second:
                continue
            key = (first, second) if first < second else (second, first)
            weights[key] = weights.get(key, 0.0) + 1.0
    nodes = sorted({item for edge in weights for item in edge})
    index = {item: position for position, item in enumerate(nodes)}
    rows, cols, data = [], [], []
    for (first, second), weight in weights.items():
        rows.append(index[first])
        cols.append(index[second])
        data.append(weight)
    graph = sp.coo_matrix((data, (rows, cols)),
                           shape=(len(nodes), len(nodes))).tocsr()
    graph = graph.maximum(graph.transpose().tocsr())
    graph.setdiag(0)
    graph.eliminate_zeros()
    return nodes, index, graph


def triangles(graph: sp.csr_matrix, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The number of shared neighbours of every pair: the triangles through it.

    A pair whose item is not in the training graph has no triangles: the
    statistic is undefined, not zero, so it is reported as -1.
    """
    if len(left) == 0:
        return np.zeros(0)
    valid = (left >= 0) & (right >= 0)
    safe_left = np.where(valid, left, 0)
    safe_right = np.where(valid, right, 0)
    common = graph[safe_left].multiply(graph[safe_right]).sum(axis=1)
    counts = np.asarray(common).ravel()
    return np.where(valid, counts, -1.0)


def prefixes(path: Path) -> dict:
    """The item -> SID table of one arm."""
    rows = json.loads(Path(path).read_text())
    table = {}
    for key, value in rows.items():
        table[int(key)] = list(value)[:3]
    return table


def probe(arm: str, path: Path, anchors: int, seed: int) -> dict:
    frame = pd.read_parquet(TRAIN)
    nodes, index, graph = behaviour_graph(frame)
    pairs = np.load(PAIRS)
    all_anchors = pairs["anchors"]
    positives = pairs["positives"]
    negatives = pairs["negatives"]
    generator = np.random.default_rng(seed)
    chosen = generator.choice(len(all_anchors), size=min(anchors, len(all_anchors)),
                              replace=False)
    table = prefixes(path)
    # the graph's own item -> row mapping: the pairs address embedding ids, and
    # only the items the training graph contains have a row
    position = index

    def graph_rows(ids):
        flat = np.asarray(ids).ravel()
        return np.array([position.get(int(item), -1) for item in flat])

    def sid_prefix_agreement(left_ids, right_ids):
        """How coarse the two items' SIDs are: 3, 2 or 1 shared levels."""
        levels = []
        for first, second in zip(left_ids, right_ids):
            first = table.get(int(first))
            second = table.get(int(second))
            if first is None or second is None:
                levels.append(0)
                continue
            shared = 0
            for level in range(3):
                if first[level] == second[level]:
                    shared += 1
                else:
                    break
            levels.append(shared)
        return np.array(levels)

    anchor_ids = all_anchors[chosen]
    positive_ids = positives[chosen]
    negative_ids = negatives[chosen]

    anchor_rows = graph_rows(anchor_ids)
    positive_rows = graph_rows(positive_ids)
    negative_rows = graph_rows(negative_ids).reshape(negative_ids.shape)

    positive_triangles = triangles(graph, anchor_rows, positive_rows)
    negative_triangles = np.empty(negative_rows.shape, dtype=float)
    for row in range(negative_rows.shape[0]):
        negative_triangles[row] = triangles(graph, np.full(negative_rows.shape[1],
                                                           anchor_rows[row]),
                                           negative_rows[row])

    positive_levels = sid_prefix_agreement(anchor_ids, positive_ids)
    negative_levels = np.empty(negative_ids.shape, dtype=float)
    for row in range(negative_ids.shape[0]):
        negative_levels[row] = sid_prefix_agreement(
            [anchor_ids[row]] * negative_ids.shape[1], negative_ids[row])

    # 1. the signal: is the positive's overlap higher?
    valid_anchor = positive_triangles >= 0
    negative_defined = negative_triangles >= 0
    signal = {
        "anchors_kept": int(valid_anchor.sum()),
        "anchors_dropped_not_in_the_training_graph": int((~valid_anchor).sum()),
        "median_triangles_positive": float(np.median(positive_triangles[valid_anchor])),
        "median_triangles_negative": float(
            np.median(negative_triangles[negative_defined])),
        "positive_beats_the_mean_negative_rate": float(np.mean(
            positive_triangles[valid_anchor] >
            negative_triangles[valid_anchor].mean(axis=1))),
        "positives_with_any_triangle_rate": float(
            np.mean(positive_triangles[valid_anchor] > 0)),
        "negatives_with_any_triangle_rate": float(
            np.mean(negative_triangles[negative_defined] > 0)),
    }

    # 2. the headroom: the prefix rate by overlap tertile, and the paired form
    flat_triangles = positive_triangles[valid_anchor]
    flat_levels = positive_levels[valid_anchor]
    order = np.argsort(flat_triangles)
    tertile = np.zeros(len(order), dtype=int)
    tertile[order] = np.repeat([0, 1, 2], len(order) // 3 + 1)[:len(order)]
    by_tertile = {}
    for group in (0, 1, 2):
        mask = tertile == group
        if not mask.any():
            continue
        by_tertile[str(group)] = {
            "pairs": int(mask.sum()),
            "median_triangles": float(np.median(flat_triangles[mask])),
            "share_l1": float(np.mean(flat_levels[mask] >= 1)),
            "share_l1_l2": float(np.mean(flat_levels[mask] >= 2)),
            "share_l3": float(np.mean(flat_levels[mask] >= 3)),
        }
    # the decisive measurement for a COHESION mechanism: among the pairs that DO
    # share the first level, is the behaviour overlap larger than among the pairs
    # that do not? If it is not, the SID today groups BRIDGES rather than
    # cohesive pairs, and a cohesion loss has real headroom.
    cohesion = {}
    for label, mask in (("shares_l1", flat_levels >= 1), ("does_not_share_l1", flat_levels < 1)):
        if not mask.any():
            continue
        cohesion[label] = {
            "pairs": int(mask.sum()),
            "median_triangles": float(np.median(flat_triangles[mask])),
            "mean_triangles": float(np.mean(flat_triangles[mask])),
            "share_with_any_triangle": float(np.mean(flat_triangles[mask] > 0)),
        }
    if "shares_l1" in cohesion and "does_not_share_l1" in cohesion:
        cohesion["the_sid_groups_cohesive_pairs"] = bool(
            cohesion["shares_l1"]["median_triangles"] >
            cohesion["does_not_share_l1"]["median_triangles"])

    # the paired relation the design asks for: within an anchor, the partner
    # with the larger overlap should share the prefix more often
    paired_hits = 0
    paired_total = 0
    for row in np.flatnonzero(valid_anchor):
        negative_row = negative_triangles[row]
        if len(negative_row) == 0:
            continue
        best = int(np.argmax(negative_row))
        if negative_row[best] < 0 or negative_row[best] <= positive_triangles[row]:
            continue
        paired_total += 1
        if negative_levels[row][best] >= positive_levels[row]:
            paired_hits += 1
    paired = {
        "anchors_where_the_bridge_beats_the_positive_on_overlap": paired_total,
        "of_those_the_bridge_shares_the_same_or_a_coarser_prefix":
            paired_hits,
        "rate": (paired_hits / paired_total) if paired_total else None,
    }
    return {"arm": arm, "anchors": int(len(chosen)), "signal": signal,
            "prefix_rate_by_overlap_tertile": by_tertile,
            "cohesion_of_the_groups": cohesion,
            "paired_relation": paired}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    arguments = parser.parse_args()
    report = {}
    for arm in arguments.arms:
        path = ARMS.get(arm)
        if path is None or not Path(path).is_file():
            report[arm] = {"status": "NOT_AVAILABLE", "path": str(path)}
            continue
        report[arm] = probe(arm, Path(path), arguments.anchors, arguments.seed)
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
