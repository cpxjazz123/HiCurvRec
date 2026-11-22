"""What does the AF3 curvature know about the behaviour the codes must capture?

The campaign's five community arms all failed to move the Stage-3 recall, and
the repulsion arm failed in the most informative way: pushing the items of a
community onto *different* level-1 codes **raised** the held-out behaviour
pairs' hit rate. That only makes sense if the community partition and the
behaviour pairs are close to orthogonal: if a behaviour pair usually sits
*inside* a community, pushing the community apart would have separated it.

This script measures the two things that decide it, on the same independent
sample the quality gate uses:

1. **the overlap lift** -- the share of behaviour-positive pairs whose two items
   share a community, against the share a random labelling of the same sizes
   would give;
2. **the curvature's information about the behaviour edges** -- the mean AF3
   curvature of the edges that carry a behaviour pair against the mean of the
   edges that do not, and the rank correlation between the two.

Run:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        tools/research/science/community_curvature/alignment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.science.community_curvature import data  # noqa: E402

OUT = data.REPO / "results" / "research" / "issue121" / "community_curvature"
PAIRS = data.REPO / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_valid_s42.npz"
META = PAIRS.with_suffix(".meta.json")
COMMUNITIES = OUT / "pi_E_free_64_hyper_64_curvgcn.npy"


def load_pairs(path: Path, meta: Path):
    """The held-out behaviour pairs: the anchors, their positives and negatives."""
    payload = np.load(path, allow_pickle=True)
    fields = list(payload.keys())
    anchors = np.asarray(payload["anchors"], dtype=np.int64)
    positives = np.asarray(payload["positives"], dtype=np.int64)
    negatives = np.asarray(payload["negatives"], dtype=np.int64)
    return anchors, positives, negatives, fields


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """The rank correlation of two arrays."""
    def ranks(values):
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(values.size, dtype=np.float64)
        return ranks
    left, right = ranks(np.asarray(left, dtype=float)), ranks(np.asarray(right, dtype=float))
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt((left ** 2).sum() * (right ** 2).sum()))
    return float((left * right).sum() / denominator) if denominator else 0.0


def main() -> int:
    probabilities = np.load(COMMUNITIES)
    labels = probabilities.argmax(axis=1)
    anchors, positives, negatives, fields = load_pairs(PAIRS, META)
    print("pairs file fields:", fields, "| positives", positives.shape, "| negatives", negatives.shape)

    sizes = np.bincount(labels, minlength=probabilities.shape[1]).astype(float)
    chance = float((sizes / sizes.sum() ** 1.0 * sizes / sizes.sum()).sum())

    def share_same(pairs: np.ndarray) -> float:
        if pairs.ndim == 2:
            left, right = pairs[:, 0], pairs[:, 1]
        else:
            left, right = pairs, np.roll(pairs, 1)
        keep = (left < labels.size) & (right < labels.size) & (left != right)
        if not keep.any():
            return float("nan")
        return float((labels[left[keep]] == labels[right[keep]]).mean())

    positive_share = share_same(np.stack([anchors, positives], axis=1))
    negative_share = share_same(np.stack([anchors, negatives[:, 0]], axis=1))

    # the behaviour edges and the rest of the graph, with their AF3 curvature
    graph = data.build_graph("train")
    curvature = data.curvature_of(graph)
    nodes = np.asarray(graph.nodes, dtype=np.int64)
    edges = {(int(a), int(b)): float(v)
             for a, b, v in zip(nodes[graph.edges[0]], nodes[graph.edges[1]],
                                curvature["af3"])}
    behaviour = set()
    for row in positives:
        for item in np.atleast_1d(row):
            behaviour.add(int(item))
    # the behaviour edges of the training split, read the way the graph is built
    import pandas as pd
    frame = pd.read_parquet(data.TRAIN_PARQUET, columns=["history", "target"])
    pairs_in_sequences = set()
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first != second:
                pairs_in_sequences.add((min(first, second), max(first, second)))

    on_behaviour = [value for key, value in edges.items() if key in pairs_in_sequences]
    off_behaviour = [value for key, value in edges.items() if key not in pairs_in_sequences]
    curvature_separation = (
        (float(np.mean(on_behaviour)) - float(np.mean(off_behaviour)))
        if on_behaviour and off_behaviour else None)

    payload = {
        "community_partition": str(COMMUNITIES),
        "n_items": int(labels.size),
        "n_communities": int(probabilities.shape[1]),
        "largest_community_share": float(sizes.max() / sizes.sum()),
        "behaviour_pairs": {
            "file": str(PAIRS),
            "positives_same_community": positive_share,
            "negatives_same_community": negative_share,
            "chance_same_community": chance,
            "positive_overlap_lift": (positive_share / chance) if chance else None,
            "negative_overlap_lift": (negative_share / chance) if chance else None,
        },
        "af3_curvature_on_the_graph": {
            "behaviour_edges": len(on_behaviour),
            "other_edges": len(off_behaviour),
            "mean_af3_behaviour_edges": float(np.mean(on_behaviour)) if on_behaviour else None,
            "mean_af3_other_edges": float(np.mean(off_behaviour)) if off_behaviour else None,
            "separation": curvature_separation,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "community_behaviour_alignment.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
