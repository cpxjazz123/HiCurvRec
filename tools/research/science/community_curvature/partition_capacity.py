"""Can a behaviour-aligned partition discriminate the behaviour pairs better than the codes?

The alignment measurement showed that the community partition the campaign
learned from the AF3 curvature discriminates the held-out behaviour pairs only
**2.4x** (positives share a community 2.59x above chance, negatives 1.07x),
while every codebook in the campaign discriminates them **3.0x to 16.4x**. That
is why all five community arms lost recall: the partition is a strictly weaker
version of what the identifiers already encode, so any term pulling the codes
towards it can only dilute them.

This script asks the question that decides whether the direction can be saved
at all, and it asks it without a GPU: **a partition built directly from the
transition structure** -- the same structure the codes pick up incidentally --
how well can it discriminate the held-out behaviour pairs? Three partitions are
measured on the same sample and the same yardstick:

* the curvature-guided community partition the campaign learned;
* a k-means partition of the row-normalised transition profile (the behaviour
  a community model would have to learn from the transitions);
* the identifiers' own level-1 codes, for the bar to clear.

If a transition-built partition cannot pass the codes' ratio, the community
mechanism has no room to add anything to the identifiers, whatever its
supervision; if it can, the direction is worth one more arm.

Run:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        tools/research/science/community_curvature/partition_capacity.py
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


def load_pairs(path: Path):
    payload = np.load(path, allow_pickle=True)
    return (np.asarray(payload["anchors"], dtype=np.int64),
            np.asarray(payload["positives"], dtype=np.int64),
            np.asarray(payload["negatives"], dtype=np.int64))


def discrimination(labels: np.ndarray, anchors: np.ndarray, positives: np.ndarray,
                    negatives: np.ndarray) -> dict:
    """The positives' and negatives' share rate and the ratio between them."""
    def share(left: np.ndarray, right: np.ndarray) -> float:
        keep = ((left < labels.size) & (right < labels.size) & (left != right)
                & (left >= 0) & (right >= 0))
        if not keep.any():
            return float("nan")
        return float((labels[left[keep]] == labels[right[keep]]).mean())
    positive = share(anchors, positives)
    negative = share(anchors, negatives[:, 0] if negatives.ndim > 1 else negatives)
    sizes = np.bincount(labels[labels >= 0]).astype(float)
    chance = float(((sizes / sizes.sum()) ** 2).sum())
    return {
        "positives_share": positive,
        "negatives_share": negative,
        "ratio": (positive / negative) if negative else float("inf"),
        "positive_lift": (positive / chance) if chance else None,
        "chance": chance,
    }


def transition_profile(graph: data.Graph):
    """The row-normalised transition profile of every item, as a sparse matrix.

    It is what a community model would have to learn in order to reproduce the
    transition structure: for every item, which items it is adjacent to and how
    strongly. Sparse, because a dense 24587-square matrix is 2.4 GB.
    """
    import scipy.sparse as sp
    rows = np.asarray(graph.nodes, dtype=np.int64)
    adjacency = graph.adjacency.tocsr().astype(np.float32)
    norm = np.sqrt(adjacency.multiply(adjacency).sum(axis=1)).A.ravel()
    scale = sp.diags(1.0 / np.maximum(norm, 1e-12))
    return (scale @ adjacency).tocsr(), rows


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-partition", action="append", default=[],
                        help="a saved soft assignment (.npy) to measure as well")
    arguments = parser.parse_args()

    anchors, positives, negatives = load_pairs(PAIRS)
    results = {}

    kept = {}

    def labels_of(name: str) -> np.ndarray:
        return kept[name]

    # 1. the curvature-guided partition the campaign learned
    labels = np.load(COMMUNITIES).argmax(axis=1)
    kept["curvature_community_64"] = labels
    results["curvature_community_64"] = discrimination(labels, anchors, positives, negatives)

    # 2. a partition built directly from the transition structure
    from sklearn.cluster import MiniBatchKMeans
    graph = data.build_graph("train")
    profile, rows = transition_profile(graph)
    for k in (64, 256, 1024):
        assignment = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=42,
                                     batch_size=4096).fit_predict(profile)
        labels = np.full(data.load_embeddings().shape[0], -1, dtype=np.int64)
        labels[rows] = assignment
        kept[f"transition_kmeans_{k}"] = labels
        results[f"transition_kmeans_{k}"] = discrimination(labels, anchors, positives, negatives)

    # 3. the identifiers' own level-1 codes: the bar to clear
    for name, arm in (("frozen baseline", "TIGER_RQ-VAE"),
                      ("resample reference", None)):
        path = (Path("results/stage2_RQ-VAE", arm, "item_sids.json") if arm else
                Path("results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json"))
        if not path.is_file():
            continue
        table = json.loads(path.read_text())
        # the level-1 code of every item, relabelled to a dense range: the
        # measurement only asks whether two items share a code, and the exported
        # identifiers carry a deterministic disambiguation token
        values = [int(row[0]) for row in table.values()]
        ranking = {value: index for index, value in enumerate(sorted(set(values)))}
        codes = np.zeros(len(table), dtype=np.int64)
        for position, item in enumerate(table):
            codes[position] = ranking[values[position]]
        kept[f"l1_codes_{name.replace(' ', '_')}"] = codes
        results[f"l1_codes_{name.replace(' ', '_')}"] = discrimination(
            codes, anchors, positives, negatives)

    # 3b. any extra partition the caller wants measured (a new arm's output)
    for path in arguments.extra_partition:
        name = Path(path).stem
        extra = np.load(path).argmax(axis=1)
        kept[name] = extra
        results[name] = discrimination(extra, anchors, positives, negatives)

    # 4. the same yardstick on the TRAINING pairs: the partition is fitted on
    # the train graph, so its discrimination there is the one that would matter
    # if it were used to choose what the encoder is contrasted on
    import pandas as pd
    frame = pd.read_parquet(data.TRAIN_PARQUET, columns=["history", "target"])
    train_edges = set()
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first != second:
                train_edges.add((min(first, second), max(first, second)))
    edges = np.asarray(sorted(train_edges), dtype=np.int64)
    train_anchors, train_positives = edges[:, 0], edges[:, 1]
    rng = np.random.default_rng(42)
    train_negatives = rng.integers(0, labels.size, size=train_anchors.size)
    keep = train_negatives != train_anchors
    train_anchors, train_positives = train_anchors[keep], train_positives[keep]
    train_negatives = train_negatives[keep]
    for name in list(kept):
        results["train_pairs__" + name] = discrimination(
            labels_of(name), train_anchors, train_positives, train_negatives)
    for name in ("l1_codes_frozen_baseline", "l1_codes_resample_reference"):
        if name in results:
            results["train_pairs__" + name] = discrimination(
                labels_of(name), train_anchors, train_positives, train_negatives)

    payload = {"file": str(PAIRS), "measurements": results}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "partition_capacity.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                  encoding="utf-8")
    print('%-34s %10s %10s %8s' % ('partition', 'pos', 'neg', 'ratio'))
    for name, value in results.items():
        print('%-34s %10.4f %10.4f %8.2f' % (name, value['positives_share'],
                                             value['negatives_share'], value['ratio']))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
