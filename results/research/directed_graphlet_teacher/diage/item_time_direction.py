#!/usr/bin/env python3
"""Do item-level *time* statistics carry the direction that item-level embeddings miss?

The graph's direction is fully consistent with purchase order (2,507,694 forward edges and
zero backward in the previous check).  That observation alone does not say whether direction
is learnable from item-level quantities -- it only says the graph was built in that order.

This script asks the narrower question with the cheapest possible features: two scalars per
item, its mean relative position inside user sequences and its occurrence count.  If those
predict the direction of a held-out edge, item-level statistics do encode it and the
embeddings are simply missing the channel.  If they do not, the signal is weak at item
level even when the time is known.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
GRAPH = REPO / "results" / "research" / "behavior_graph_directed" / "graph.npz"
OUT = HERE / "item_time_direction.json"
N_ITEMS = 24587
N_PAIRS = 20000


def main() -> None:
    frame = pd.read_parquet(TRAIN, columns=["user", "history", "target"])
    position_sum = np.zeros(N_ITEMS)
    position_count = np.zeros(N_ITEMS)
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        sequence = list(history) + [target]
        length = max(len(sequence) - 1, 1)
        for rank, item in enumerate(sequence):
            position_sum[item] += rank / length
            position_count[item] += 1
    mean_position = np.where(position_count > 0,
                             position_sum / np.maximum(position_count, 1), np.nan)

    graph = np.load(GRAPH)
    rows = graph["rows"].astype(np.int64)
    cols = graph["cols"].astype(np.int64)
    directed = set(zip(rows.tolist(), cols.tolist()))
    one_way = [(int(a), int(b)) for a, b in zip(rows.tolist(), cols.tolist())
               if (b, a) not in directed]
    rng = np.random.default_rng(0)
    chosen = [one_way[i] for i in rng.permutation(len(one_way))[:N_PAIRS]]
    forward = np.array(chosen, dtype=np.int64)
    backward = forward[:, ::-1].copy()
    both = np.concatenate([forward, backward], axis=0)
    labels = np.concatenate([np.ones(len(forward)), np.zeros(len(forward))])
    features = np.stack([mean_position[both[:, 0]], mean_position[both[:, 1]],
                         np.log1p(position_count[both[:, 0]]),
                         np.log1p(position_count[both[:, 1]])], axis=1)
    valid = np.isfinite(features).all(axis=1)
    features, labels = features[valid], labels[valid]
    order = np.random.default_rng(1).permutation(len(features))
    cut = int(0.7 * len(order))
    model = LogisticRegression(max_iter=1000).fit(features[order[:cut]], labels[order[:cut]])
    predicted = model.predict_proba(features[order[cut:]])[:, 1]
    truth = labels[order[cut:]]
    report = {"pairs_used": int(len(features)),
              "features": "mean relative sequence position and log occurrence count of both "
                          "endpoints",
              "auc": float(roc_auc_score(truth, predicted)),
              "direction_accuracy": float(((predicted > 0.5) == (truth > 0.5)).mean()),
              "mean_position_shift_source_minus_target":
                  float((mean_position[forward[:, 0]] - mean_position[forward[:, 1]]).mean()),
              "read": "item-level time statistics alone give a direction signal only this "
                      "large; any claim that item vectors cannot carry direction has to be "
                      "weaker than 'impossible', and any claim that they can needs a channel "
                      "beyond position"}
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
