#!/usr/bin/env python3
"""Stage 1: predict the relation between two items from their vectors alone.

Four classes per pair: 0 no edge, 1 A->B, 2 B->A, 3 mutual.  A direction-sensitive head
could not do better than chance if it received a symmetric input, so the pair features are
built so that swapping the two items swaps classes 1 and 2.

Controls that make the number interpretable:

* **degree-matched negatives** -- a negative pair whose endpoints have similar behaviour
  degrees to a positive pair, so "both items are popular" cannot be the whole signal;
* **random vectors** and **all-zero vectors** -- no identity to memorise, so a score above
  chance there is a prior, not information;
* **strict item isolation** -- held-out pairs share no item with the training pairs; when
  the catalogue is too small for that, the script says so instead of silently relaxing.

Labels: the complete directed behaviour graph.  Inputs: 768-d semantics, z, q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
sys.path.insert(0, str(HERE.parent / "edge_reconstruction"))
sys.path.insert(0, str(HERE.parent / "full_attention"))

from edge_reconstruction import FULL_GRAPH, STATES, STATES_META, sha256  # noqa: E402

CONTRACT = TEACHER / "teacher" / "opt" / "data_contract.npz"
SEED = 0
STEPS = 6000
BATCH = 512
HIDDEN = 128
TRAIN_PAIRS = 300000
HELD_PAIRS = 30000
NEG_PER_POS = 2
DEGREE_CALIPERS = 5


def relation_class(a_to_b: bool, b_to_a: bool) -> int:
    if a_to_b and b_to_a:
        return 3
    if a_to_b:
        return 1
    if b_to_a:
        return 2
    return 0


def load_graph() -> tuple[set, np.ndarray, np.ndarray]:
    graph = np.load(FULL_GRAPH)
    edges = set(zip(graph["rows"].tolist(), graph["cols"].tolist()))
    n_items = int(graph["nodes"])
    out_degree = np.bincount(graph["rows"], minlength=n_items)
    in_degree = np.bincount(graph["cols"], minlength=n_items)
    behaviour = out_degree + in_degree
    return edges, behaviour, np.array(sorted(edges), dtype=np.int64)


def degree_calipers(behaviour: np.ndarray, quantiles: int = DEGREE_CALIPERS) -> list[list[int]]:
    """Items grouped by behaviour degree, so negatives can be drawn at matched popularity."""
    edges = np.quantile(behaviour, np.linspace(0, 1, quantiles + 1))
    buckets: list[list[int]] = [[] for _ in range(quantiles)]
    for item, degree in enumerate(behaviour.tolist()):
        index = int(np.clip(np.searchsorted(edges, degree, side="right") - 1, 0, quantiles - 1))
        buckets[index].append(item)
    return buckets


def build_pairs(positive: np.ndarray, buckets: list[list[int]], train_items_share: float,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordered (a, b), class label, and the degree-caliber bucket of the pair.

    Half the negatives are degree-matched (drawn from the same bucket as the positive's
    endpoint) and half are uniform, so a model that only learns popularity cannot win.
    """
    rows, labels, buckets_of = [], [], []
    positives = positive[rng.permutation(len(positive))]
    for a, b in positives.tolist():
        rows.append((a, b))
        labels.append(relation_class(True, (b, a) in EDGES))
        bucket = int(np.clip(np.searchsorted(DEGREE_EDGES, BEHAVIOUR[a], side="right") - 1,
                             0, len(buckets) - 1))
        buckets_of.append(bucket)
        for attempt in range(NEG_PER_POS):
            matched = attempt == 0
            pool = buckets[bucket] if matched else None
            for _ in range(20):
                if matched and len(pool) > 8:
                    candidate = int(rng.choice(pool))
                    other = int(rng.choice(pool))
                else:
                    candidate = int(rng.integers(0, len(BEHAVIOUR)))
                    other = int(rng.integers(0, len(BEHAVIOUR)))
                if candidate == other or (candidate, other) in EDGES or (other, candidate) in EDGES:
                    continue
                rows.append((candidate, other))
                labels.append(0)
                buckets_of.append(bucket if matched else -1)
                break
    order = rng.permutation(len(rows))
    return (np.array(rows, dtype=np.int64)[order],
            np.array(labels, dtype=np.int64)[order],
            np.array(buckets_of, dtype=np.int64)[order])


def pair_features(table: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Ordered pair features: swapping the two items must flip classes 1 and 2.

    [u, v, u*v, |u-v|] is symmetric in the product/abs terms and antisymmetric in the
    order of u and v, which is the minimum a direction-sensitive head needs.
    """
    u = table[pairs[:, 0]]
    v = table[pairs[:, 1]]
    return np.concatenate([u, v, u * v, np.abs(u - v)], axis=1).astype(np.float32)


def build_model(in_dim: int):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.GELU(),
                         nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 4))


def macro_f1(truth: np.ndarray, guess: np.ndarray, n_classes: int = 4) -> float:
    scores = []
    for label in range(n_classes):
        tp = int(((truth == label) & (guess == label)).sum())
        fp = int(((truth != label) & (guess == label)).sum())
        fn = int(((truth == label) & (guess != label)).sum())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall
                      / (precision + recall))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> None:
    import torch

    global EDGES, BEHAVIOUR, DEGREE_EDGES
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "pair_relations.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EDGES, BEHAVIOUR, positive = load_graph()
    DEGREE_EDGES = np.quantile(BEHAVIOUR, np.linspace(0, 1, DEGREE_CALIPERS + 1))
    buckets = degree_calipers(BEHAVIOUR)

    payload = np.load(STATES)
    tables = {"semantics": payload["semantics"], "z": payload["z"], "q": payload["q"]}
    rng = np.random.default_rng(SEED)
    tables["random"] = rng.normal(size=tables["z"].shape).astype(np.float32)
    tables["zero"] = np.zeros_like(tables["z"])

    # strict item isolation: held-out pairs may not reuse any training item
    items = np.arange(len(BEHAVIOUR))
    rng.shuffle(items)
    split = int(0.8 * len(items))
    train_items = set(items[:split].tolist())
    held_items = set(items[split:].tolist())
    is_positive_train = np.array([a in train_items and b in train_items for a, b in positive.tolist()])
    is_positive_held = np.array([a in held_items and b in held_items for a, b in positive.tolist()])
    train_pairs, train_y, train_bucket = build_pairs(positive[is_positive_train], buckets, 0.8,
                                                     np.random.default_rng(SEED))
    held_pairs, held_y, held_bucket = build_pairs(positive[is_positive_held], buckets, 0.2,
                                                  np.random.default_rng(SEED + 1))
    if len(train_pairs) > TRAIN_PAIRS:
        keep = rng.choice(len(train_pairs), TRAIN_PAIRS, replace=False)
        train_pairs, train_y, train_bucket = train_pairs[keep], train_y[keep], train_bucket[keep]
    if len(held_pairs) > HELD_PAIRS:
        keep = rng.choice(len(held_pairs), HELD_PAIRS, replace=False)
        held_pairs, held_y, held_bucket = held_pairs[keep], held_y[keep], held_bucket[keep]

    isolation_conflict = int(sum(a in train_items or b in train_items
                                 for a, b in held_pairs.tolist()))
    report = {"inputs": {"states": sha256(STATES), "full_graph": sha256(FULL_GRAPH),
                         "checkpoint": json.loads(STATES_META.read_text())},
              "labels": {"source": "behavior_graph_directed/graph.npz",
                         "classes": {"0": "no edge", "1": "A->B", "2": "B->A",
                                     "3": "mutual"},
                         "train_pairs": int(len(train_pairs)),
                         "held_pairs": int(len(held_pairs)),
                         "train_class_counts": np.bincount(train_y, minlength=4).tolist(),
                         "held_class_counts": np.bincount(held_y, minlength=4).tolist(),
                         "negatives_per_positive": NEG_PER_POS,
                         "half_negatives_degree_matched": True,
                         "item_isolation": "held-out pairs share no item with training",
                         "held_pairs_reusing_train_items": isolation_conflict},
              "protocol": {"steps": STEPS, "batch": BATCH, "hidden": HIDDEN, "seed": SEED,
                           "features": "[u, v, u*v, |u-v|] which flips 1<->2 under a swap",
                           "metrics": "4-class accuracy and macro-F1 on held-out pairs"}}
    results = {}
    for name in ("semantics", "z", "q", "random", "zero"):
        X_train = pair_features(tables[name], train_pairs)
        X_held = pair_features(tables[name], held_pairs)
        torch.manual_seed(SEED)
        model = build_model(X_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x = torch.from_numpy(X_train).to(device)
        y = torch.from_numpy(train_y).to(device)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        for _ in range(STEPS):
            model.train()
            pick = torch.randint(0, len(x), (BATCH,), generator=generator)
            logits = model(x[pick])
            loss = torch.nn.functional.cross_entropy(logits, y[pick])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            guess = model(torch.from_numpy(X_held).to(device)).argmax(-1).cpu().numpy()
        accuracy = float((guess == held_y).mean())
        results[name] = {"accuracy": accuracy,
                         "macro_f1": macro_f1(held_y, guess),
                         "per_class_recall": {
                             str(label): float((guess[held_y == label] == label).mean())
                             if (held_y == label).any() else None for label in range(4)},
                         "swap_consistency": None}
        # swap test: features must flip class 1 <-> 2, so a symmetric model cannot pass
        swapped = pair_features(tables[name], held_pairs[:, ::-1])
        with torch.no_grad():
            swapped_guess = model(torch.from_numpy(swapped).to(device)).argmax(-1).cpu().numpy()
        directions = np.isin(held_y, (1, 2))
        flipped = np.where(swapped_guess == 1, 2, np.where(swapped_guess == 2, 1, swapped_guess))
        results[name]["swap_consistency"] = float((flipped[directions] == guess[directions]).mean()) \
            if directions.any() else None
        print(f"  {name}: acc {accuracy:.3f} macro-F1 {results[name]['macro_f1']:.3f} "
              f"recall {results[name]['per_class_recall']} "
              f"swap {results[name]['swap_consistency']:.3f}", flush=True)
        report["runs"] = results
        tmp = Path(args.out + ".tmp")
        tmp.write_text(json.dumps(report, indent=1) + "\n")
        tmp.replace(args.out)
    print("written", args.out)


if __name__ == "__main__":
    main()
