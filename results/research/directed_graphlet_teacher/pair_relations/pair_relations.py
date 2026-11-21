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


def degree_calipers(behaviour: np.ndarray, allowed: np.ndarray | None = None,
                    quantiles: int = DEGREE_CALIPERS) -> list[list[int]]:
    """Items grouped by behaviour degree, so negatives can be drawn at matched popularity."""
    edges = np.quantile(behaviour, np.linspace(0, 1, quantiles + 1))
    buckets: list[list[int]] = [[] for _ in range(quantiles)]
    for item, degree in enumerate(behaviour.tolist()):
        if allowed is not None and not allowed[item]:
            continue
        index = int(np.clip(np.searchsorted(edges, degree, side="right") - 1, 0, quantiles - 1))
        buckets[index].append(item)
    return buckets


def degree_calipers_within(allowed: np.ndarray) -> list[list[int]]:
    return degree_calipers(BEHAVIOUR, allowed)


def bucket_of(item: int) -> int:
    return int(np.clip(np.searchsorted(DEGREE_EDGES, BEHAVIOUR[item], side="right") - 1,
                       0, DEGREE_CALIPERS - 1))


def build_pairs(unit_edges: np.ndarray, buckets: list[list[int]], train_items_share: float,
                rng: np.random.Generator, target_pairs: int) -> tuple:
    """Ordered (a, b), class label, and the degree-caliber bucket of the pair.

    Half the negatives are degree-matched (drawn from the same bucket as the positive's
    endpoint) and half are uniform, so a model that only learns popularity cannot win.
    """
    labels_of = {}

    def add(first: int, second: int, wanted: int, bucket: int) -> None:
        """Store the unordered pair and derive the label from the *given* order.

        The previous version computed the class from the canonical storage orientation and
        then stored the pair in that orientation, so the two were always read as class 1:
        a reverse edge was "A->B" whose endpoints had been swapped.  Here the class is
        recomputed from (first, second) itself, so 1 and 2 depend only on which item is
        written first.
        """
        key = (min(first, second), max(first, second))
        if key in labels_of:
            return
        labels_of[key] = relation_class((first, second) in EDGES, (second, first) in EDGES)
        buckets_of.append(bucket)
        ordered.append((first, second))

    ordered: list[tuple[int, int]] = []
    buckets_of: list[int] = []
    all_items = np.arange(len(BEHAVIOUR))
    rng.shuffle(all_items)
    # class 3 first: mutual pairs are rare (8242), so collect every one available
    for index, (a, b) in enumerate(UNIT_EDGES[rng.permutation(len(UNIT_EDGES))]):
        if len(labels_of) >= target_pairs:
            break
        if (a, b) in EDGES and (b, a) in EDGES:
            add(a, b, 3, bucket_of(a))
    # Classes 1 and 2 are defined by the *input order*, not by the graph: either item may
    # be written first.  Sampling both orders gives both classes from the same edges and
    # removes the asymmetry that a fixed canonical orientation would create.
    counts = {1: 0, 2: 0, 0: 0, 3: len(labels_of)}
    one_way = [(a, b) for a, b in UNIT_EDGES.tolist()
               if ((a, b) in EDGES) != ((b, a) in EDGES)]
    rng.shuffle(one_way)
    for a, b in one_way:
        if counts[1] >= target_pairs // 4 and counts[2] >= target_pairs // 4:
            break
        forward_first = rng.random() < 0.5
        first, second = (a, b) if forward_first else (b, a)
        wanted = relation_class((first, second) in EDGES, (second, first) in EDGES)
        if wanted in (1, 2) and counts[wanted] < target_pairs // 4:
            add(first, second, wanted, bucket_of(first))
            counts[wanted] += 1
            # the mirrored pair of the *same* evidence: the model must learn that the
            # class depends on the order, so the same two items are also offered in the
            # other order with the other label (class 0 rows are what keeps it honest)
            mirror_label = relation_class((second, first) in EDGES, (first, second) in EDGES)
            if mirror_label in (1, 2) and counts[mirror_label] < target_pairs // 4:
                add(second, first, mirror_label, bucket_of(second))
                counts[mirror_label] += 1
    # class 0: degree-matched negatives; both endpoints must come from the same bucket,
    # which is itself restricted to this split's item pool
    pool = np.array(sorted({item for bucket in buckets for item in bucket}), dtype=np.int64)
    while counts[0] < target_pairs // 2 and len(labels_of) < target_pairs:
        first = int(rng.choice(pool))
        candidates = [item for item in buckets[bucket_of(first)]
                      if item != first and (first, item) not in EDGES
                      and (item, first) not in EDGES]
        if not candidates:
            continue
        second = int(rng.choice(candidates))
        add(first, second, 0, bucket_of(first))
        counts[0] += 1
    mismatched = 0
    for first, second in ordered:
        expected = relation_class((first, second) in EDGES, (second, first) in EDGES)
        if labels_of[(min(first, second), max(first, second))] != expected:
            mismatched += 1
    if mismatched:
        raise SystemExit(f"{mismatched} stored labels disagree with the adjacency matrix")
    order = rng.permutation(len(ordered))
    return (np.array([ordered[i] for i in order], dtype=np.int64),
            np.array([labels_of[(min(a, b), max(a, b))] for a, b in
                      [ordered[i] for i in order]], dtype=np.int64),
            np.array([buckets_of[i] for i in order], dtype=np.int64))


def pair_features(table: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Ordered pair features: swapping the two items must flip classes 1 and 2.

    [u, v, u*v, |u-v|] is symmetric in the product/abs terms and antisymmetric in the
    order of u and v, which is the minimum a direction-sensitive head needs.
    """
    u = table[pairs[:, 0]]
    v = table[pairs[:, 1]]
    # u and v appear in a fixed order, so a linear layer legitimately reads direction;
    # the extra antisymmetric channel (u - v) makes that signal explicit instead of
    # forcing the model to learn a difference from two 768-wide blocks.
    return np.concatenate([u, v, u * v, np.abs(u - v), u - v], axis=1).astype(np.float32)


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

    global EDGES, BEHAVIOUR, DEGREE_EDGES, UNIT_EDGES
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "pair_relations.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EDGES, BEHAVIOUR, positive = load_graph()
    # one row per unordered pair that has at least one direction
    UNIT_EDGES = np.array(sorted({(min(a, b), max(a, b)) for a, b in EDGES}), dtype=np.int64)
    DEGREE_EDGES = np.quantile(BEHAVIOUR, np.linspace(0, 1, DEGREE_CALIPERS + 1))
    buckets = degree_calipers(BEHAVIOUR)

    payload = np.load(STATES)
    tables = {"semantics": payload["semantics"], "z": payload["z"], "q": payload["q"]}
    rng = np.random.default_rng(SEED)
    tables["random"] = rng.normal(size=tables["z"].shape).astype(np.float32)
    tables["zero"] = np.zeros_like(tables["z"])

    # strict item isolation: build each split from its own item pool only, so no
    # sampler can reach across the boundary
    items = np.arange(len(BEHAVIOUR))
    rng.shuffle(items)
    split = int(0.8 * len(items))
    train_items = np.sort(items[:split])
    held_items = np.sort(items[split:])
    train_set, held_set = set(train_items.tolist()), set(held_items.tolist())
    unit_train = UNIT_EDGES[np.isin(UNIT_EDGES[:, 0], train_items)
                            & np.isin(UNIT_EDGES[:, 1], train_items)]
    unit_held = UNIT_EDGES[np.isin(UNIT_EDGES[:, 0], held_items)
                           & np.isin(UNIT_EDGES[:, 1], held_items)]

    def build_within(unit: np.ndarray, item_pool: np.ndarray, target: int, seed: int):
        """Same four-class sampler, restricted to one item pool for every draw.

        The sampler draws its items through the module-level pools, so the pools are
        swapped for this split's own before calling it.
        """
        global EDGES, BEHAVIOUR, UNIT_EDGES
        local = np.zeros(len(BEHAVIOUR), dtype=bool)
        local[item_pool] = True
        saved = (EDGES, BEHAVIOUR, UNIT_EDGES)
        pool_edges = {(a, b) for a, b in EDGES if local[a] and local[b]}
        EDGES = pool_edges
        UNIT_EDGES = unit
        BEHAVIOUR = np.where(local, BEHAVIOUR, 0)   # never draw an item outside the pool
        pairs, labels, buckets = build_pairs(unit, degree_calipers_within(local), 0.0,
                                             np.random.default_rng(seed), target)
        bad = int(sum(not (local[a] and local[b]) for a, b in pairs.tolist()))
        EDGES, BEHAVIOUR, UNIT_EDGES = saved
        if bad:
            raise SystemExit(f"{bad} sampled pairs leave the split's item pool")
        return pairs, labels, buckets, 0

    train_pairs, train_y, train_bucket, train_bad = build_within(
        unit_train, train_items, TRAIN_PAIRS, SEED)
    held_pairs, held_y, held_bucket, held_bad = build_within(
        unit_held, held_items, HELD_PAIRS, SEED + 1)
    isolation_conflict = int(train_bad + held_bad)
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
