"""Can behaviour relations be recovered from a pair of item vectors?

The three-graph comparison asked whether behaviour-connected items sit close together, which
presumes that a behaviour edge means semantic similarity.  It does not: a phone and its case are
bought together, a phone and another phone are alike.  So the question is asked the other way
round -- a small model is trained to read a **relation** off a **pair** of vectors:

    input   [u - v, u * v] of the two items' vectors (permutation invariant, so the model cannot
            memorise which item is which)
    target  whether the two items are connected in the TRAIN behaviour graph
    measure ROC-AUC and PR-AUC on relations the model never saw

Two evaluations, because they answer different questions:

    pair-level  the edges and non-edges are split; the held-out relations involve items the model
                saw, so this asks whether the relation can be read off vectors of known items
    item-level  a fifth of the items is held out entirely; the held-out relations involve items
                the model never saw, which is the harder question

Negatives come in two kinds: random pairs, and **degree-matched** pairs (the two endpoints are
drawn from pools of items with the same degrees as the edge's endpoints), because a random
non-edge is usually trivially far away and would make any model look good.

The run is done twice per arm: the vectors before quantization and the model's own quantized
readout.  What the difference says is how much of the behaviour relation the codebook cost.

    python3 results/research/issue215/relation_predictability.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
JOURNAL = OUT / "relation_predictability.json"
ARMS = ("TIGER", "F0", "B", "C", "D")
SEED = 42
#: Full-batch steps: one gradient step per epoch over every training pair.  The pair features are
#: built once and kept on the device, so a training is a few hundred small matrix products
#: instead of tens of thousands of launches -- the same protocol for every predictor, and the
#: protocol is what makes the arms comparable.
EPOCHS = 600
BATCH = 0
HIDDEN = 256
LR = 2e-3
DEGREE_BINS = 20
NEGATIVES_PER_EDGE = 1


def behaviour_edges() -> np.ndarray:
    frame = pd.read_parquet(TRAIN, columns=["seen_history", "target"])
    pairs = set()
    for history, target in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)):
        if history is None or len(history) == 0:
            continue
        pairs.add((int(history[-1]), int(target)))
    return np.asarray(sorted(pairs), dtype=np.int64)


def sample_negatives(edges: np.ndarray, degrees: np.ndarray, n_items: int, rng: np.random.Generator,
                     matched: bool) -> np.ndarray:
    """Non-edges: random pairs, or pairs whose endpoints have the same degrees as the edge's."""
    order = np.argsort(degrees, kind="mergesort")
    rank = np.empty(n_items, dtype=np.int64)
    rank[order] = np.arange(n_items)
    bins = np.minimum((rank * DEGREE_BINS) // max(n_items, 1), DEGREE_BINS - 1)
    pools = [np.flatnonzero(bins == b) for b in range(DEGREE_BINS)]
    edge_set = {(int(a), int(b)) for a, b in edges.tolist()}
    out = []
    for a, b in edges.tolist():
        for _ in range(NEGATIVES_PER_EDGE):
            for _ in range(64):
                if matched:
                    pool_a = pools[bins[a]]
                    pool_b = pools[bins[b]]
                    if pool_a.size == 0 or pool_b.size == 0:
                        continue
                    left = int(pool_a[rng.integers(pool_a.size)])
                    right = int(pool_b[rng.integers(pool_b.size)])
                else:
                    left = int(rng.integers(n_items))
                    right = int(rng.integers(n_items))
                if left == right:
                    continue
                key = (min(left, right), max(left, right))
                if key in edge_set:
                    continue
                out.append(key)
                break
    return np.asarray(out, dtype=np.int64)


def pair_features(vectors: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """[u - v, u * v]: invariant under swapping the two items, so the model reads a relation.

    Built on the device: materialising the features for every pair on the host costs gigabytes for
    the 768-dimensional reference and dominates the run.
    """
    left, right = vectors[pairs[:, 0]], vectors[pairs[:, 1]]
    return torch.cat([left - right, left * right], dim=1)


def train_and_score(features: np.ndarray, positives: np.ndarray, negatives: np.ndarray,
                    train_positive: np.ndarray, train_negative: np.ndarray,
                    test_positive: np.ndarray, test_negative: np.ndarray,
                    device: torch.device) -> dict:
    vectors = torch.tensor(features, dtype=torch.float32, device=device)
    train_pairs = torch.tensor(np.concatenate([positives[train_positive], negatives[train_negative]]),
                               dtype=torch.long, device=device)
    test_pairs = torch.tensor(np.concatenate([positives[test_positive], negatives[test_negative]]),
                              dtype=torch.long, device=device)
    y_train = torch.cat([torch.ones(train_positive.size), torch.zeros(train_negative.size)]).to(device)
    y_test = np.concatenate([np.ones(test_positive.size), np.zeros(test_negative.size)])

    with torch.no_grad():
        sample = pair_features(vectors, train_pairs[: min(len(train_pairs), 200_000)])
        mean = sample.mean(0, keepdim=True)
        scale = sample.std(0, keepdim=True).clamp(min=1e-6)
        x_train = (pair_features(vectors, train_pairs) - mean) / scale
        x_test = (pair_features(vectors, test_pairs) - mean) / scale

    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(sample.shape[1], HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1),
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x_train), y_train.unsqueeze(1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(x_test)).cpu().numpy().ravel()
    return {"roc_auc": float(roc_auc_score(y_test, probability)),
            "pr_auc": float(average_precision_score(y_test, probability)),
            "n_test_positive": int(test_positive.size), "n_test_negative": int(test_negative.size)}


def evaluate(vectors: np.ndarray, positives: np.ndarray, negatives: np.ndarray, degrees: np.ndarray,
             n_items: int, device: torch.device) -> dict:
    rng = np.random.default_rng(SEED)
    # pair-level split: relations the model never saw
    order = rng.permutation(positives.shape[0])
    cut = int(0.8 * order.size)
    pair_train, pair_test = order[:cut], order[cut:]
    neg_order = rng.permutation(negatives.shape[0])
    neg_cut = int(0.8 * neg_order.size)
    out = {"pair_level": {
        "random_negatives": train_and_score(vectors, positives, negatives, pair_train,
                                             neg_order[:neg_cut], pair_test, neg_order[neg_cut:], device),
    }}
    # item-level split: a fifth of the items is held out entirely
    item_order = rng.permutation(n_items)
    held = set(item_order[: n_items // 5].tolist())
    item_train = np.asarray([i for i in range(positives.shape[0])
                             if positives[i, 0] not in held and positives[i, 1] not in held], dtype=np.int64)
    item_test = np.asarray([i for i in range(positives.shape[0])
                             if positives[i, 0] in held and positives[i, 1] in held], dtype=np.int64)
    if item_test.size and item_train.size:
        neg_item_train = neg_order[: int(0.8 * neg_order.size)]
        out["item_level"] = {
            "random_negatives": train_and_score(vectors, positives, negatives, item_train,
                                                neg_item_train, item_test,
                                                neg_order[int(0.8 * neg_order.size):], device),
            "held_items": int(len(held)),
        }
    return out


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edges = behaviour_edges()
    n_items = int(max(edges.max(), 0)) + 1
    degrees = np.bincount(edges.ravel(), minlength=n_items)
    positives = edges
    print(f"behaviour edges {len(edges)} | items {n_items}", flush=True)

    report = {"edges": int(len(edges)), "seed": SEED, "epochs": EPOCHS, "predictors": {}}

    rng = np.random.default_rng(SEED + 1)
    matched = sample_negatives(positives, degrees, n_items, rng, matched=True)
    random_negatives = sample_negatives(positives, degrees, n_items, rng, matched=False)
    print(f"negatives: {len(matched)} degree-matched, {len(random_negatives)} random", flush=True)

    def run(name: str, vectors: np.ndarray) -> None:
        block = {}
        for negative_name, negative_set in (("degree_matched", matched), ("random", random_negatives)):
            scores = evaluate(vectors, positives, negative_set, degrees, n_items, device)
            for level, values in scores.items():
                if level == "item_level":
                    block.setdefault("degree_matched" if negative_name == "degree_matched" else "random", {}) \
                         .setdefault(level, {}).update(values["random_negatives"])
                else:
                    block.setdefault(negative_name, {}).setdefault(level, {}).update(values["random_negatives"])
        report["predictors"][name] = block
        for negative_name in ("degree_matched", "random"):
            for level in ("pair_level", "item_level"):
                if level not in block.get(negative_name, {}):
                    continue
                values = block[negative_name][level]
                print(f"{name:<16} {negative_name:<14} {level:<11} AUC {values['roc_auc']:.4f} "
                      f"| PR-AUC {values['pr_auc']:.4f} (+{values['n_test_positive']}/-{values['n_test_negative']})",
                      flush=True)

    embedding = np.load(EMBEDDING).astype(np.float32)
    run("semantic 768d", embedding)
    for arm in ARMS:
        z_path, q_path = VECTORS / f"{arm}_z.npy", VECTORS / f"{arm}_q.npy"
        if not z_path.is_file() or not q_path.is_file():
            continue
        run(f"{arm} z (pre)", np.load(z_path).astype(np.float32))
        run(f"{arm} q (post)", np.load(q_path).astype(np.float32))

    JOURNAL.write_text(json.dumps(report, indent=1, default=float))
    print("\nquantization effect on recoverable relations (post minus pre):")
    for arm in ARMS:
        pre = report["predictors"].get(f"{arm} z (pre)", {})
        post = report["predictors"].get(f"{arm} q (post)", {})
        for negative_name in ("degree_matched", "random"):
            for level in ("pair_level", "item_level"):
                a = pre.get(negative_name, {}).get(level)
                b = post.get(negative_name, {}).get(level)
                if not a or not b:
                    continue
                print(f"  {arm:<6} {negative_name:<14} {level:<11} AUC {a['roc_auc']:.4f} -> {b['roc_auc']:.4f} "
                      f"({b['roc_auc'] - a['roc_auc']:+.4f}) | PR-AUC {a['pr_auc']:.4f} -> {b['pr_auc']:.4f} "
                      f"({b['pr_auc'] - a['pr_auc']:+.4f})")
    print(f"\nwritten: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
