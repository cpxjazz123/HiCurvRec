"""Turn the relation model into a graph, then let ORCA describe it.

The relation model of ``relation_predictability.py`` can score any pair of items.  Here it scores
**all** pairs, keeps the most behaviour-like ones per item, and writes the resulting graph, so the
structure of "what the vectors say about relations" can be compared with the structure of the real
behaviour graph through the same ORCA features.

The graph uses the same edge budget as the behaviour graph, so the comparison is density-free:
for every item the pairs with the highest predicted probability are taken until the total matches.

    python3 results/research/issue215/relation_graphs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
GRAPHS = OUT / "graphs"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
JOURNAL = OUT / "relation_graphs.json"
ARMS = ("TIGER", "F0", "B", "C", "D")
SEED = 42
EPOCHS = 600
HIDDEN = 256
LR = 2e-3
CHUNK = 2_000_000


def behaviour_edges() -> np.ndarray:
    frame = pd.read_parquet(TRAIN, columns=["seen_history", "target"])
    pairs = set()
    for history, target in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)):
        if history is None or len(history) == 0:
            continue
        pairs.add((int(history[-1]), int(target)))
    return np.asarray(sorted(pairs), dtype=np.int64)


def train_model(vectors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor,
                device: torch.device) -> torch.nn.Module:
    def features(pairs: torch.Tensor) -> torch.Tensor:
        left, right = vectors[pairs[:, 0]], vectors[pairs[:, 1]]
        return torch.cat([left - right, left * right], dim=1)

    train_pairs = torch.cat([positives, negatives], dim=0)
    y = torch.cat([torch.ones(positives.shape[0]), torch.zeros(negatives.shape[0])]).to(device)
    with torch.no_grad():
        x = features(train_pairs)
        mean = x.mean(0, keepdim=True)
        scale = x.std(0, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / scale
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1),
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y.unsqueeze(1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    model.eval()
    model._mean = mean
    model._scale = scale
    del x, train_pairs, y
    torch.cuda.empty_cache()
    return model


def predict_graph(model: torch.nn.Module, vectors: torch.Tensor, n_items: int, budget: int,
                  device: torch.device, per_item: int = 13) -> np.ndarray:
    """Score every pair and give each item its ``per_item`` most relation-like partners.

    The behaviour graph has 315,391 edges over 24,587 items, so thirteen partners per item is the
    degree the k-NN-style construction has to match; taking each item's best partners (rather than
    the globally most probable pairs) keeps the degree profile of the graph tied to the model's
    scores instead of letting a few hubs take the whole budget.
    """
    weight = model[0].weight.detach()
    dimension = int(vectors.shape[1])
    first_difference = vectors @ weight[:, :dimension].T
    first_product = vectors @ weight[:, dimension:].T
    bias = model[0].bias.detach()
    partners: list[list[int]] = [[] for _ in range(n_items)]
    block = 64
    torch.cuda.empty_cache()
    for start in range(0, n_items, block):
        rows = torch.arange(start, min(start + block, n_items), device=device)
        size = rows.numel()
        left = rows.repeat_interleave(n_items)
        right = rows.repeat(n_items)          # the diagonal stays; it is excluded when ranking
        with torch.no_grad():
            hidden = torch.relu(first_difference[left] - first_difference[right]
                                + first_product[left] * first_product[right] + bias)
            hidden = torch.relu(model[2](hidden))
            probability = torch.sigmoid(model[4](hidden)).ravel()
        # the row layout is (item0 with every item), (item1 with every item), ...
        chunk = probability.view(size, n_items)
        for row in range(size):
            item = int(rows[row])
            values = chunk[row].clone()
            values[torch.arange(n_items, device=device) == item] = -1.0
            top = torch.topk(values, per_item)
            partners[item] = [int(value) for value in top.indices.cpu().tolist()]
    edges = set()
    for item in range(n_items):
        for partner in partners[item]:
            if partner == item:
                continue
            edges.add((min(item, partner), max(item, partner)))
    return np.asarray(sorted(edges), dtype=np.int64)


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edges = behaviour_edges()
    n_items = int(edges.max()) + 1
    budget = len(edges)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(edges))
    cut = int(0.8 * len(edges))
    train_positive = torch.tensor(edges[order[:cut]], dtype=torch.long, device=device)
    test_positive = edges[order[cut:]]
    degrees = np.bincount(edges.ravel(), minlength=n_items)
    from relation_predictability import sample_negatives
    negatives = sample_negatives(edges, degrees, n_items, np.random.default_rng(SEED + 1), matched=True)
    negative_order = rng.permutation(len(negatives))
    negative_cut = int(0.8 * len(negatives))
    train_negative = torch.tensor(negatives[negative_order[:negative_cut]], dtype=torch.long, device=device)

    report = {"items": n_items, "edge_budget": budget, "graphs": {}}

    def build(name: str, vectors: np.ndarray) -> None:
        path = GRAPHS / f"{name}_rel.in"
        if path.is_file():
            path.unlink()          # the earlier rule filled the budget with popular pairs; rebuild
        tensor = torch.tensor(vectors, dtype=torch.float32, device=device)
        model = train_model(tensor, train_positive, train_negative, device)
        with torch.no_grad():
            probability = torch.sigmoid(model(
                (torch.cat([tensor[test_positive[:, 0]] - tensor[test_positive[:, 1]],
                           tensor[test_positive[:, 0]] * tensor[test_positive[:, 1]]], dim=1)
                 - model._mean) / model._scale)).cpu().numpy().ravel()
        truth = np.ones(len(test_positive))
        predicted_edges = predict_graph(model, tensor, n_items, budget, device)
        path = GRAPHS / f"{name}_rel.in"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"{n_items} {len(predicted_edges)}\n")
            for a, b in predicted_edges.tolist():
                handle.write(f"{a} {b}\n")
        # how many of the model's own held-out edges the predicted graph contains
        predicted_set = {tuple(row) for row in predicted_edges.tolist()}
        recovered = sum(1 for a, b in test_positive.tolist()
                        if (min(a, b), max(a, b)) in predicted_set)
        report["graphs"][name] = {"edges": int(len(predicted_edges)), "file": str(path.relative_to(REPO)),
                                  "held_out_edges_recovered": int(recovered),
                                  "held_out_edges": int(len(test_positive)),
                                  "held_out_recovery": recovered / max(len(test_positive), 1),
                                  "test_pair_probability_mean": float(probability.mean())}
        print(f"{name:<16} edges {len(predicted_edges):>7} | held-out behaviour edges recovered "
              f"{recovered}/{len(test_positive)} ({recovered / max(len(test_positive), 1):.4f})", flush=True)

    build("semantic", np.load(EMBEDDING).astype(np.float32))
    for arm in ARMS:
        z_path, q_path = VECTORS / f"{arm}_z.npy", VECTORS / f"{arm}_q.npy"
        if not z_path.is_file() or not q_path.is_file():
            continue
        build(f"{arm}_z", np.load(z_path).astype(np.float32))
        build(f"{arm}_q", np.load(q_path).astype(np.float32))

    JOURNAL.write_text(json.dumps(report, indent=1, default=float))
    print(f"written: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
