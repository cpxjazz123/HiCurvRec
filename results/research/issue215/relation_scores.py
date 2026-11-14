"""Keep the relation model's scores for every behaviour edge and for hard local negatives.

The local triangle test needs the model's *ranking* inside a neighbourhood, not a thresholded graph.
This trains the same relation model as `relation_predictability.py` (same architecture, folds, seed
and budget) and saves, for every behaviour edge and for a set of hard local negatives, the model's
probability for that pair.

A local negative for the edge (u, v) is a pair (u, w) where w is a neighbour of u that is **not**
itself connected to u: an item one step away, of the kind a local model confuses with the real
thing.  The behaviour graph itself is only the answer key; nothing of it enters the model.

    python3 results/research/issue215/relation_scores.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
GRAPHS = OUT / "graphs"
SCORES = OUT / "scores"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
JOURNAL = OUT / "relation_scores.json"
ARMS = ("TIGER", "F0", "B", "C", "D")
SEED = 42
NEGATIVES_PER_ENDPOINT = 4


def behaviour_edges() -> np.ndarray:
    with (GRAPHS / "behaviour.in").open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    return edges


def local_negatives(edges: np.ndarray, adjacency: list[set[int]], n_items: int,
                    rng: np.random.Generator) -> np.ndarray:
    """For every edge, pairs (u, w) where w sits next to u without being connected to u."""
    out = []
    for u, v in edges.tolist():
        u, v = int(u), int(v)
        # a hard negative for (u, v): an item next to v that is NOT connected to u -- one step away
        # in the graph, so a local model can confuse it with the real thing
        for item, other in ((u, v), (v, u)):
            candidates = [w for w in adjacency[other] if w != item and w not in adjacency[item]]
            if not candidates:
                continue
            picked = rng.choice(len(candidates), size=min(NEGATIVES_PER_ENDPOINT, len(candidates)),
                                replace=False)
            for index in picked.tolist():
                w = candidates[index]
                out.append((min(item, w), max(item, w)))
    return np.asarray(sorted(set(out)), dtype=np.int64)


def score_pairs(model: torch.nn.Module, vectors: torch.Tensor, pairs: np.ndarray,
                device: torch.device) -> np.ndarray:
    weight = model[0].weight.detach()
    dimension = int(vectors.shape[1])
    first_difference = vectors @ weight[:, :dimension].T
    first_product = vectors @ weight[:, dimension:].T
    bias = model[0].bias.detach()
    left = torch.tensor(pairs[:, 0], dtype=torch.long, device=device)
    right = torch.tensor(pairs[:, 1], dtype=torch.long, device=device)
    scores = []
    with torch.no_grad():
        for start in range(0, len(pairs), 1_000_000):
            a = left[start:start + 1_000_000]
            b = right[start:start + 1_000_000]
            hidden = torch.relu(first_difference[a] - first_difference[b]
                                + first_product[a] * first_product[b] + bias)
            hidden = torch.relu(model[2](hidden))
            scores.append(torch.sigmoid(model[4](hidden)).ravel().cpu().numpy())
    return np.concatenate(scores)


def main() -> int:
    import sys
    sys.path.insert(0, str(OUT))
    from relation_graphs import train_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SCORES.mkdir(parents=True, exist_ok=True)
    edges = behaviour_edges()
    n_items = int(edges.max()) + 1
    adjacency: list[set[int]] = [set() for _ in range(n_items)]
    for a, b in edges.tolist():
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    rng = np.random.default_rng(SEED)
    negatives = local_negatives(edges, adjacency, n_items, rng)
    print(f"edges {len(edges)} | hard local negatives {len(negatives)}", flush=True)

    order = rng.permutation(len(edges))
    cut = int(0.8 * len(edges))
    train_positive = torch.tensor(edges[order[:cut]], dtype=torch.long, device=device)
    train_negative = torch.tensor(negatives[order[:cut]], dtype=torch.long, device=device)

    report = {"edges": int(len(edges)), "local_negatives": int(len(negatives)), "predictors": {}}

    def run(name: str, vectors: np.ndarray) -> None:
        tensor = torch.tensor(vectors, dtype=torch.float32, device=device)
        model = train_model(tensor, train_positive, train_negative, device)
        pairs = np.concatenate([edges, negatives], axis=0)
        score = score_pairs(model, tensor, pairs, device)
        path = SCORES / f"{name}_scores.npz"
        np.savez(path, pairs=pairs, score=score)
        report["predictors"][name] = {"pairs": int(len(pairs)), "file": str(path.relative_to(REPO))}
        print(f"{name:<16} scored {len(pairs)} pairs", flush=True)

    run("semantic", np.load(EMBEDDING).astype(np.float32))
    for arm in ARMS:
        z_path, q_path = VECTORS / f"{arm}_z.npy", VECTORS / f"{arm}_q.npy"
        if not z_path.is_file() or not q_path.is_file():
            continue
        run(f"{arm}_z", np.load(z_path).astype(np.float32))
        run(f"{arm}_q", np.load(q_path).astype(np.float32))

    JOURNAL.write_text(json.dumps(report, indent=1, default=float))
    print(f"written: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
