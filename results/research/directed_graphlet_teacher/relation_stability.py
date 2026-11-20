"""Before any relational distillation: is the Teacher's node state a property of the item?

The Teacher reads the instance's own adjacency, so `h_T(A)` is a function of the
subgraph A sits in, not of A.  A fixed SID can only carry one vector per item, so
the question this script answers is whether that teacher state is stable enough
across instances to be a target at all:

  * within-item: cosine similarity between the teacher states of the same item in
    two different instances,
  * between-item: the same similarity for two different items inside one instance
    (the scale of "no relation" in this representation),
  * how much of the state a linear map from the 768-d item embedding explains
    (ridge regression R^2 on held-out rows).

If within-item is no better than between-item, a fixed SID cannot match the
target and the loss would be fitting noise; the honest response is to change the
target (only relations that are stable across instances), not to raise lambda.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))

from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import FAMILIES, load_pool, load_vocabulary  # noqa: E402
from teacher_model import load_frozen_teacher  # noqa: E402

CAP = 20000
CHUNK = 2048
MIN_OCCURRENCES = 3


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher, record = load_frozen_teacher(HERE / "teacher" / "teacher_directed_v1.pth")
    teacher.to(device).eval()
    semantic = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy",
                       mmap_mode="r")
    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    edges = set((rows.astype(np.int64) * n_items + cols).tolist())

    vocab = load_vocabulary("directed")
    families = load_pool("train", vocab, max_instances=CAP)

    states: dict[int, list[np.ndarray]] = {}
    embeddings: dict[int, np.ndarray] = {}
    for family in FAMILIES:
        nodes = families[family][0]
        width = nodes.shape[1]
        adjacency = np.zeros((len(nodes), width, width), dtype=np.float32)
        for i in range(width):
            for j in range(width):
                if i == j:
                    continue
                keys = nodes[:, i].astype(np.int64) * n_items + nodes[:, j]
                adjacency[:, i, j] = [int(key) in edges for key in keys]
        for start in range(0, len(nodes), CHUNK):
            block = nodes[start:start + CHUNK]
            index = torch.from_numpy(block).long().to(device)
            with torch.no_grad():
                batch_items = torch.from_numpy(
                    np.asarray(semantic[block.reshape(-1)], dtype=np.float32)).to(device)
                _, _, hidden = teacher(batch_items.view(len(block), width, -1),
                                       torch.from_numpy(adjacency[start:start + CHUNK]).to(device))
            hidden = hidden.detach().cpu().numpy()
            for row in range(len(block)):
                for column in range(width):
                    item = int(block[row, column])
                    states.setdefault(item, []).append(hidden[row, column])
                    if item not in embeddings:
                        embeddings[item] = np.asarray(semantic[item], dtype=np.float32)
        print(family, "done", flush=True)

    repeated = {item: vectors for item, vectors in states.items()
                if len(vectors) >= MIN_OCCURRENCES}
    within = []
    between = []
    rng = np.random.default_rng(224)
    items = sorted(repeated)
    for item in items:
        vectors = np.stack(repeated[item])
        normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        similarity = normed @ normed.T
        upper = similarity[np.triu_indices(len(vectors), 1)]
        if len(upper):
            within.append(float(upper.mean()))
    for _ in range(20000):
        left, right = rng.choice(items, 2, replace=False)
        a = repeated[left][rng.integers(len(repeated[left]))]
        b = repeated[right][rng.integers(len(repeated[right]))]
        between.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))

    probe = _ridge_probe(items, repeated, embeddings)
    payload = {
        "teacher": record.get("extra", {}),
        "instances": int(sum(len(v) for v in states.values())),
        "distinct_items": len(states),
        "items_with_at_least": MIN_OCCURRENCES,
        "repeated_items": len(repeated),
        "within_item_cosine": _summary(within),
        "between_item_cosine": _summary(between),
        "embedding_probe": probe,
        "verdict": ("stable" if np.mean(within) > np.mean(between) + 0.10 else "unstable"),
    }
    (HERE / "relation_stability.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1)[:1200])


def _summary(values: list[float]) -> dict:
    array = np.asarray(values)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "p10": float(np.quantile(array, 0.10)), "p90": float(np.quantile(array, 0.90)),
            "n": int(array.size)}


def _ridge_probe(items: list[int], repeated: dict[int, list[np.ndarray]],
                 embeddings: dict[int, np.ndarray]) -> dict:
    """R^2 of a linear map item-embedding -> teacher state, on held-out items."""
    features, targets = [], []
    for item in items:
        features.append(embeddings[item])
        targets.append(np.stack(repeated[item]).mean(0))
    features = np.stack(features).astype(np.float64)
    targets = np.stack(targets).astype(np.float64)
    split = int(len(features) * 0.8)
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = targets[:split], targets[split:]
    mean = train_x.mean(0)
    scale = train_x.std(0) + 1e-6
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    ridge = 1e-2
    gram = train_x.T @ train_x + ridge * np.eye(train_x.shape[1])
    weights = np.linalg.solve(gram, train_x.T @ train_y)
    prediction = test_x @ weights
    residual = ((test_y - prediction) ** 2).sum()
    total = ((test_y - test_y.mean(0)) ** 2).sum()
    return {"r2_heldout": float(1.0 - residual / total),
            "train_items": int(split), "test_items": int(len(features) - split),
            "target_dim": int(targets.shape[1])}


if __name__ == "__main__":
    main()
