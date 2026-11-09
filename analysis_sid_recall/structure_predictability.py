"""Can a semantic embedding predict an item's structural role in the behaviour graph?

This is the go/no-go question for per-item geometry gating: if semantics cannot
predict whether an item sits in a dense pocket or on a branching branch, a gated
Euclidean/hyperbolic/spherical distance has nothing trustworthy to gate on.

Targets, all computed from TRAIN behaviour edges only:
    clustering      local clustering coefficient (degree < 2 counts as 0)
    log_degree      log(1 + degree)
    af3_mean        mean Forman-Ricci value of the item's incident edges
    tri_free_share  share of the item's edges that sit in no triangle

Predictors, both fitted on the same five folds:
    semantic        the 768-d stage-1 sentence embedding
    degree_only     the item's own degree (a diagnostic ceiling, not deployable)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
EMBEDDINGS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
ITEM_IDS = REPO / "stage1_GeneEmbedding" / "output" / "item_ids.json"
OUT = Path(__file__).resolve().parent / "structure_predictability.json"
FOLDS = 5
RIDGE_ALPHA = 10.0


def behaviour(pairs: np.ndarray, n_items: int):
    """Structural features scattered into item-id space (0..n_items-1)."""
    nodes, inverse = np.unique(pairs, return_inverse=True)
    n = len(nodes)
    index = inverse.reshape(-1, 2)
    A = sparse.coo_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])),
                          shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    S = A @ A
    edge_tri = np.asarray(S[index[:, 0], index[:, 1]]).ravel()
    af3 = 4.0 - deg[index[:, 0]] - deg[index[:, 1]] + 3.0 * edge_tri
    node_tri = np.asarray(S.multiply(A).sum(1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        clustering = np.where(deg >= 2, node_tri / (deg * (deg - 1.0)), 0.0)
    af3_sum = np.zeros(n)
    tri_free = np.zeros(n)
    edge_count = np.zeros(n)
    for (u, v), value, tri in zip(index, af3, edge_tri):
        for node in (u, v):
            af3_sum[node] += value
            edge_count[node] += 1
            tri_free[node] += float(tri == 0)
    def scatter(values):
        out = np.zeros(n_items)
        out[nodes] = values
        return out
    return (scatter(deg), scatter(clustering),
            scatter(af3_sum / np.maximum(edge_count, 1)),
            scatter(tri_free / np.maximum(edge_count, 1)))


def folds_for(n: int):
    return list(KFold(n_splits=FOLDS, shuffle=True, random_state=42).split(np.arange(n)))


def evaluate(features: np.ndarray, target: np.ndarray, folds) -> dict:
    rho, r2 = [], []
    for train_idx, test_idx in folds:
        mean = features[train_idx].mean(0, keepdims=True)
        std = features[train_idx].std(0, keepdims=True) + 1e-9
        model = Ridge(alpha=RIDGE_ALPHA).fit((features[train_idx] - mean) / std, target[train_idx])
        pred = model.predict((features[test_idx] - mean) / std)
        rho.append(float(spearmanr(pred, target[test_idx]).statistic))
        residual = ((target[test_idx] - pred) ** 2).sum()
        total = ((target[test_idx] - target[train_idx].mean()) ** 2).sum()
        r2.append(float(1.0 - residual / total))
    return {"spearman_mean": float(np.mean(rho)), "spearman_sd": float(np.std(rho)),
            "r2_mean": float(np.mean(r2))}


def knn_predict(features: np.ndarray, target: np.ndarray, folds, k: int = 20) -> dict:
    """Does an item's semantic neighbourhood share its structural role?"""
    from sklearn.neighbors import NearestNeighbors

    rng = []
    for train_idx, test_idx in folds:
        model = NearestNeighbors(n_neighbors=k).fit(features[train_idx])
        _, index = model.kneighbors(features[test_idx])
        pred = target[train_idx][index].mean(axis=1)
        rng.append(float(spearmanr(pred, target[test_idx]).statistic))
        residual = ((target[test_idx] - pred) ** 2).sum()
        total = ((target[test_idx] - target[train_idx].mean()) ** 2).sum()
    return {"spearman_mean": float(np.mean(rng)), "spearman_sd": float(np.std(rng))}


def mlp_predict(features: np.ndarray, target: np.ndarray, folds) -> dict:
    """A small nonlinear head, to check the ridge result is not just underfitting."""
    import torch
    from torch import nn

    rng = []
    for train_idx, test_idx in folds:
        torch.manual_seed(0)
        mean = features[train_idx].mean(0, keepdims=True)
        std = features[train_idx].std(0, keepdims=True) + 1e-9
        x = torch.from_numpy(((features - mean) / std).astype(np.float32))
        y = torch.from_numpy(((target - target[train_idx].mean()) /
                              (target[train_idx].std() + 1e-9)).astype(np.float32)).view(-1, 1)
        model = nn.Sequential(nn.Linear(x.shape[1], 256), nn.GELU(), nn.Linear(256, 1))
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        tr = torch.from_numpy(train_idx)
        for _ in range(15):
            model.train()
            for chunk in torch.split(tr, 2048):
                optimiser.zero_grad()
                loss = nn.functional.mse_loss(model(x[chunk]), y[chunk])
                loss.backward()
                optimiser.step()
        model.eval()
        with torch.no_grad():
            pred = model(x[torch.from_numpy(test_idx)]).view(-1).numpy()
        rng.append(float(spearmanr(pred, target[test_idx]).statistic))
    return {"spearman_mean": float(np.mean(rng)), "spearman_sd": float(np.std(rng)),
            "epochs": 15}


def main() -> None:
    frame = pd.read_parquet(TRAIN)
    chunks = []
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        items = np.asarray([int(x) for x in history])
        chunks.append(np.stack([items[:-1], items[1:]], axis=1))
    pairs = np.unique(np.sort(np.concatenate(chunks), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    embeddings = np.load(EMBEDDINGS)
    item_ids = json.loads(ITEM_IDS.read_text())
    if isinstance(item_ids, dict):
        order = [int(k) for k in sorted(item_ids, key=lambda s: int(s))]
        position = {int(v): i for i, v in item_ids.items()}
    else:
        order = [int(x) for x in item_ids]
        position = {int(v): i for i, v in enumerate(item_ids)}
    n = len(order)

    deg, clustering, af3_mean, tri_free = behaviour(pairs, n)
    folds = folds_for(n)
    targets = {
        "clustering": clustering,
        "log_degree": np.log1p(deg),
        "af3_mean": af3_mean,
        "tri_free_share": tri_free,
    }
    report = {"items": int(n), "folds": FOLDS, "ridge_alpha": RIDGE_ALPHA, "targets": {}}
    for name, target in targets.items():
        semantic = evaluate(embeddings, target, folds)
        degree_only = evaluate(deg.reshape(-1, 1).astype(float), target, folds)
        knn = knn_predict(embeddings, target, folds)
        report["targets"][name] = {"semantic_ridge": semantic, "degree_only": degree_only,
                                   "semantic_knn": knn}
        print("%-15s ridge rho %.3f (r2 %.3f) | knn rho %.3f | degree-only rho %.3f"
              % (name, semantic["spearman_mean"], semantic["r2_mean"],
                 knn["spearman_mean"], degree_only["spearman_mean"]))
    for name in ("clustering", "af3_mean"):
        mlp = mlp_predict(embeddings, targets[name], folds)
        report["targets"][name]["semantic_mlp"] = mlp
        print("%-15s mlp rho %.3f" % (name, mlp["spearman_mean"]))
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
