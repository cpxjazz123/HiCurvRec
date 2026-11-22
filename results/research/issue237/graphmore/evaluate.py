"""Evaluation of the enhanced 768-dim embedding against the original TIGER vectors.

The issue asks for the *final 768-dim* vectors to be scored, not the native mixed-space
decoder (C22), so every link-prediction number below is computed from a plain Euclidean
distance in the exported space.  The native mixed-space score is reported separately and
labelled as such.

AUC and AP depend only on the ordering of scores, and the Fermi-Dirac decoder is monotone
in the distance, so no decoder calibration is needed for a fair comparison between the two
representations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import data as data_module
from .train import auroc_ap

NEAR_ZERO = 1e-9
KNN_K = 10
KNN_SAMPLE = 2000
CHUNK = 2048


def _unit(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def pair_distances(embedding: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    left = embedding[pairs[:, 0]]
    right = embedding[pairs[:, 1]]
    return np.linalg.norm(left - right, axis=1)


def link_prediction(embedding: np.ndarray, edges: np.ndarray,
                    negatives: np.ndarray) -> dict:
    """AUC/AP from final-space distances; a closer pair is predicted to be an edge."""
    pos = pair_distances(embedding, edges)
    neg = pair_distances(embedding, negatives)
    # score = -distance, so AUC is computed on the ranking the distance induces
    scores = np.concatenate([-pos, -neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    auc, ap = auroc_ap(scores, labels)
    return {"auc": auc, "ap": ap, "pos_distance_mean": float(pos.mean()),
            "neg_distance_mean": float(neg.mean())}


def semantic_fidelity(enhanced: np.ndarray, original: np.ndarray) -> dict:
    """Cosine, ridge reconstruction R^2 and kNN-neighbourhood overlap."""
    cosine = np.sum(_unit(enhanced) * _unit(original), axis=1)

    # ridge from enhanced -> original, closed form with a small ridge term
    x = enhanced.astype(np.float64)
    y = original.astype(np.float64)
    gram = x.T @ x + 1e-3 * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, x.T @ y)
    predicted = x @ weights
    residual = float(((y - predicted) ** 2).sum())
    total = float(((y - y.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - residual / total

    rng = np.random.default_rng(42)
    sample = rng.choice(enhanced.shape[0], size=min(KNN_SAMPLE, enhanced.shape[0]),
                        replace=False)
    overlap = _knn_overlap(enhanced, original, sample, KNN_K)
    return {"cosine_mean": float(cosine.mean()), "cosine_median": float(np.median(cosine)),
            "ridge_r2": float(r2), "knn_overlap_at_%d" % KNN_K: overlap}


def _knn_overlap(left: np.ndarray, right: np.ndarray, sample: np.ndarray, k: int) -> float:
    left_unit = _unit(left)
    right_unit = _unit(right)
    overlaps = []
    for start in range(0, sample.size, 256):
        block = sample[start:start + 256]
        left_scores = left_unit[block] @ left_unit.T
        right_scores = right_unit[block] @ right_unit.T
        np.put_along_axis(left_scores, block[:, None], -np.inf, axis=1)
        np.put_along_axis(right_scores, block[:, None], -np.inf, axis=1)
        left_top = np.argpartition(-left_scores, k, axis=1)[:, :k]
        right_top = np.argpartition(-right_scores, k, axis=1)[:, :k]
        for row in range(block.size):
            overlaps.append(len(set(left_top[row].tolist()) & set(right_top[row].tolist())) / k)
    return float(np.mean(overlaps))


def distance_distortion(embedding: np.ndarray, pairs: np.ndarray,
                        graph_distance: np.ndarray) -> dict:
    """Spearman correlation and ratio between embedding and graph distance."""
    mask = np.isfinite(graph_distance) & (graph_distance > 0)
    embedded = pair_distances(embedding, pairs)[mask]
    target = graph_distance[mask]
    rank = lambda values: np.argsort(np.argsort(values)).astype(float)
    a, b = rank(embedded), rank(target)
    a -= a.mean()
    b -= b.mean()
    spearman = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    return {"spearman": spearman, "pairs": int(mask.sum()),
            "mean_ratio": float((embedded / target).mean())}


def graphlet_fidelity(embedding: np.ndarray, banks_dir: Path,
                      split: str = "validation") -> dict:
    """Structure retention on the isolated Graphlet banks.

    The banks hold directed role triples ``(A, B, C)``; on the undirected projection the
    relations become distance orderings that a continuous embedding can be tested on:

    * chain  -- ``A-B-C`` with ``B`` between them: expects ``d(A,B) = d(B,C) = 1`` and
      ``d(A,C) = 2``, so the embedding should place ``A`` and ``C`` furthest apart;
    * fork   -- ``A`` is the centre adjacent to both ``B`` and ``C``: expects ``B`` and
      ``C`` furthest apart;
    * merge  -- the undirected projection of a merge is identical to a fork, so the two
      share one formula and **cannot be told apart** here (stated in the report).

    The reported number is the fraction of instances where the expected ordering holds,
    plus the mean normalised margin.
    """
    out = {}
    for kind in ("chain", "fork", "merge"):
        path = banks_dir / f"{kind}_{split}.npy"
        if not path.is_file():
            out[kind] = {"note": "bank missing"}
            continue
        triples = np.load(path)
        a, b, c = triples[:, 0], triples[:, 1], triples[:, 2]
        if kind == "chain":
            far, mid1, mid2 = (a, c), (a, b), (b, c)
        else:
            far, mid1, mid2 = (b, c), (a, b), (a, c)
        d_far = np.linalg.norm(embedding[far[0]] - embedding[far[1]], axis=1)
        d_1 = np.linalg.norm(embedding[mid1[0]] - embedding[mid1[1]], axis=1)
        d_2 = np.linalg.norm(embedding[mid2[0]] - embedding[mid2[1]], axis=1)
        expected = d_far > np.maximum(d_1, d_2)
        margin = (d_far - np.maximum(d_1, d_2)) / np.maximum(d_far, 1e-12)
        out[kind] = {"instances": int(triples.shape[0]),
                     "ordering_rate": float(expected.mean()),
                     "mean_margin": float(margin.mean())}
    return out


def native_score(model, adjacency, egos, x, edges, negatives, device) -> dict:
    """The paper's own weight-aligned distance, kept apart from the exported space (C22)."""
    import torch
    with torch.no_grad():
        embedding, weights = model(x, adjacency, egos)
        pos = model.align(embedding, weights,
                          torch.as_tensor(edges[:, 0], device=device),
                          torch.as_tensor(edges[:, 1], device=device)).cpu().numpy()
        neg = model.align(embedding, weights,
                          torch.as_tensor(negatives[:, 0], device=device),
                          torch.as_tensor(negatives[:, 1], device=device)).cpu().numpy()
    scores = np.concatenate([-pos, -neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    auc, ap = auroc_ap(scores, labels)
    return {"auc": auc, "ap": ap}


def expert_report(weights: np.ndarray, degree: np.ndarray,
                  curvature_csv: Path) -> dict:
    """Weight mass per expert, gating entropy, degree and curvature strata (C25)."""
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
    usage = np.bincount(weights.argmax(axis=1), minlength=weights.shape[1])
    report = {
        "mean_weight_per_expert": weights.mean(axis=0).tolist(),
        "argmax_usage_fraction": (usage / weights.shape[0]).tolist(),
        "entropy_mean": float(entropy.mean()),
        "entropy_max": float(np.log(weights.shape[1])),
        "collapsed_experts": int((usage / weights.shape[0] < 0.01).sum()),
        "degree_strata": [],
        "curvature_strata": [],
    }
    edges = [1, 5, 13, 50, 200, 10 ** 9]
    labels = ["1-4", "5-12", "13-49", "50-199", "200+"]
    for low, high, label in zip(edges[:-1], edges[1:], labels):
        block = (degree >= low) & (degree < high)
        if not block.any():
            continue
        report["degree_strata"].append({
            "degree_range": label, "items": int(block.sum()),
            "mean_weight": weights[block].mean(axis=0).tolist(),
            "entropy_mean": float(entropy[block].mean()),
        })
    if curvature_csv.is_file():
        curvature = {}
        for line in curvature_csv.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split(",")
            if parts[3]:
                curvature[int(parts[0])] = float(parts[3])
        values = np.array([curvature.get(item, np.nan) for item in range(weights.shape[0])])
        valid = np.isfinite(values)
        for label, block in (
                ("negative", valid & (values < -NEAR_ZERO)),
                ("near_zero", valid & (np.abs(values) <= NEAR_ZERO)),
                ("positive", valid & (values > NEAR_ZERO))):
            if not block.any():
                continue
            report["curvature_strata"].append({
                "curvature_sign": label, "items": int(block.sum()),
                "mean_weight": weights[block].mean(axis=0).tolist(),
                "entropy_mean": float(entropy[block].mean()),
            })
        report["curvature_source"] = str(curvature_csv.name)
    return report


def evaluate(directory: Path | None = None, device_name: str = "cuda") -> dict:
    here = Path(__file__).resolve().parents[1]
    directory = directory or here
    enhanced = np.load(directory / "enhanced_768.npy")
    weights = np.load(directory / "gating_weights.npy")
    bundle = data_module.build()
    original = bundle["embeddings"]
    degree = np.diff(bundle["train_adjacency"].indptr).astype(int)

    from .export import load_model
    import torch
    from .model import sparse_adjacency

    device = torch.device(device_name)
    model = load_model(directory / "best_checkpoint.pt", device)
    adjacency = sparse_adjacency(
        torch.as_tensor(bundle["split"]["train"].T.copy(), dtype=torch.long, device=device),
        bundle["n"], device)
    egos = [torch.sparse_coo_tensor(
                torch.as_tensor(np.stack(matrix.nonzero()), dtype=torch.long, device=device),
                torch.as_tensor(matrix.data, dtype=torch.float32, device=device),
                matrix.shape).coalesce()
            for matrix in (bundle["ego"][2], bundle["ego"][3])]
    x = torch.as_tensor(original, dtype=torch.float32, device=device)

    banks = data_module.REPO / "results" / "research" / "issue231" / "banks"
    curvature_csv = (data_module.REPO / "results" / "research" / "issue236"
                     / "node_curvature_exact_geodesic.csv")

    report = {
        "semantic_fidelity": semantic_fidelity(enhanced, original),
        "graph_structure": {
            "enhanced": {
                "val": link_prediction(enhanced, bundle["split"]["val"],
                                       bundle["negatives"]["val"]),
                "test": link_prediction(enhanced, bundle["split"]["test"],
                                        bundle["negatives"]["test"]),
                "distance_distortion": distance_distortion(
                    enhanced, bundle["distortion_pairs"], bundle["distortion_target"]),
            },
            "original": {
                "val": link_prediction(original, bundle["split"]["val"],
                                       bundle["negatives"]["val"]),
                "test": link_prediction(original, bundle["split"]["test"],
                                        bundle["negatives"]["test"]),
                "distance_distortion": distance_distortion(
                    original, bundle["distortion_pairs"], bundle["distortion_target"]),
            },
        },
        "graphlet_fidelity": {
            "enhanced": graphlet_fidelity(enhanced, banks),
            "original": graphlet_fidelity(original, banks),
        },
        "native_mixed_space": {
            "val": native_score(model, adjacency, egos, x, bundle["split"]["val"],
                                bundle["negatives"]["val"], device),
            "test": native_score(model, adjacency, egos, x, bundle["split"]["test"],
                                 bundle["negatives"]["test"], device),
            "note": "paper's weight-aligned distance; NOT the exported 768-dim vector",
        },
        "experts": expert_report(weights, degree, curvature_csv),
    }
    (directory / "evaluation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report
