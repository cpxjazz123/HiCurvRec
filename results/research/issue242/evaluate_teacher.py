"""Issue #242 step 5: does the mixed geometry carry real structure on unseen pairs?

Evaluation pairs are eval users' directed edges that the visible graph does not contain,
so neither the gate nor the experts ever saw them.  The target ``g`` is the unweighted
shortest path in the *visible* graph's undirected projection -- the same definition the
teacher was trained against, but on pairs it was never sampled with.

Controls, both non-trained:

    semantic   euclidean distance between the raw 768-dim T5 embeddings
    jaccard    1 / (1 + |N(i) ^ N(j)| / |N(i) | N(j)|) on the undirected visible graph

A teacher that cannot beat both controls on unseen pairs has learned a global scale, not
structure.  Reported per distance bucket and per degree quintile, as the issue asks.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csgraph

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from teacher import (CURVATURES, DEVICE, MixedGeometryTeacher, bfs_distances, ego_sample,
                     load_all)  # noqa: E402

GRAPH = HERE / "graph"
OUT = HERE / "teacher"
SEED = 242003
MAX_PAIRS = 4000


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def main() -> int:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    out_csr, in_csr, embeddings, structural = load_all()
    n = embeddings.shape[0]
    blob = torch.load(OUT / "teacher.pt", map_location="cpu")
    model = MixedGeometryTeacher(embeddings.shape[1], structural.shape[1]).to(DEVICE)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    semantic = torch.tensor(embeddings, device=DEVICE)
    structural_t = torch.tensor(structural, device=DEVICE)
    from teacher import edge_index
    edges = edge_index(out_csr)

    undirected = (out_csr + out_csr.T).tocsr()
    undirected.data[:] = 1
    visible_pairs = {(int(a), int(b)) for a, b in
                     zip(out_csr.tocoo().row, out_csr.tocoo().col)}
    eval_edges = np.load(GRAPH / "eval_edges.npy")
    unseen = np.array([(int(a), int(b)) for a, b in eval_edges
                       if (int(a), int(b)) not in visible_pairs])
    print("unseen eval edges", len(unseen), flush=True)

    # sample roots and one unseen partner each
    roots = np.unique(unseen[:, 0])
    if len(roots) > MAX_PAIRS // 2:
        roots = rng.choice(roots, size=MAX_PAIRS // 2, replace=False)
    by_root = {}
    for a, b in unseen:
        if int(a) in set(int(r) for r in roots):
            by_root.setdefault(int(a), []).append(int(b))

    pairs = []
    for root, partners in by_root.items():
        partner = int(partners[rng.integers(0, len(partners))])
        pairs.append((int(root), partner, -1))
    pairs = pairs[:MAX_PAIRS]
    # one C-level BFS for every root; the Python loop was ~0.2 s per root
    roots_eval = np.unique([p[0] for p in pairs])
    distance_matrix = csgraph.shortest_path(
        undirected, method="D", unweighted=True, indices=roots_eval)
    position = {int(r): k for k, r in enumerate(roots_eval)}
    resolved = []
    for root, partner, _ in pairs:
        gap = distance_matrix[position[int(root)], int(partner)]
        if np.isfinite(gap) and gap > 0:
            resolved.append((root, partner, int(gap)))
    pairs = resolved
    print("pairs with finite visible-graph distance", len(pairs), flush=True)
    if not pairs:
        raise SystemExit("no evaluable pairs")

    index_i = torch.tensor([p[0] for p in pairs], device=DEVICE)
    index_j = torch.tensor([p[1] for p in pairs], device=DEVICE)
    gap = np.array([p[2] for p in pairs], dtype=np.float64)

    unique = sorted({x for p in pairs for x in (p[0], p[1])})
    position = {node: k for k, node in enumerate(unique)}
    descriptors = model.descriptor.encode_two_scales(
        unique, out_csr, in_csr, rng, embeddings, structural)
    with torch.no_grad():
        gate = model.weights_for(descriptors)
        expert_emb = model.embeddings(torch.cat([semantic, structural_t], dim=1), edges)
        local_i = torch.tensor([position[int(x)] for x in index_i.tolist()], device=DEVICE)
        local_j = torch.tensor([position[int(x)] for x in index_j.tolist()], device=DEVICE)
        d_geo, per_expert, pair_w = model.distance(
            expert_emb, gate[local_i], gate[local_j], index_i, index_j)
    d_geo = d_geo.detach().cpu().numpy().astype(np.float64)
    per_expert = per_expert.detach().cpu().numpy()

    semantic_gap = np.linalg.norm(
        embeddings[index_i.cpu().numpy()] - embeddings[index_j.cpu().numpy()], axis=1)
    union = out_csr[index_i.cpu().numpy()] + out_csr[index_j.cpu().numpy()]
    jaccard = np.asarray(union.getnnz(axis=1)).ravel().astype(np.float64)

    distortion = np.abs((d_geo / gap) ** 2 - 1.0)
    degree_i = np.diff(out_csr.indptr)[index_i.cpu().numpy()]
    quintile = np.digitize(degree_i, np.quantile(degree_i, [0.2, 0.4, 0.6, 0.8]))

    report = {
        "pairs": int(len(pairs)),
        "unseen_eval_edges": int(len(unseen)),
        "distance_buckets": dict(sorted(Counter(gap.tolist()).items())),
        "overall": {
            "d_geo_mean": float(d_geo.mean()),
            "d_geo_median": float(np.median(d_geo)),
            "gap_mean": float(gap.mean()),
            "distortion_mean": float(distortion.mean()),
            "distortion_median": float(np.median(distortion)),
            "spearman_d_geo_gap": spearman(d_geo, gap),
            "spearman_semantic_gap": spearman(semantic_gap, gap),
            "spearman_jaccard_gap": spearman(jaccard, gap),
            "spearman_d_geo_semantic": spearman(d_geo, semantic_gap),
        },
        "by_distance": {},
        "by_degree_quintile": {},
        "expert_usage": {
            "curvatures": list(CURVATURES),
            "pair_weight_mean": pair_w.detach().cpu().numpy().mean(0).tolist(),
            "per_expert_distance_mean": per_expert.mean(0).tolist(),
        },
    }
    for bucket in sorted(set(gap.tolist())):
        mask = gap == bucket
        if mask.sum() < 5:
            continue
        report["by_distance"][str(int(bucket))] = {
            "pairs": int(mask.sum()),
            "d_geo_mean": float(d_geo[mask].mean()),
            "distortion_mean": float(distortion[mask].mean()),
            "spearman_within": spearman(d_geo[mask], semantic_gap[mask]) if mask.sum() > 5 else None,
        }
    for q in range(5):
        mask = quintile == q
        if mask.sum() < 5:
            continue
        report["by_degree_quintile"][str(q)] = {
            "pairs": int(mask.sum()),
            "distortion_mean": float(distortion[mask].mean()),
            "spearman_d_geo_gap": spearman(d_geo[mask], gap[mask]),
        }

    # direction sensitivity: flip the root's edges in one ego subgraph and measure T/W drift
    sample_root = int(index_i[0])
    base = ego_sample(sample_root, out_csr, in_csr, rng, radius=2)
    flipped = {**base, "edges": base["edges"][:, ::-1].copy()}
    with torch.no_grad():
        t_base = model.descriptor.encode([base], embeddings, structural)
        t_flip = model.descriptor.encode([flipped], embeddings, structural)
        w_base = model.weights_for(t_base)[0]
        w_flip = model.weights_for(t_flip)[0]
    report["direction_sensitivity"] = {
        "root": sample_root,
        "t_l2_drift": float((t_base - t_flip).norm()),
        "t_cosine": float(torch.nn.functional.cosine_similarity(t_base, t_flip, dim=1)),
        "gate_l1_drift": float((w_base - w_flip).abs().sum()),
    }

    (OUT / "teacher_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
