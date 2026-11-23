"""Independent teacher evaluation for issue #244 §4.

Unseen pairs are eval-user transitions with **neither direction** present in the visible
graph.  The target ``g`` is the unweighted shortest path in the visible graph's undirected
projection.  Controls are non-trained: raw 768-dim semantic distance and the real Jaccard
overlap of undirected neighbourhoods (production ``metrics``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csgraph

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402
import topology  # noqa: E402
from teacher import CURVATURES, DEVICE, MixedGeometryTeacher, bfs_distances, load_all  # noqa: E402

GRAPH = HERE.parent / "issue242" / "graph"
OUT = HERE / "teacher"
SEED = 243101
MAX_PAIRS = 3000


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

    undirected = (out_csr + out_csr.T).tocsr()
    undirected.data[:] = 1
    coo = out_csr.tocoo()
    visible = {(int(a), int(b)) for a, b in zip(coo.row, coo.col)}
    eval_edges = np.load(GRAPH / "eval_edges.npy")
    unseen = np.array([(int(a), int(b)) for a, b in eval_edges
                       if (int(a), int(b)) not in visible and (int(b), int(a)) not in visible])
    print("unseen eval edges (both directions excluded):", len(unseen), flush=True)

    roots = np.unique(unseen[:, 0])
    if len(roots) > MAX_PAIRS // 2:
        roots = rng.choice(roots, size=MAX_PAIRS // 2, replace=False)
    keep = set(int(r) for r in roots)
    by_root = {}
    for a, b in unseen:
        if int(a) in keep:
            by_root.setdefault(int(a), []).append(int(b))
    roots_eval = np.array(sorted(by_root), dtype=np.int64)
    distance_matrix = csgraph.shortest_path(undirected, method="D", unweighted=True, indices=roots_eval)
    position = {int(r): k for k, r in enumerate(roots_eval)}

    pairs = []
    for root, partners in by_root.items():
        partner = int(partners[rng.integers(0, len(partners))])
        gap = distance_matrix[position[root], partner]
        if np.isfinite(gap) and gap > 0:
            pairs.append((root, partner, int(gap)))
    pairs = pairs[:MAX_PAIRS]
    print("pairs with finite visible distance:", len(pairs), flush=True)
    if not pairs:
        raise SystemExit("no evaluable pairs")

    index_i = torch.tensor([p[0] for p in pairs], device=DEVICE)
    index_j = torch.tensor([p[1] for p in pairs], device=DEVICE)
    gap = np.array([p[2] for p in pairs], dtype=np.float64)

    unique = sorted({x for p in pairs for x in (p[0], p[1])})
    local = {node: k for k, node in enumerate(unique)}
    descriptors = model.descriptor.encode_two_scales(unique, out_csr, in_csr, rng, embeddings, structural)
    with torch.no_grad():
        gate = model.weights_for(descriptors)
        expert_emb = model.embeddings(torch.cat([semantic, structural_t], dim=1),
                                      __import__("teacher").edge_index(out_csr))
        li = torch.tensor([local[int(x)] for x in index_i.tolist()], device=DEVICE)
        lj = torch.tensor([local[int(x)] for x in index_j.tolist()], device=DEVICE)
        d_geo, per_expert, pair_weight = model.distance(expert_emb, gate[li], gate[lj], index_i, index_j)
    d_geo = d_geo.detach().cpu().numpy().astype(np.float64)
    per_expert = per_expert.detach().cpu().numpy()

    semantic_gap = np.linalg.norm(embeddings[index_i.cpu().numpy()] - embeddings[index_j.cpu().numpy()], axis=1)
    jaccard = np.array([metrics.neighbour_jaccard(out_csr, in_csr, int(a), int(b))
                        for a, b in zip(index_i.cpu().numpy(), index_j.cpu().numpy())])
    distortion = np.abs((d_geo / gap) ** 2 - 1.0)
    degree = np.diff(out_csr.indptr)[index_i.cpu().numpy()]
    quintile = np.digitize(degree, np.quantile(degree, [0.2, 0.4, 0.6, 0.8]))

    report = {
        "pairs": int(len(pairs)),
        "unseen_eval_edges": int(len(unseen)),
        "distance_buckets": {str(k): int(v) for k, v in sorted(zip(*np.unique(gap, return_counts=True)))},
        "overall": {
            "d_geo_mean": float(d_geo.mean()),
            "gap_mean": float(gap.mean()),
            "distortion_mean": float(distortion.mean()),
            "distortion_median": float(np.median(distortion)),
            "spearman_d_geo_gap": metrics.spearman(d_geo, gap),
            "spearman_semantic_gap": metrics.spearman(semantic_gap, gap),
            "spearman_jaccard_gap": metrics.spearman(-jaccard, gap),
            "spearman_d_geo_semantic": metrics.spearman(d_geo, semantic_gap),
        },
        "by_distance": {},
        "by_degree_quintile": {},
        "expert_usage": {
            "curvatures": list(CURVATURES),
            "pair_weight_mean": pair_weight.detach().cpu().numpy().mean(0).tolist(),
            "per_expert_distance_mean": per_expert.mean(0).tolist(),
            "per_expert_distance_std": per_expert.std(0).tolist(),
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
        }
    for q in range(5):
        mask = quintile == q
        if mask.sum() < 5:
            continue
        report["by_degree_quintile"][str(q)] = {
            "pairs": int(mask.sum()),
            "distortion_mean": float(distortion[mask].mean()),
            "spearman_d_geo_gap": metrics.spearman(d_geo[mask], gap[mask]),
        }

    root = int(index_i[0])
    one = topology.ego_induced(root, out_csr, in_csr, rng, radius=1)
    two = topology.ego_induced(root, out_csr, in_csr, rng, radius=2)
    one_flipped = {**one, "edges": one["edges"][:, ::-1].copy()}
    two_flipped = {**two, "edges": two["edges"][:, ::-1].copy()}
    with torch.no_grad():
        # both scales, so the descriptor is the 128-dim vector the gate consumes
        t_a = torch.cat([model.descriptor.encode([one], embeddings, structural),
                         model.descriptor.encode([two], embeddings, structural)], dim=1)
        t_b = torch.cat([model.descriptor.encode([one_flipped], embeddings, structural),
                         model.descriptor.encode([two_flipped], embeddings, structural)], dim=1)
        w_a = model.weights_for(t_a)[0]
        w_b = model.weights_for(t_b)[0]
    report["direction_sensitivity"] = {
        "root": int(index_i[0]),
        "t_shape": list(t_a.shape),
        "t_l2_drift": float((t_a - t_b).norm()),
        "t_cosine": float(torch.nn.functional.cosine_similarity(t_a, t_b, dim=1)),
        "gate_l1_drift": float((w_a - w_b).abs().sum()),
    }
    verdict = {
        "beats_semantic": (report["overall"]["spearman_d_geo_gap"] is not None
                           and report["overall"]["spearman_semantic_gap"] is not None
                           and abs(report["overall"]["spearman_d_geo_gap"]) >
                               abs(report["overall"]["spearman_semantic_gap"])),
        "beats_jaccard": (report["overall"]["spearman_d_geo_gap"] is not None
                          and report["overall"]["spearman_jaccard_gap"] is not None
                          and abs(report["overall"]["spearman_d_geo_gap"]) >
                              abs(report["overall"]["spearman_jaccard_gap"])),
    }
    report["verdict"] = verdict
    (OUT / "teacher_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pairs", "overall", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
