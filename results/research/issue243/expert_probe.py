"""Per-expert probe for issue #244 §3 (C12).

The five experts' mean distances in #243 spanned [2.695, 0.224, 2.299, 0.061, 0.023] -- a
factor of ~117.  Balanced gate weights do not make balanced distance contributions, so the
calibration that follows needs the actual per-expert statistics first:

    distance median / p10 / p25 / p75 / p90 / std
    output norm min / median / max, and the boundary-saturation rate
    the gradient of the REAL L_D restricted to each expert's parameters
    the actual parameter displacement after one optimizer step

Nothing here changes the model; it only measures it.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from scipy.sparse import csgraph

HERE_PATH = __file__
import sys
from pathlib import Path

HERE = Path(HERE_PATH).resolve().parent
sys.path.insert(0, str(HERE))

import manifold  # noqa: E402
import teacher as T  # noqa: E402

OUT = HERE / "teacher"
PROBE_PAIRS = 2000
SEED = 243201
# calibration measures the INITIALISATION, because that is what it has to correct
USE_FRESH_MODEL = True


def main() -> int:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    out_csr, in_csr, embeddings, structural = T.load_all()
    n = embeddings.shape[0]
    torch.manual_seed(SEED)
    model = T.MixedGeometryTeacher(embeddings.shape[1], structural.shape[1]).to(T.DEVICE)
    if not USE_FRESH_MODEL:
        blob = torch.load(OUT / "teacher.pt", map_location="cpu")
        model.load_state_dict(blob["state_dict"])
    model.train()

    full_x = torch.cat([torch.tensor(embeddings, device=T.DEVICE),
                        torch.tensor(structural, device=T.DEVICE)], dim=1)
    edges = T.edge_index(out_csr)
    undirected = (out_csr + out_csr.T).tocsr()
    undirected.data[:] = 1

    # a fixed probe: roots spread over the degree quintiles, hop drawn uniformly
    degree = np.diff(out_csr.indptr)
    active = np.where(degree > 0)[0]
    quintile_edges = np.quantile(degree[active], [0.2, 0.4, 0.6, 0.8])
    strata = [active[np.digitize(degree[active], quintile_edges) == q] for q in range(5)]
    roots = np.concatenate([rng.choice(s, size=max(1, PROBE_PAIRS // 10), replace=False) for s in strata])
    distance_matrix = csgraph.shortest_path(undirected, method="D", unweighted=True, indices=roots)
    pairs = []
    for row, root in enumerate(roots):
        hops = {}
        for node, hop in enumerate(distance_matrix[row]):
            if 1 <= hop <= 4:
                hops.setdefault(int(hop), []).append(int(node))
        if not hops:
            continue
        keys = sorted(hops)
        hop = int(keys[rng.integers(0, len(keys))])
        pairs.append((int(root), int(hops[hop][rng.integers(0, len(hops[hop]))]), hop))
    pairs = pairs[:PROBE_PAIRS]
    index_i = torch.tensor([p[0] for p in pairs], device=T.DEVICE)
    index_j = torch.tensor([p[1] for p in pairs], device=T.DEVICE)
    gap = torch.tensor([float(p[2]) for p in pairs], device=T.DEVICE)

    unique = sorted({x for p in pairs for x in (p[0], p[1])})
    descriptors = model.descriptor.encode_two_scales(unique, out_csr, in_csr, rng, embeddings, structural)
    gate = model.weights_for(descriptors)
    position = {node: k for k, node in enumerate(unique)}
    local_i = torch.tensor([position[int(x)] for x in index_i.tolist()], device=T.DEVICE)
    local_j = torch.tensor([position[int(x)] for x in index_j.tolist()], device=T.DEVICE)

    expert_emb = model.embeddings(full_x, edges)
    d_geo, per_expert, pair_weight = model.distance(expert_emb, gate[local_i], gate[local_j],
                                                    index_i, index_j)
    loss = ((d_geo / gap) ** 2 - 1.0).abs().mean()

    report = {
        "pairs": int(len(pairs)),
        "loss": float(loss.detach()),
        "experts": [],
        "pair_weight_mean": pair_weight.detach().cpu().numpy().mean(0).tolist(),
    }
    # per-expert distances and saturation, straight from the full-graph embeddings
    for index, (curvature, embedding) in enumerate(zip(T.CURVATURES, expert_emb)):
        norms = embedding.detach().norm(dim=-1)
        if curvature < 0:
            radius = 1.0 / math.sqrt(abs(curvature))
            saturation = float((norms > 0.999 * radius).float().mean())
        elif curvature > 0:
            radius = 1.0 / math.sqrt(curvature)
            saturation = float((norms > 50.0 * radius).float().mean())
        else:
            radius = None
            saturation = 0.0
        d = per_expert.detach().cpu().numpy()[:, index]
        report["experts"].append({
            "index": index,
            "curvature": curvature,
            "radius": radius,
            "distance_median": float(np.median(d)),
            "distance_p10": float(np.quantile(d, 0.10)),
            "distance_p25": float(np.quantile(d, 0.25)),
            "distance_p75": float(np.quantile(d, 0.75)),
            "distance_p90": float(np.quantile(d, 0.90)),
            "distance_std": float(d.std()),
            "norm_min": float(norms.min()),
            "norm_median": float(norms.median()),
            "norm_max": float(norms.max()),
            "saturation_rate": saturation,
        })

    # gradient of the real L_D restricted to each expert
    for index, expert in enumerate(model.experts):
        grads = torch.autograd.grad(loss, list(expert.parameters()), retain_graph=True,
                                    allow_unused=True)
        total = sum(float(g.pow(2).sum()) for g in grads if g is not None) ** 0.5
        none_count = sum(1 for g in grads if g is None)
        report["experts"][index]["grad_l2"] = total
        report["experts"][index]["grad_none_params"] = none_count

    # one real optimizer step: actual displacement per expert
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [[p.detach().clone() for p in expert.parameters()] for expert in model.experts]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for index, expert in enumerate(model.experts):
        delta = sum(float((p.detach() - b).norm() ** 2) for p, b in zip(expert.parameters(), before[index])) ** 0.5
        report["experts"][index]["adam_delta"] = delta

    distances = [e["distance_median"] for e in report["experts"] if e["distance_median"] > 0]
    report["scale_spread"] = (max(distances) / min(distances)) if len(distances) > 1 else None
    (OUT / "expert_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for entry in report["experts"]:
        print(f"k={entry['curvature']:+.0f} median={entry['distance_median']:.4f} "
              f"std={entry['distance_std']:.4f} sat={entry['saturation_rate']:.4f} "
              f"grad={entry['grad_l2']:.4e} delta={entry['adam_delta']:.4e}")
    print("scale spread:", report["scale_spread"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
