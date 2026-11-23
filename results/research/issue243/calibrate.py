"""Issue #244 §3: freeze and solve the per-expert distance scale.

Rule, frozen before any formal training (recorded in CONTRACT §8):

* target  = the median of the five experts' probe median distances on the initialisation;
* action  = a legal tangent-space rescaling per expert, z -> exp(alpha * log(z)), which keeps
            every point inside the same manifold;
* solver  = bisection on alpha in [0.05, 20] so the expert's probe median distance reaches the
            target; if the target is unreachable (a sphere's maximum geodesic distance is
            finite) the bound is recorded rather than silently accepted;
* the probe pairs and the seed are fixed here and reused by expert_probe.py.

The gate formula is untouched; this only changes the units each expert's distance is measured
in, which is exactly what a comparison across curvatures needs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csgraph

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manifold  # noqa: E402
import teacher as T  # noqa: E402

PROBE_PAIRS = 2000
SEED = 243201
ALPHA_BOUNDS = (0.05, 20.0)
BISECTION_STEPS = 40


def build_probe():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    out_csr, in_csr, embeddings, structural = T.load_all()
    degree = np.diff(out_csr.indptr)
    active = np.where(degree > 0)[0]
    edges = np.quantile(degree[active], [0.2, 0.4, 0.6, 0.8])
    strata = [active[np.digitize(degree[active], edges) == q] for q in range(5)]
    roots = np.concatenate([rng.choice(s, size=max(1, PROBE_PAIRS // 10), replace=False) for s in strata])
    undirected = (out_csr + out_csr.T).tocsr()
    undirected.data[:] = 1
    matrix = csgraph.shortest_path(undirected, method="D", unweighted=True, indices=roots)
    pairs = []
    for row, root in enumerate(roots):
        hops = {}
        for node, hop in enumerate(matrix[row]):
            if 1 <= hop <= 4:
                hops.setdefault(int(hop), []).append(int(node))
        if not hops:
            continue
        keys = sorted(hops)
        hop = int(keys[rng.integers(0, len(keys))])
        pairs.append((int(root), int(hops[hop][rng.integers(0, len(hops[hop]))]), hop))
    return out_csr, in_csr, embeddings, structural, pairs[:PROBE_PAIRS], rng


def main() -> int:
    out_csr, in_csr, embeddings, structural, pairs, rng = build_probe()
    model = T.MixedGeometryTeacher(embeddings.shape[1], structural.shape[1]).to(T.DEVICE)
    model.eval()
    full_x = torch.cat([torch.tensor(embeddings, device=T.DEVICE),
                        torch.tensor(structural, device=T.DEVICE)], dim=1)
    index_i = torch.tensor([p[0] for p in pairs], device=T.DEVICE)
    index_j = torch.tensor([p[1] for p in pairs], device=T.DEVICE)

    with torch.no_grad():
        raw = model.embeddings(full_x, T.edge_index(out_csr))

    def median_distance(embedding, curvature, alpha: float) -> float:
        scaled = manifold.rescale(embedding, alpha, curvature)
        d = manifold.dist(scaled[index_i], scaled[index_j], curvature)
        return float(d.median())

    before = [median_distance(emb, c, 1.0) for emb, c in zip(raw, T.CURVATURES)]
    target = float(np.median(before))
    alphas, after, notes = [], [], []
    for embedding, curvature, current in zip(raw, T.CURVATURES, before):
        low, high = ALPHA_BOUNDS
        if curvature > 0:
            span = math.pi / math.sqrt(curvature)
        else:
            span = None
        if span is not None and target > span:
            alphas.append(high)
            after.append(median_distance(embedding, curvature, high))
            notes.append(f"curvature {curvature}: target {target:.4f} exceeds the maximum "
                         f"geodesic distance {span:.4f}; alpha pinned at {high}")
            continue
        for _ in range(BISECTION_STEPS):
            middle = 0.5 * (low + high)
            value = median_distance(embedding, curvature, middle)
            if abs(value - target) < 1e-4 * max(target, 1e-6):
                break
            if value < target:
                low = middle
            else:
                high = middle
        alpha = 0.5 * (low + high)
        alphas.append(alpha)
        after.append(median_distance(embedding, curvature, alpha))
        notes.append(f"curvature {curvature}: alpha {alpha:.6f}")

    report = {
        "target_median_distance": target,
        "target_rule": "median of the five experts' probe median distances on a freshly initialised model",
        "probe_pairs": len(pairs),
        "seed": SEED,
        "alpha": alphas,
        "before": before,
        "after": after,
        "notes": notes,
        "curvatures": list(T.CURVATURES),
        "post_scale_spread": (max(after) / min(after)) if min(after) > 0 else None,
    }
    (HERE / "teacher").mkdir(parents=True, exist_ok=True)
    (HERE / "teacher" / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
