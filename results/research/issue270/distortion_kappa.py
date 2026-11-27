"""Issue #270: J under the frozen #269 kappa_t ruler on the 20k sample (primary arms)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.graph_infra.assets import load_sparse
OUT = REPO / "results/research/issue270"
GEOM = OUT / "assets/geometry"
GRAPH = REPO / "assets/train/visible_undirected.npz"
TARGETS = REPO / "assets/v8_runtime_layout/targets.npy"
OFFSETS = REPO / "assets/v8_runtime_layout/target_offsets.npy"
KAPPA = REPO / "results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284/kappa.npy"
S = 0.06828367045803868
SEED, SAMPLE, REFS, N = 269, 20000, 256, 24587


def squared(x, y, k):
    xy = ((x - y) ** 2).sum(-1)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    mag = np.maximum(np.abs(k), 1e-30)
    den = (1 + k * x2) * (1 + k * y2)
    t2 = np.clip(mag * xy / den, 0, None)
    root = np.sqrt(np.minimum(t2, 1 - 1e-7))
    sphere = np.arcsin(root) ** 2 / mag
    hyp = np.arcsinh(np.sqrt(t2)) ** 2 / mag
    return 4 * np.where(k > 0, sphere, np.where(k < 0, hyp, xy))


def main() -> None:
    arm = sys.argv[1]
    z = np.load(GEOM / (arm + ".npz"))["q"].astype(np.float64)
    w = z / (2 * S)
    g = load_sparse(GRAPH).tocsr().astype(np.int8)
    g = g.maximum(g.T).tocsr(); g.setdiag(0); g.eliminate_zeros()
    _, labels = connected_components(g, directed=False)
    rng = np.random.default_rng(SEED)
    blocks = []
    for c in range(int(labels.max()) + 1):
        members = np.flatnonzero(labels == c)
        if len(members) >= 2:
            blocks.append(np.sort(rng.choice(members, min(REFS, len(members)), replace=False)))
    refs = np.concatenate(blocks)
    gd = shortest_path(g, unweighted=True, directed=False, indices=refs).T
    use = np.isfinite(gd) & (gd > 0)
    gv = np.where(use, gd, 1.0)
    t = np.load(TARGETS, mmap_mode="r"); o = np.load(OFFSETS, mmap_mode="r")
    rows = np.sort(np.random.default_rng(SEED + 1).choice(int(o[-1]), SAMPLE, replace=False))
    rows = rows[np.isfinite(np.asarray(t["target"][rows]))]
    centre = np.searchsorted(o, rows, side="right") - 1
    b = np.asarray(t["b"][rows], dtype=np.int64); c = np.asarray(t["c"][rows], dtype=np.int64)
    kappa = np.load(KAPPA, mmap_mode="r")
    k = np.asarray(kappa[rows], dtype=np.float64)
    total = np.zeros(len(rows)); denom = np.zeros(len(rows)); wr = w[refs]
    for node in (centre, b, c):
        for start in range(0, len(rows), 512):
            stop = min(start + 512, len(rows))
            kk = k[start:stop, None]
            d2 = squared(w[node[start:stop]][:, None, :], wr[None], kk)
            ratio = np.sqrt(np.maximum(d2, 0)) / gv[node[start:stop]]
            total[start:stop] += np.where(use[node[start:stop]], (ratio - 1) ** 2, 0).sum(1)
            denom[start:stop] += use[node[start:stop]].sum(1)
    keys = set()
    coo = g.tocoo(); up = coo.row < coo.col
    keys = set((coo.row[up].astype(np.int64) * N + coo.col[up].astype(np.int64)).tolist())
    pg = np.fromiter((1.0 if int(x) in keys else 2.0 for x in
                      np.minimum(b, c) * N + np.maximum(b, c)), dtype=np.float64, count=len(b))
    for start in range(0, len(rows), 512):
        stop = min(start + 512, len(rows))
        kk = k[start:stop, None]
        d2 = squared(w[b[start:stop]], w[c[start:stop]], kk[:, 0])
        total[start:stop] += (np.sqrt(np.maximum(d2, 0)) / pg[start:stop] - 1) ** 2
        denom[start:stop] += 1
    j = total / denom
    out = GEOM / (arm + ".distortion_kappa.json")
    out.write_text(json.dumps({"schema": "genrec.issue270.distortion_kappa/1", "arm": arm,
                               "rows": int(len(rows)), "scale": S, "mean_j": float(j.mean()),
                               "median_j": float(np.median(j))}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arm": arm, "mean_j": float(j.mean())}))


if __name__ == "__main__":
    main()
