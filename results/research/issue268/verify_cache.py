"""Issue #268 stage E: independent cache verification and stratification.

Re-reads the frozen cache, checks dtype/shape/legal values/hashes, recomputes
J at kappa=0 and at the stored kappa for deterministic rows **from the raw pairs**
(no node decomposition), and reports the curvature distribution by centre degree,
endpoint hop distance and target sign.  Read-only; it never rewrites the cache.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.graph_infra.assets import load_sparse

ISSUE = REPO / "results/research/issue268"
ART = ISSUE / "artifacts"
CACHE = REPO / "results/stage2_RQ-VAE/TIGER_fixed_triplet_kappa_rq/kappa_cache"
GRAPH = REPO / "assets/train/visible_undirected.npz"
TARGETS = REPO / "assets/v8_runtime_layout/targets.npy"
OFFSETS = REPO / "assets/v8_runtime_layout/target_offsets.npy"
Z_REF = ART / "z_ref.npy"
CHECK_ROWS = 1000
SEED = 268


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    started = time.time()
    manifest = json.loads((CACHE / "manifest.json").read_text())
    grid = np.asarray(manifest["grid"], dtype=np.float64)
    kappa = np.load(CACHE / "kappa.npy", mmap_mode="r")
    j_zero = np.load(CACHE / "j_zero.npy", mmap_mode="r")
    j_fit = np.load(CACHE / "j_fit.npy", mmap_mode="r")
    hashes = {name: sha256(CACHE / name) for name in
              ("kappa.npy", "j_zero.npy", "j_fit.npy", "manifest.json")}
    if kappa.dtype != np.float32 or j_zero.dtype != np.float32 or j_fit.dtype != np.float32:
        raise ValueError("cache dtype changed")
    if not (len(kappa) == len(j_zero) == len(j_fit) == manifest["rows"]):
        raise ValueError("cache row count changed")
    if not np.isfinite(kappa).all():
        raise ValueError("non-finite kappa in cache")
    for name, key in (("kappa.npy", "kappa_sha256"),
                      ("j_zero.npy", "j_zero_sha256"),
                      ("j_fit.npy", "j_fit_sha256")):
        if hashes[name] != manifest["outputs"][key]:
            raise ValueError("cache file changed after the fit: " + name)
    sample = np.asarray(kappa[np.linspace(0, len(kappa) - 1, 100000).astype(np.int64)])
    nearest = grid[np.abs(sample[:, None] - grid[None, :]).argmin(1)]
    off_grid = float(np.abs(nearest - sample).max())

    z = np.load(Z_REF).astype(np.float64)
    graph = load_sparse(GRAPH).tocsr().astype(np.int8)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.diff(graph.indptr)
    coo = graph.tocoo()
    upper = coo.row < coo.col
    edge_keys = set((coo.row[upper].astype(np.int64) * len(z)
                     + coo.col[upper].astype(np.int64)).tolist())
    lengths = np.linalg.norm(z[coo.row[upper]] - z[coo.col[upper]], axis=1)
    scale = float(np.median(lengths))
    w = z / (2.0 * scale)
    count, labels = connected_components(graph, directed=False)
    rng = np.random.default_rng(SEED)
    blocks = []
    for component in range(count):
        members = np.flatnonzero(labels == component)
        if len(members) < 2:
            continue
        blocks.append(np.sort(rng.choice(members, min(256, len(members)), replace=False)))
    refs = np.concatenate(blocks)
    gdist = shortest_path(graph, unweighted=True, directed=False, indices=refs).T
    usable = np.isfinite(gdist) & (gdist > 0)
    gsafe = np.where(usable, gdist, 1.0)
    d = np.linalg.norm(w[:, None, :] - w[refs][None, :, :], axis=2)

    def j_direct(kappas, rows):
        out = np.empty(len(rows), dtype=np.float64)
        targets = np.load(TARGETS, mmap_mode="r")
        offsets = np.load(OFFSETS, mmap_mode="r")
        centres = (np.searchsorted(offsets, rows, side="right") - 1).astype(np.int64)
        b = np.asarray(targets["b"][rows], dtype=np.int64)
        c = np.asarray(targets["c"][rows], dtype=np.int64)
        for index, row in enumerate(rows):
            kk = float(kappas[index])
            total = 0.0
            for node in (centres[index], b[index], c[index]):
                mask = usable[node]
                ref_ids = refs[mask]
                xy = d[node][mask] ** 2
                x2 = float((w[node] ** 2).sum())
                y2 = (w[ref_ids] ** 2).sum(1)
                if kk > 0:
                    t2 = np.clip(kk * xy / ((1.0 + kk * x2) * (1.0 + kk * y2)),
                                 0.0, 1.0 - 1e-7)
                    node_d2 = 4.0 * np.arcsin(np.sqrt(t2)) ** 2 / kk
                elif kk < 0:
                    a = -kk
                    t2 = np.clip(a * xy / ((1.0 - a * x2) * (1.0 - a * y2)), 0.0, None)
                    node_d2 = 4.0 * np.arcsinh(np.sqrt(t2)) ** 2 / a
                else:
                    node_d2 = 4.0 * xy
                ratio = np.sqrt(node_d2) / gsafe[node][mask]
                total += float(((ratio - 1.0) ** 2).sum())
            g_bc = 1.0 if (min(int(b[index]), int(c[index])) * len(z)
                           + max(int(b[index]), int(c[index]))) in edge_keys else 2.0
            a_bc = np.linalg.norm(w[b[index]] - w[c[index]])
            if kk == 0.0:
                pair = (2.0 * a_bc / g_bc - 1.0) ** 2
            else:
                magnitude = abs(kk)
                t2 = (magnitude * a_bc ** 2
                      / ((1.0 + kk * (w[b[index]] ** 2).sum())
                         * (1.0 + kk * (w[c[index]] ** 2).sum())))
                t2 = max(0.0, t2)
                if kk > 0:
                    pair = (2.0 * np.arcsin(min(np.sqrt(t2), 1 - 1e-7))
                            / np.sqrt(kk) / g_bc - 1.0) ** 2
                else:
                    pair = (2.0 * np.arcsinh(np.sqrt(t2))
                            / np.sqrt(magnitude) / g_bc - 1.0) ** 2
            denominator = int(usable[centres[index]].sum() + usable[b[index]].sum()
                              + usable[c[index]].sum() + 1)
            out[index] = (total + pair) / denominator
        return out

    pick = np.random.default_rng(SEED + 11)
    rows = np.sort(pick.choice(len(kappa), CHECK_ROWS, replace=False))
    stored_fit = np.asarray(j_fit[rows], dtype=np.float64)
    stored_zero = np.asarray(j_zero[rows], dtype=np.float64)
    recomputed_fit = j_direct(np.asarray(kappa[rows], dtype=np.float64), rows)
    recomputed_zero = j_direct(np.zeros(len(rows)), rows)
    references = np.load(TARGETS, mmap_mode="r")
    offsets = np.load(OFFSETS, mmap_mode="r")
    centres = (np.searchsorted(offsets, rows, side="right") - 1).astype(np.int64)
    b = np.asarray(references["b"][rows], dtype=np.int64)
    c = np.asarray(references["c"][rows], dtype=np.int64)
    target = np.asarray(references["target"][rows], dtype=np.float64)
    kappa_rows = np.asarray(kappa[rows], dtype=np.float64)
    triples = np.column_stack((centres, b, c))
    graph_degree = degree[centres]
    hop = np.fromiter(
        (1.0 if (min(int(u), int(v)) * len(z) + max(int(u), int(v))) in edge_keys
         else 2.0 for u, v in zip(b, c)), dtype=np.float64, count=len(b))

    def strata(values):
        buckets = {}
        for name, mask in (("degree_le_10", graph_degree <= 10),
                           ("degree_11_30", (graph_degree > 10) & (graph_degree <= 30)),
                           ("degree_gt_30", graph_degree > 30),
                           ("target_negative", target < 0),
                           ("target_nonnegative", target >= 0),
                           ("bc_adjacent", hop == 1.0),
                           ("bc_two_hop", hop == 2.0)):
            if mask.any():
                buckets[name] = {"rows": int(mask.sum()),
                                 "kappa_mean": float(values[mask].mean()),
                                 "kappa_std": float(values[mask].std()),
                                 "at_floor_fraction": float(
                                     (values[mask] <= grid[0] + 1e-9).mean())}
        return buckets

    summary = {
        "schema": "genrec.issue268.cache_verification/1",
        "rows": int(len(kappa)),
        "dtype": str(kappa.dtype),
        "hashes_match_manifest": True,
        "max_off_grid_on_100k_uniform_rows": off_grid,
        "recompute_rows": int(len(rows)),
        "recompute_j_zero_max_abs_error": float(np.abs(recomputed_zero - stored_zero).max()),
        "recompute_j_fit_max_abs_error": float(np.abs(recomputed_fit - stored_fit).max()),
        "stratification": strata(kappa_rows),
        "runtime_s": time.time() - started,
    }
    (ISSUE / "analysis").mkdir(parents=True, exist_ok=True)
    (ISSUE / "analysis/CACHE_VERIFICATION.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
