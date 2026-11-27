"""Issue #269 stage 2: full per-triple curvature fit at the calibrated global scale.

Frozen inputs: Z_ref (`artifacts/z_ref.npy`), the training-visible undirected
graph, the packed target table and its offsets.  Frozen procedure: the
graph-unit global scale (median Euclidean length of a visible edge), the
kappa-stereographic family, a deterministic 256-reference subset per component,
and the closed-form node decomposition of the distortion.

Outputs are immutable after this script finishes: `kappa.npy` (one float32 value
per target row), per-row baseline/fitted distortion, and a manifest with every
input and output hash, coverage, boundary/flat statistics and the runtime.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components, shortest_path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.graph_infra.assets import load_sparse

ISSUE = REPO / "results/research/issue269"
ART = REPO / "results/research/issue268/artifacts"
CACHE = REPO / "results/stage2_RQ-VAE/TIGER_fixed_triplet_kappa_rq/kappa_cache_s0p068284"
GRAPH = REPO / "assets/train/visible_undirected.npz"
TARGETS = REPO / "assets/v8_runtime_layout/targets.npy"
OFFSETS = REPO / "assets/v8_runtime_layout/target_offsets.npy"
Z_REF = ART / "z_ref.npy"

SEED = 268
REFERENCE_SIZE = 256
GRID_STEP = 0.005
CHUNK = 65536


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def references_and_distances(graph):
    count, labels = connected_components(graph, directed=False)
    rng = np.random.default_rng(SEED)
    blocks = []
    for component in range(count):
        members = np.flatnonzero(labels == component)
        if len(members) < 2:
            continue
        take = min(REFERENCE_SIZE, len(members))
        blocks.append(np.sort(rng.choice(members, take, replace=False)))
    refs = np.concatenate(blocks)
    distances = shortest_path(graph, unweighted=True, directed=False, indices=refs).T
    return refs, distances


def squared_torch(x, y, k):
    xy = (x - y).square().sum(-1)
    x2 = x.square().sum(-1)
    y2 = y.square().sum(-1)
    magnitude = k.abs().clamp_min(torch.finfo(x.dtype).tiny)
    denominator = (1.0 + k * x2) * (1.0 + k * y2)
    if bool((denominator <= 0).any()):
        raise ValueError("coordinate outside the frozen curvature domain")
    t2 = (magnitude * xy / denominator).clamp_min(0.0)
    sphere = torch.asin(t2.clamp(max=1.0 - 1e-7).sqrt()).square() / magnitude
    hyperbolic = torch.asinh(t2.sqrt()).square() / magnitude
    values = torch.where(k > 0, sphere, torch.where(k < 0, hyperbolic, xy))
    return 4.0 * values


def main() -> None:
    started = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = np.load(Z_REF).astype(np.float64)
    graph = load_sparse(GRAPH).tocsr().astype(np.int8)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    coo = graph.tocoo()
    upper = coo.row < coo.col
    edge_keys = set((coo.row[upper].astype(np.int64) * len(z)
                     + coo.col[upper].astype(np.int64)).tolist())
    lengths = np.linalg.norm(z[coo.row[upper]] - z[coo.col[upper]], axis=1)
    edge_scale = float(np.median(lengths))
    # Frozen by the #269 weighted-least-squares check (GATE.json), not refitted here.
    scale = 0.06828367045803868
    max_coord = float(np.linalg.norm(z / (2.0 * scale), axis=1).max())
    lower = float(np.ceil((-0.99 / (max_coord ** 2)) / GRID_STEP) * GRID_STEP)
    grid = np.round(np.arange(lower, 1.0 + 1e-9, GRID_STEP), 10)
    grid = np.unique(np.round(np.append(grid, 0.0), 10))

    refs, gdist = references_and_distances(graph)
    usable = np.isfinite(gdist) & (gdist > 0)
    counts = usable.sum(1).astype(np.float32)
    w = torch.from_numpy((z / (2.0 * scale)).astype(np.float32)).to(device)
    wr = w[torch.from_numpy(refs).to(device)]
    gval = torch.from_numpy(np.where(usable, gdist, 1.0).astype(np.float32)).to(device)
    usable_t = torch.from_numpy(usable).to(device)

    nodes = torch.arange(len(z), device=device)
    f_table = torch.zeros((len(grid), len(z)), dtype=torch.float32, device=device)
    with torch.no_grad():
        for gindex, k in enumerate(grid):
            kk = torch.as_tensor(float(k), dtype=torch.float32, device=device)
            total = torch.zeros(len(z), device=device)
            for start in range(0, len(z), 512):
                ids = nodes[start:start + 512]
                d2 = squared_torch(w[ids][:, None, :].expand(-1, len(refs), -1),
                                   wr[None], kk)
                ratio = d2.clamp_min(0).sqrt() / gval[ids]
                terms = torch.where(usable_t[ids], (ratio - 1.0).square(),
                                    torch.zeros_like(ratio))
                total[ids] = terms.sum(1)
            f_table[gindex] = total
        f_table = f_table.cpu().numpy()

    targets = np.load(TARGETS, mmap_mode="r")
    offsets = np.load(OFFSETS, mmap_mode="r")
    total_rows = int(offsets[-1])
    kappa = np.zeros(total_rows, dtype=np.float32)
    j_zero = np.zeros(total_rows, dtype=np.float32)
    j_fit = np.full(total_rows, np.inf, dtype=np.float32)
    second = np.full(total_rows, np.inf, dtype=np.float32)
    valid = np.isfinite(targets["target"])
    zero_index = int(np.flatnonzero(grid == 0.0)[0])

    for start in range(0, total_rows, CHUNK):
        stop = min(start + CHUNK, total_rows)
        rows = np.arange(start, stop, dtype=np.int64)
        centres = (np.searchsorted(offsets, rows, side="right") - 1).astype(np.int32)
        b = np.asarray(targets["b"][start:stop], dtype=np.int64)
        c = np.asarray(targets["c"][start:stop], dtype=np.int64)
        low = np.minimum(b, c) * len(z) + np.maximum(b, c)
        g_bc = np.fromiter((1.0 if key in edge_keys else 2.0 for key in low.tolist()),
                           dtype=np.float32, count=len(low))
        keep = valid[start:stop]
        m_t = torch.from_numpy(centres).to(device)
        b_t = torch.from_numpy(b.astype(np.int32)).to(device)
        c_t = torch.from_numpy(c.astype(np.int32)).to(device)
        xb = w[b_t]
        xc = w[c_t]
        xy = (xb - xc).square().sum(-1).cpu().numpy().astype(np.float32)
        x2 = xb.square().sum(-1).cpu().numpy().astype(np.float32)
        y2 = xc.square().sum(-1).cpu().numpy().astype(np.float32)
        denom = counts[centres] + counts[b] + counts[c] + 1.0
        chunk_best = np.full(len(rows), np.inf, dtype=np.float32)
        chunk_best_k = np.zeros(len(rows), dtype=np.float32)
        chunk_second = np.full(len(rows), np.inf, dtype=np.float32)
        for gindex, k in enumerate(grid):
            kk = float(k)
            magnitude = abs(kk) if kk != 0 else 1e-30
            t2 = magnitude * xy / ((1.0 + kk * x2) * (1.0 + kk * y2))
            # pairwise term via the same elementwise formula
            if kk > 0:
                d_pair = 2.0 * np.arcsin(np.sqrt(np.clip(t2, 0.0, 1.0 - 1e-7))) / np.sqrt(magnitude)
            elif kk < 0:
                d_pair = 2.0 * np.arcsinh(np.sqrt(np.clip(t2, 0.0, None))) / np.sqrt(magnitude)
            else:
                d_pair = 2.0 * np.sqrt(xy)
            q = (d_pair / g_bc - 1.0) ** 2
            value = (f_table[gindex][centres] + f_table[gindex][b]
                     + f_table[gindex][c] + q.astype(np.float32)) / denom
            value = np.where(keep, value.astype(np.float32), np.inf)
            update = value < chunk_best
            chunk_second = np.where(update, chunk_best, np.minimum(chunk_second, value))
            chunk_best = np.where(update, value, chunk_best)
            chunk_best_k = np.where(update, np.float32(kk), chunk_best_k)
        # Exact ties at the Euclidean baseline resolve to kappa=0.
        value_zero = (f_table[zero_index][centres] + f_table[zero_index][b]
                      + f_table[zero_index][c] + (2.0 * np.sqrt(xy) / g_bc - 1.0) ** 2) / denom
        value_zero = value_zero.astype(np.float32)
        tie = chunk_best >= value_zero - 1e-12
        chunk_best_k = np.where(tie, np.float32(0.0), chunk_best_k)
        kappa[start:stop] = np.where(keep, chunk_best_k, np.float32(0.0))
        j_zero[start:stop] = np.where(keep, value_zero, np.nan)
        j_fit[start:stop] = np.where(keep, chunk_best, np.nan)
        second[start:stop] = np.where(keep, chunk_second, np.nan)
        print("rows %d..%d best_mean=%.5f j0_mean=%.5f" % (
            start, stop, float(chunk_best[keep].mean()), float(value_zero[keep].mean())),
            flush=True)

    valid_kappa = kappa[np.asarray(valid)].astype(np.float64)
    j0v = j_zero[np.asarray(valid)].astype(np.float64)
    jfv = j_fit[np.asarray(valid)].astype(np.float64)
    improved = jfv < j0v - 1e-12
    improvement = np.where(np.isfinite(j0v), (j0v - jfv) / j0v, np.nan)
    boundary = (kappa <= grid[0] + 1e-9) | (kappa >= grid[-1] - 1e-9)
    flat = (np.abs(j0v - jfv) <= 1e-12)
    np.save(CACHE / "kappa.npy", kappa)
    np.save(CACHE / "j_zero.npy", j_zero)
    np.save(CACHE / "j_fit.npy", j_fit)
    manifest = {
        "schema": "genrec.issue269.kappa_cache/1",
        "issue": 269,
        "rows": int(total_rows),
        "valid_rows": int(np.asarray(valid).sum()),
        "grid": grid.tolist(), "grid_step": GRID_STEP,
        "k_min": float(grid[0]), "k_max": float(grid[-1]),
        "scale": scale,
        "scale_rule": "calibrated weighted-least-squares optimum (#269)",
        "edge_scale": edge_scale,
        "scale_source_sha256": sha256(ISSUE / "analysis" / "GATE.json"),
        "max_coord_norm": max_coord,
        "references": int(len(refs)), "reference_seed": SEED,
        "reference_size": REFERENCE_SIZE,
        "inputs": {
            "z_ref": str(Z_REF.relative_to(REPO)), "z_ref_sha256": sha256(Z_REF),
            "graph": str(GRAPH.relative_to(REPO)), "graph_sha256": sha256(GRAPH),
            "targets_sha256": sha256(TARGETS), "offsets_sha256": sha256(OFFSETS),
            "script_sha256": sha256(Path(__file__)),
        },
        "outputs": {
            "kappa_sha256": sha256(CACHE / "kappa.npy"),
            "j_zero_sha256": sha256(CACHE / "j_zero.npy"),
            "j_fit_sha256": sha256(CACHE / "j_fit.npy"),
        },
        "statistics": {
            "mean_j_zero": float(np.nanmean(j0v)),
            "mean_j_fit": float(np.nanmean(jfv)),
            "improved_fraction": float(improved.mean()),
            "improvement_median": float(np.nanmedian(improvement)),
            "improvement_p10": float(np.nanpercentile(improvement, 10)),
            "improvement_p90": float(np.nanpercentile(improvement, 90)),
            "flat_fraction": float(flat.mean()),
            "boundary_fraction": float(boundary[np.asarray(valid)].mean()),
            "kappa_mean": float(valid_kappa.mean()),
            "kappa_std": float(valid_kappa.std()),
            "kappa_positive_fraction": float((valid_kappa > 0).mean()),
            "kappa_zero_fraction": float((valid_kappa == 0).mean()),
            "median_gap_to_second": float(np.nanmedian(
                np.abs(np.asarray(second[valid], dtype=np.float64) - jfv))),
        },
        "runtime_s": time.time() - started,
        "package": {"numpy": np.__version__, "torch": torch.__version__},
        "dtype": {"kappa": "float32", "j_zero": "float32", "j_fit": "float32"},
    }
    (CACHE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["statistics"], indent=1))


if __name__ == "__main__":
    main()
