"""Issue #269 stage 1-2: is per-triple curvature identifiable after scale calibration?

Fit a per-triple curvature on reference set A (seed 268), evaluate it on the
independent reference set B (seed 269); compare against one global curvature
fitted on A, the calibrated Euclidean kappa=0, and the coordinate floor.  The
gate is the issue's pre-registered one: mean held-out J at least 1 % below the
global-kappa J, a centre-bootstrap 95 % lower bound above zero, and a
reference-perturbation Spearman of the two per-triple fits >= 0.5.

This is an offline diagnostic; it reads the frozen graph, targets and Z_ref and
writes only under results/research/issue269/.
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
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.graph_infra.assets import load_sparse

ISSUE = REPO / "results/research/issue269"
ART = REPO / "results/research/issue268/artifacts"
GRAPH = REPO / "assets/train/visible_undirected.npz"
TARGETS = REPO / "assets/v8_runtime_layout/targets.npy"
OFFSETS = REPO / "assets/v8_runtime_layout/target_offsets.npy"
Z_REF = ART / "z_ref.npy"
SCALE = 0.06828367045803868  # frozen by #268 scale_control; recomputed below
GRID_STEP = 0.005
REFERENCE_SIZE = 256
SAMPLE_ROWS = 2048
SEED_A = 268
SEED_B = 269
BOOTSTRAP = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def references(graph, labels, seed, size):
    count = int(labels.max()) + 1
    rng = np.random.default_rng(seed)
    blocks = []
    for component in range(count):
        members = np.flatnonzero(labels == component)
        if len(members) < 2:
            continue
        blocks.append(np.sort(rng.choice(members, min(size, len(members)), replace=False)))
    return np.concatenate(blocks)


def squared_torch(x, y, k):
    """Same arithmetic as the frozen offline probe (issue #268)."""
    diff = x - y
    xy = (diff * diff).sum(-1)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    kk = k if torch.is_tensor(k) else torch.as_tensor(k, dtype=x.dtype, device=x.device)
    kk = kk.to(x.dtype)
    magnitude = kk.abs().clamp(min=1e-30)
    denominator = (1.0 + kk * x2) * (1.0 + kk * y2)
    if bool((denominator <= 0).any()):
        raise ValueError("coordinate outside the frozen curvature domain")
    t2 = (magnitude * xy / denominator).clamp(min=0.0)
    root = t2.clamp(max=1.0 - 1e-7).sqrt()
    sphere = torch.asin(root).square() / magnitude
    hyperbolic = torch.asinh(root).square() / magnitude
    values = torch.where(kk > 0, sphere,
                         torch.where(kk < 0, hyperbolic, xy))
    return 4.0 * values


def main() -> None:
    started = time.time()
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
    _, labels = connected_components(graph, directed=False)

    # Recompute the global scale exactly as #268's scale_control did.
    lengths = np.linalg.norm(z[coo.row[upper]] - z[coo.col[upper]], axis=1)
    edge_scale = float(np.median(lengths))
    scale = SCALE
    if abs(scale - edge_scale / 1.5508007260566352) > 1e-12:
        raise ValueError("frozen scale does not match the #268 closed-form optimum")
    w = torch.from_numpy((z / (2.0 * scale)).astype(np.float32)).to(device)
    max_coord = float(np.linalg.norm(z / (2.0 * scale), axis=1).max())
    k_min = float(np.ceil((-0.99 / (max_coord ** 2)) / GRID_STEP) * GRID_STEP)
    grid = np.round(np.arange(k_min, 1.0 + 1e-9, GRID_STEP), 10)
    if not np.any(grid == 0.0):
        grid = np.unique(np.round(np.append(grid, 0.0), 10))

    targets = np.load(TARGETS, mmap_mode="r")
    offsets = np.load(OFFSETS, mmap_mode="r")
    rng = np.random.default_rng(SEED_A + 1)
    rows = np.sort(rng.choice(int(offsets[-1]), SAMPLE_ROWS, replace=False))
    rows = rows[np.isfinite(np.asarray(targets["target"][rows]))]
    centres = (np.searchsorted(offsets, rows, side="right") - 1).astype(np.int64)
    b = np.asarray(targets["b"][rows], dtype=np.int64)
    c = np.asarray(targets["c"][rows], dtype=np.int64)
    pair_g = np.fromiter(
        (1.0 if (min(int(u), int(v)) * len(z) + max(int(u), int(v))) in edge_keys else 2.0
         for u, v in zip(b, c)), dtype=np.float32, count=len(b))

    def node_table(refs):
        gdist = shortest_path(graph, unweighted=True, directed=False, indices=refs).T
        usable = np.isfinite(gdist) & (gdist > 0)
        gval = np.where(usable, gdist, 1.0).astype(np.float32)
        counts = usable.sum(1).astype(np.float32)
        usable_t = torch.from_numpy(usable).to(device)
        gval_t = torch.from_numpy(gval).to(device)
        wr = w[torch.from_numpy(refs).to(device)]
        table = np.empty((len(grid), len(z)), dtype=np.float32)
        all_nodes = torch.arange(len(w), device=device)
        with torch.no_grad():
            for index, k in enumerate(grid):
                kk = torch.as_tensor(float(k), dtype=torch.float32, device=device)
                total = torch.zeros(len(z), device=device)
                for start in range(0, len(all_nodes), 512):
                    ids = all_nodes[start:start + 512]
                    d2 = squared_torch(w[ids][:, None, :].expand(-1, len(refs), -1), wr[None], kk)
                    ratio = d2.clamp_min(0).sqrt() / gval_t[ids]
                    terms = torch.where(usable_t[ids], (ratio - 1.0).square(),
                                        torch.zeros_like(ratio))
                    total[ids] = terms.sum(1)
                table[index] = total.cpu().numpy()
        return table, counts, refs

    def j_table(ref_table, counts):
        """J for every grid value and every sampled triple under one reference set."""
        denominator = counts[centres] + counts[b] + counts[c] + 1.0
        out = np.empty((len(grid), len(rows)), dtype=np.float64)
        # Pair distance does not depend on the reference set, only on kappa.
        xy = ((w[torch.from_numpy(b.astype(np.int64)).to(device)]
               - w[torch.from_numpy(c.astype(np.int64)).to(device)]) ** 2).sum(-1).cpu().numpy()
        x2 = (w[torch.from_numpy(b.astype(np.int64)).to(device)] ** 2).sum(-1).cpu().numpy()
        y2 = (w[torch.from_numpy(c.astype(np.int64)).to(device)] ** 2).sum(-1).cpu().numpy()
        for index, k in enumerate(grid):
            if k == 0:
                pair = 4.0 * xy
            else:
                magnitude = abs(float(k))
                t2 = magnitude * xy / ((1.0 + float(k) * x2) * (1.0 + float(k) * y2))
                if k > 0:
                    pair = 4.0 * np.arcsin(np.sqrt(np.clip(t2, 0.0, 1.0 - 1e-7))) ** 2 / magnitude
                else:
                    pair = 4.0 * np.arcsinh(np.sqrt(np.clip(t2, 0.0, None))) ** 2 / magnitude
            total = (ref_table[index][centres] + ref_table[index][b]
                     + ref_table[index][c] + (pair / pair_g - 1.0) ** 2)
            out[index] = total / denominator
        return out

    refs_a = references(graph, labels, SEED_A, REFERENCE_SIZE)
    refs_b = references(graph, labels, SEED_B, REFERENCE_SIZE)
    table_a, counts_a, _ = node_table(refs_a)
    table_b, counts_b, _ = node_table(refs_b)
    j_a = j_table(table_a, counts_a)
    j_b = j_table(table_b, counts_b)

    zero_index = int(np.argmin(np.abs(grid)))
    floor_index = 0
    # Per-triple fit on A; evaluate the same grid index on B.
    best_index_a = np.argmin(j_a, axis=0)
    best_index_b = np.argmin(j_b, axis=0)
    j_triple_a = j_a[best_index_a, np.arange(len(rows))]
    j_triple_b = j_b[best_index_a, np.arange(len(rows))]
    global_index_a = int(np.argmin(j_a.mean(1)))
    global_k_a = float(grid[global_index_a])
    j_global_a = j_a[global_index_a]
    j_global_b = j_b[global_index_a]
    j_triple_fit_b = j_b[best_index_b, np.arange(len(rows))]  # in-sample-B ceiling (diagnostic)
    j_zero_a = j_a[zero_index]
    j_zero_b = j_b[zero_index]

    # Closed-form reproduction of #268's global scale on the exact 20k sample:
    # c* = (sum a + a_bc) / (sum a^2 + a_bc^2) with a = d/(edge_scale*g).
    refs_scale = references(graph, labels, SEED_B, REFERENCE_SIZE)
    gdist_scale = shortest_path(graph, unweighted=True, directed=False,
                                indices=refs_scale).T
    usable_scale = np.isfinite(gdist_scale) & (gdist_scale > 0)
    d_scale = np.linalg.norm(z[:, None, :] - z[refs_scale][None, :, :], axis=2)
    ratio_scale = np.where(usable_scale,
                           d_scale / (edge_scale * np.where(usable_scale, gdist_scale, 1.0)),
                           0.0)
    t1 = ratio_scale.sum(1).astype(np.float64)
    t2 = (ratio_scale * ratio_scale).sum(1).astype(np.float64)
    n_valid = usable_scale.sum(1).astype(np.float64)
    pick = np.random.default_rng(271)
    rows20k = np.sort(pick.choice(int(offsets[-1]), 20000, replace=False))
    rows20k = rows20k[np.isfinite(np.asarray(targets["target"][rows20k]))]
    c20 = (np.searchsorted(offsets, rows20k, side="right") - 1).astype(np.int64)
    b20 = np.asarray(targets["b"][rows20k], dtype=np.int64)
    c20n = np.asarray(targets["c"][rows20k], dtype=np.int64)
    a_bc = np.linalg.norm(z[b20] - z[c20n], axis=1) / edge_scale
    a_bc = a_bc / np.fromiter(
        (1.0 if (min(int(u), int(v)) * len(z) + max(int(u), int(v))) in edge_keys else 2.0
         for u, v in zip(b20, c20n)), dtype=np.float64, count=len(b20))
    t1_sum = t1[c20] + t1[b20] + t1[c20n]
    t2_sum = t2[c20] + t2[b20] + t2[c20n]
    denominator20 = n_valid[c20] + n_valid[b20] + n_valid[c20n] + 1.0
    # Weighted least squares: one global multiplier minimises the summed
    # quadratic, so the optimum is the aggregate ratio, not the mean of rows.
    c_star = float((t1_sum + a_bc).sum() / (t2_sum + a_bc ** 2).sum())
    scale_reproduced = edge_scale / c_star
    scale_error = abs(scale_reproduced - scale) / scale

    # Scale-invariant shape error on the held-out set: the exact per-row
    # minimiser over one free multiplier (diagnostic only, not a mechanism).
    rho_scale = np.linalg.norm(
        z[centres][:, None, :] - z[refs_b][None, :, :], axis=2)
    v_scale = np.isfinite(gdist_b := shortest_path(
        graph, unweighted=True, directed=False, indices=refs_b).T)
    usable_b = v_scale & (gdist_b > 0)
    a_m = np.where(usable_b[centres], rho_scale / (edge_scale * np.where(usable_b[centres], gdist_b[centres], 1.0)), 0.0)
    a_b = np.linalg.norm(z[b] - z[c], axis=1) / edge_scale
    a_b = a_b / pair_g
    sum_a = a_m.sum(1) + a_b
    sum_a2 = (a_m * a_m).sum(1) + a_b * a_b
    denominator_shape = usable_b[centres].sum(1) + 1.0
    c_row = sum_a / np.maximum(sum_a2, 1e-30)
    j_shape = (sum_a2 * c_row ** 2 - 2.0 * sum_a * c_row + denominator_shape) / denominator_shape
    improvement = (j_global_b - j_triple_b) / j_global_b
    # Centre bootstrap of the mean improvement.
    unique_centres, centre_index = np.unique(centres, return_inverse=True)
    rng_boot = np.random.default_rng(SEED_A)
    order = np.argsort(centre_index, kind="stable")
    grouped = np.split(order, np.cumsum(np.bincount(centre_index))[:-1])
    boot = np.empty(BOOTSTRAP)
    for draw in range(BOOTSTRAP):
        pick = rng_boot.integers(0, len(unique_centres), len(unique_centres))
        selected = np.concatenate([grouped[index] for index in pick])
        boot[draw] = float(improvement[selected].mean())
    rho = float(spearmanr(grid[best_index_a], grid[best_index_b]).correlation)

    boundary_a = float(((best_index_a == floor_index) | (best_index_a == len(grid) - 1)).mean())
    boundary_b = float(((best_index_b == floor_index) | (best_index_b == len(grid) - 1)).mean())
    summary = {
        "schema": "genrec.issue269.gate/1",
        "scale": scale,
        "edge_scale": edge_scale,
        "max_coord_norm": max_coord,
        "grid_min": float(grid[0]), "grid_max": float(grid[-1]),
        "grid_points": int(len(grid)),
        "rows": int(len(rows)),
        "reference_sets": {"A": int(len(refs_a)), "B": int(len(refs_b))},
        "mean_j_zero_a": float(j_zero_a.mean()), "mean_j_zero_b": float(j_zero_b.mean()),
        "mean_j_global_a": float(j_global_a.mean()), "mean_j_global_b": float(j_global_b.mean()),
        "mean_j_triple_a": float(j_triple_a.mean()), "mean_j_triple_b": float(j_triple_b.mean()),
        "mean_j_triple_insample_b": float(j_triple_fit_b.mean()),
        "global_kappa_a": global_k_a,
        "improvement_mean_heldout": float(improvement.mean()),
        "improvement_median_heldout": float(np.median(improvement)),
        "improvement_bootstrap_ci95": [float(np.quantile(boot, 0.025)),
                                       float(np.quantile(boot, 0.975))],
        "improvement_bootstrap_mean": float(boot.mean()),
        "reference_spearman": rho,
        "boundary_fraction_a": boundary_a,
        "scale_reproduced": scale_reproduced,
        "scale_relative_error": scale_error,
        "rows_scale_check": int(len(rows20k)),
        "mean_j_shape_free_scale_b": float(j_shape.mean()),
        "boundary_fraction_b": boundary_b,
        "zero_vs_global_b": float((j_zero_b.mean() - j_global_b.mean()) / j_zero_b.mean()),
        "gate": {
            "mean_improvement_at_least_1pct": bool(improvement.mean() >= 0.01),
            "bootstrap_lower_above_zero": bool(np.quantile(boot, 0.025) > 0.0),
            "spearman_at_least_half": bool(rho >= 0.5),
        },
        "graph_sha256": sha256(GRAPH),
        "targets_sha256": sha256(TARGETS),
        "z_ref_sha256": sha256(Z_REF),
        "runtime_s": time.time() - started,
    }
    summary["gate"]["passed"] = bool(all(summary["gate"].values()))
    ISSUE.joinpath("analysis").mkdir(parents=True, exist_ok=True)
    (ISSUE / "analysis" / "GATE.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        ISSUE / "analysis" / "PER_ROW.npz",
        rows=rows.astype(np.int64), centres=centres.astype(np.int64),
        kappa_a=grid[best_index_a].astype(np.float32),
        kappa_b=grid[best_index_b].astype(np.float32),
        j_zero_b=j_zero_b.astype(np.float64),
        j_global_b=j_global_b.astype(np.float64),
        j_triple_b=j_triple_b.astype(np.float64),
        improvement=improvement.astype(np.float64),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "gate"}, indent=1))
    print("gate", json.dumps(summary["gate"]))


if __name__ == "__main__":
    main()
