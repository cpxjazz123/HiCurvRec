"""Do the SID's own first-level cells prefer different geometric biases?

The ORCA-role probe grouped items by a structural clustering that the SID never
sees.  This probe groups them by the partition the SID actually creates: the
first-level code each item carries.  If different graphlets need different
geometric biases, and the bias is supposed to be actionable, then the cells of
the frozen codebook should show it -- they are the objects whose boundaries the
geometry moves.

For every item the script compares its TRAIN behaviour neighbours against
degree-matched non-neighbours and records the per-item AUC of the F0 mixed
distance at each `alpha`.  Cells are then split by their structural density
(the mean triangle multiplicity of the edges incident to their items) and the
alpha curves are compared across density quintiles, with a cell-level
permutation test on the density/optimal-alpha association.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VECTORS = REPO / "results" / "research" / "tiger_cluster_graph" / "mixed_vectors.npz"
SID_JSON = REPO / "results" / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0" / "item_sids.json"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
TREE = REPO / "analysis_sid_recall" / "r2" / "19" / "TIGER_mixed_rq_F0"

EUCLIDEAN_DIM = 16
ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))
POSITIVES = 20
NEGATIVES = 20
MIN_ITEMS_PER_CELL = 30
CHUNK = 64
BOOTSTRAP = 1000
PERMUTATIONS = 500
SEED = 42


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contracts():
    spec = importlib.util.spec_from_file_location(
        "f0_model", TREE / "model" / "__init__.py",
        submodule_search_locations=[str(TREE / "model")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["f0_model"] = package
    spec.loader.exec_module(package)
    from f0_model.poincare_contract import expmap0, squared_distance as hyp_sq
    return expmap0, hyp_sq


def load_graph():
    train = pd.read_parquet(TRAIN)
    pairs = set()
    for history, target in zip(train["history"].to_numpy(), train["target"].to_numpy()):
        history = list(history)
        for left, right in zip(history, history[1:]):
            pairs.add((left, right) if left < right else (right, left))
        if history:
            tail = history[-1]
            pairs.add((tail, target) if tail < target else (target, tail))
    edges = np.array(sorted(pairs), dtype=np.int64)
    if edges.shape[0] != 315391:
        raise SystemExit(f"behaviour graph has {edges.shape[0]} edges, expected 315391")
    return edges


def item_auc(distance, positives, negatives) -> np.ndarray:
    positive = distance.gather(1, positives)
    negative = distance.gather(1, negatives)
    compare = positive.unsqueeze(2) - negative.unsqueeze(1)
    return (compare < 0).float().add((compare == 0).float() * 0.5).mean(dim=(1, 2)).numpy()


def main():
    expmap0, hyp_sq = load_contracts()
    payload = np.load(VECTORS)
    q = torch.from_numpy(payload["q"].astype(np.float32))
    tokens = payload["tokens"]
    n_items = q.shape[0]
    exported = json.loads(SID_JSON.read_text())
    expected = np.array([[int(v) for v in exported[str(i)][:3]] for i in range(n_items)])
    if not np.array_equal(tokens[:, :3], expected):
        raise SystemExit("vector tokens are not the exported F0 SID")
    cells = tokens[:, 0]

    edges = load_graph()
    adjacency = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_items, n_items)
    ).tocsr()
    adjacency = (adjacency + adjacency.T).tocsr()
    triangle_count = np.rint(np.asarray(
        (adjacency @ adjacency)[edges[:, 0], edges[:, 1]]).ravel()).astype(np.int64)
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int64)
    neighbour = [set(adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]].tolist())
                 for i in range(n_items)]
    print(f"items={n_items} edges={len(edges)} cells={len(np.unique(cells))}")

    # Per-item structural density: mean triangle multiplicity of its incident edges.
    incident = np.zeros(n_items)
    incident_count = np.zeros(n_items)
    for index, (left, right) in enumerate(edges):
        incident[left] += triangle_count[index]
        incident[right] += triangle_count[index]
        incident_count[left] += 1
        incident_count[right] += 1
    density = np.divide(incident, np.maximum(incident_count, 1))

    quintiles = np.quantile(degree, [0.2, 0.4, 0.6, 0.8])
    degree_bin = np.searchsorted(quintiles, degree, side="right")
    by_bin = [np.flatnonzero(degree_bin == b) for b in range(5)]

    rng = np.random.default_rng(SEED)
    positives = np.zeros((n_items, POSITIVES), dtype=np.int64)
    negatives = np.zeros((n_items, NEGATIVES), dtype=np.int64)
    usable = np.zeros(n_items, dtype=bool)
    for item in range(n_items):
        neighbours = sorted(neighbour[item])
        if not neighbours:
            continue
        chosen = rng.choice(neighbours, size=min(POSITIVES, len(neighbours)), replace=False)
        positives[item, :len(chosen)] = chosen
        if len(chosen) < POSITIVES:
            positives[item, len(chosen):] = chosen[0]
        pool = by_bin[degree_bin[item]]
        pool = pool[pool != item]
        pool = pool[~np.isin(pool, neighbours)]
        if pool.size == 0:
            continue
        negatives[item] = rng.choice(pool, size=NEGATIVES, replace=False)
        usable[item] = True

    euclidean = q[:, :EUCLIDEAN_DIM]
    ball = expmap0(q[:, EUCLIDEAN_DIM:], 1.0)
    rows = np.flatnonzero(usable)
    index_positive = torch.from_numpy(positives[rows])
    index_negative = torch.from_numpy(negatives[rows])
    curves = {f"{alpha:.1f}": np.full(n_items, np.nan) for alpha in ALPHAS}
    for start in range(0, rows.size, CHUNK):
        stop = min(start + CHUNK, rows.size)
        batch = rows[start:stop]
        d_e = torch.cdist(euclidean[batch], euclidean).pow(2)
        d_h = hyp_sq(ball[batch].unsqueeze(1), ball.unsqueeze(0), 1.0) / 4.0
        for alpha in ALPHAS:
            if alpha == 0.0:
                distance = d_h
            elif alpha == 1.0:
                distance = d_e
            else:
                distance = alpha * d_e + (1.0 - alpha) * d_h
            curves[f"{alpha:.1f}"][batch] = item_auc(
                distance, index_positive[start:stop], index_negative[start:stop])
        if (start // CHUNK) % 40 == 0:
            print(f"  scored {stop}/{rows.size}", flush=True)

    # Cells: enough usable items, and a structural density.
    cell_ids = sorted(set(cells[rows].tolist()))
    report_cells = []
    for cell in cell_ids:
        mask = (cells == cell) & usable
        if mask.sum() < MIN_ITEMS_PER_CELL:
            continue
        per_alpha = {key: float(np.nanmean(values[mask])) for key, values in curves.items()}
        best = max(per_alpha, key=lambda key: per_alpha[key])
        report_cells.append({
            "cell": int(cell), "items": int(mask.sum()),
            "density": float(density[mask].mean()),
            "alpha_auc": per_alpha, "best_alpha": float(best),
            "best_auc": per_alpha[best],
        })
    print(f"cells kept: {len(report_cells)}")

    # The claim's test: do denser cells prefer a different bias?  Correlate the
    # cell density with the cell's alpha curve shape (auc(0.2) - auc(0.8)) and
    # check it against a permutation of the densities across cells.
    def contrast(cell):
        return cell["alpha_auc"]["0.2"] - cell["alpha_auc"]["0.8"]

    densities = np.array([cell["density"] for cell in report_cells])
    contrasts = np.array([contrast(cell) for cell in report_cells])
    order_density = np.argsort(np.argsort(densities))
    order_contrast = np.argsort(np.argsort(contrasts))
    spearman = float(np.corrcoef(order_density, order_contrast)[0, 1])
    null = []
    generators = np.random.default_rng(SEED + 1)
    for _ in range(PERMUTATIONS):
        permuted = generators.permutation(order_contrast)
        null.append(float(np.corrcoef(order_density, permuted)[0, 1]))
    null = np.array(null)

    # Density quintiles of cells: their pooled alpha curves.
    cell_density = densities
    cuts = np.quantile(cell_density, [0.2, 0.4, 0.6, 0.8])
    cell_group = np.searchsorted(cuts, cell_density, side="right")
    quintile_curve = {}
    for group in range(5):
        members = [cell for cell, flag in zip(report_cells, cell_group == group) if flag]
        if not members:
            continue
        quintile_curve[str(group)] = {
            key: float(np.mean([cell["alpha_auc"][key] for cell in members]))
            for key in curves}
        quintile_curve[str(group)]["cells"] = len(members)
        quintile_curve[str(group)]["items"] = int(sum(cell["items"] for cell in members))

    # Paired bootstrap over items inside the two extreme cell groups.
    extreme = {}
    for group in (0, 4):
        members = [cell["cell"] for cell, flag in zip(report_cells, cell_group == group) if flag]
        mask = np.isin(cells, members) & usable
        extreme[str(group)] = mask
    sparse_hits = curves["0.2"][extreme["0"]] - curves["0.8"][extreme["0"]]
    dense_hits = curves["0.2"][extreme["4"]] - curves["0.8"][extreme["4"]]
    weights_sparse = generators.poisson(1.0, size=(BOOTSTRAP, sparse_hits.size))
    weights_dense = generators.poisson(1.0, size=(BOOTSTRAP, dense_hits.size))
    boot = ((weights_sparse @ sparse_hits) / sparse_hits.size
            - (weights_dense @ dense_hits) / dense_hits.size)

    # The item-level bootstrap treats items as independent, but items inside one
    # code cell share a code and are correlated.  The honest interval resamples
    # *cells*: the item-level CI is overconfident and is kept only as a diagnostic.
    cell_delta = np.array([cell["alpha_auc"]["0.2"] - cell["alpha_auc"]["0.8"]
                           for cell in report_cells])
    cell_size = np.array([cell["items"] for cell in report_cells])
    cell_boot = []
    for _ in range(BOOTSTRAP):
        picks = []
        for group in (0, 4):
            members = np.flatnonzero(cell_group == group)
            if members.size == 0:
                continue
            drawn = generators.choice(members, size=members.size, replace=True)
            picks.append(np.average(cell_delta[drawn], weights=cell_size[drawn]))
        if len(picks) == 2:
            cell_boot.append(picks[0] - picks[1])
    cell_boot = np.array(cell_boot)
    cluster_interaction = {
        "interaction": float(np.mean([cell_delta[cell_group == 0].mean(),
                                      ]) - cell_delta[cell_group == 4].mean()),
        "ci95": [float(np.quantile(cell_boot, 0.025)),
                 float(np.quantile(cell_boot, 0.975))],
        "share_positive": float((cell_boot > 0).mean()),
        "cells_sparse": int((cell_group == 0).sum()),
        "cells_dense": int((cell_group == 4).sum()),
    }

    report = {
        "schema": "genrec.cell_geometry_preference.v1",
        "n_items": n_items,
        "n_edges": int(len(edges)),
        "alphas": list(ALPHAS),
        "min_items_per_cell": MIN_ITEMS_PER_CELL,
        "cells": report_cells,
        "density_contrast": {
            "definition": "cell mean(density) vs cell mean alpha-AUC(0.2)-AUC(0.8)",
            "spearman": spearman,
            "permutation_mean": float(null.mean()),
            "permutation_ci95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
            "p_two_sided": float((np.abs(null) >= abs(spearman)).mean()),
            "permutations": PERMUTATIONS,
        },
        "cell_density_quintiles": quintile_curve,
        "extreme_group_interaction": {
            "sparse_delta": float(sparse_hits.mean()),
            "dense_delta": float(dense_hits.mean()),
            "interaction": float(sparse_hits.mean() - dense_hits.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "sparse_items": int(sparse_hits.size), "dense_items": int(dense_hits.size),
            "note": "item-level interval; items within a cell are not independent",
            "cell_level": cluster_interaction,
        },
        "provenance": {
            "vectors": str(VECTORS.relative_to(REPO)), "vectors_sha256": sha256_of(VECTORS),
            "sid_sha256": sha256_of(SID_JSON),
            "train_parquet_sha256": sha256_of(TRAIN),
        },
    }
    out = HERE / "cell_geometry_preference.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"cells={len(report_cells)} spearman(density, alpha-contrast)={spearman:+.4f} "
          f"p={report['density_contrast']['p_two_sided']:.3f}")
    print("extreme interaction:", report["extreme_group_interaction"])
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
