"""Does the best geometric bias depend on the edge's graphlet class?

The ORCA-role probe found one shared optimum; this probe asks the same question
with the graphlet axis the offline reports already showed to be predictive --
the number of triangles an edge sits in (dense K3/K4 structure) and the degree of
its larger endpoint (fan / star structure).

For every TRAIN behaviour edge `(u, v)` the script compares the distance in the
F0 representation of the positive pair against a degree-matched non-edge, so an
edge's AUC is `P(d(positive) < d(matched non-edge))`.  The geometric bias is the
F0 cost's `alpha` between the Euclidean and hyperbolic halves plus an
independent curvature grid on the hyperbolic branch.  Everything is reduced to
per-edge hit indicators, so an AUC is a mean and a paired bootstrap is a
weighted mean of per-edge differences.

Everything is train-only: the behaviour graph is built from `train.parquet`
exactly as the offline reports build it, and the vectors are the F0 readout
(`mixed_vectors.npz`, pinned to the arm's checkpoint and SID).
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
SPHERE_TREE = REPO / "analysis_sid_recall" / "r2" / "70" / "TIGER_role_geom_rq"

EUCLIDEAN_DIM = 16
ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))
CURVATURES = (0.25, 0.5, 1.0, 2.0, 4.0)
TRIANGLE_BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10 ** 9)]
TRIANGLE_LABELS = ["0", "1", "2", "3-5", "6+"]
SAMPLE_PER_BIN = 6000
BOOTSTRAP = 1000
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
    sphere_spec = importlib.util.spec_from_file_location(
        "role_model", SPHERE_TREE / "model" / "__init__.py",
        submodule_search_locations=[str(SPHERE_TREE / "model")],
    )
    sphere_package = importlib.util.module_from_spec(sphere_spec)
    sys.modules["role_model"] = sphere_package
    sphere_spec.loader.exec_module(sphere_package)
    from f0_model.poincare_contract import expmap0, squared_distance as hyp_sq
    from role_model.spherical_contract import sphere_expmap0, sphere_squared_distance
    return expmap0, hyp_sq, sphere_expmap0, sphere_squared_distance


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


def branch_distances(edges, negatives, q, expmap0, hyp_sq, sphere_expmap0, sphere_sq):
    """The three branches' squared distances for the positives and the negatives."""
    euclidean = q[:, :EUCLIDEAN_DIM]
    tangent = q[:, EUCLIDEAN_DIM:]
    ball = expmap0(tangent, 1.0)
    sphere = sphere_expmap0(tangent)

    def three(pairs):
        left, right = pairs[:, 0], pairs[:, 1]
        d_e = (euclidean[left] - euclidean[right]).pow(2).sum(-1)
        d_h = hyp_sq(ball[left], ball[right], 1.0) / 4.0
        d_s = sphere_sq(sphere[left], sphere[right])
        return d_e.numpy(), d_h.numpy(), d_s.numpy()

    return three(edges), three(negatives)


def curvature_squared(pairs, ball, hyp_sq, curvature):
    """`d_c^2 / 4` at an arbitrary curvature via the scaling identity."""
    left, right = pairs[:, 0], pairs[:, 1]
    root = curvature ** 0.5
    return hyp_sq(ball[left] * root, ball[right] * root, 1.0).numpy() / curvature / 4.0


def hits(positive, negative):
    """Per-edge indicator of `positive < negative`, ties counted one half."""
    compare = positive - negative
    return (compare < 0).astype(np.float64) + 0.5 * (compare == 0)


def matched_negatives(edges, adjacency_sets, degree_bin, by_bin, rng):
    """One degree-matched non-neighbour per edge, drawn on the target endpoint."""
    negatives = np.empty_like(edges)
    for index, (left, right) in enumerate(edges):
        pool = by_bin[degree_bin[right]]
        neighbours = adjacency_sets[right]
        candidate = int(pool[rng.integers(pool.size)])
        for _ in range(20):
            if candidate == left or candidate in neighbours:
                candidate = int(pool[rng.integers(pool.size)])
                continue
            break
        negatives[index] = (left, candidate)
    return negatives


def main():
    expmap0, hyp_sq, sphere_expmap0, sphere_sq = load_contracts()
    payload = np.load(VECTORS)
    q = torch.from_numpy(payload["q"].astype(np.float32))
    n_items = q.shape[0]
    edges = load_graph()
    print(f"items={n_items} behaviour edges={len(edges)}")

    adjacency = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_items, n_items)
    ).tocsr()
    adjacency = (adjacency + adjacency.T).tocsr()
    triangle_count = np.rint(np.asarray(
        (adjacency @ adjacency)[edges[:, 0], edges[:, 1]]).ravel()).astype(np.int64)
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int64)
    adjacency_sets = [set(adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]].tolist())
                      for i in range(n_items)]
    print(f"triangle multiplicity: min={triangle_count.min()} max={triangle_count.max()} "
          f"mean={triangle_count.mean():.2f}")

    quintiles = np.quantile(degree, [0.2, 0.4, 0.6, 0.8])
    degree_bin = np.searchsorted(quintiles, degree, side="right")
    by_bin = [np.flatnonzero(degree_bin == b) for b in range(5)]

    rng = np.random.default_rng(SEED)
    negatives = matched_negatives(edges, adjacency_sets, degree_bin, by_bin, rng)

    (d_e_pos, d_h_pos, d_s_pos), (d_e_neg, d_h_neg, d_s_neg) = branch_distances(
        edges, negatives, q, expmap0, hyp_sq, sphere_expmap0, sphere_sq)
    ball = expmap0(q[:, EUCLIDEAN_DIM:], 1.0)
    curvature_pos = {c: curvature_squared(edges, ball, hyp_sq, c) for c in CURVATURES}
    curvature_neg = {c: curvature_squared(negatives, ball, hyp_sq, c) for c in CURVATURES}

    branch_hits = {
        "euclidean": hits(d_e_pos, d_e_neg),
        "hyperbolic": hits(d_h_pos, d_h_neg),
        "spherical": hits(d_s_pos, d_s_neg),
    }
    alpha_hits = {}
    for alpha in ALPHAS:
        alpha_hits[f"{alpha:.1f}"] = hits(alpha * d_e_pos + (1 - alpha) * d_h_pos,
                                         alpha * d_e_neg + (1 - alpha) * d_h_neg)
    curvature_hits = {f"{c:.2f}": hits(curvature_pos[c], curvature_neg[c]) for c in CURVATURES}

    report = {
        "schema": "genrec.edge_class_geometry.v1",
        "n_items": n_items,
        "n_edges": int(len(edges)),
        "alphas": list(ALPHAS),
        "curvatures": list(CURVATURES),
        "triangle_bins": TRIANGLE_LABELS,
        "sample_per_bin": SAMPLE_PER_BIN,
        "bootstrap": BOOTSTRAP,
        "classes": {},
        "provenance": {
            "vectors": str(VECTORS.relative_to(REPO)), "vectors_sha256": sha256_of(VECTORS),
            "sid_sha256": sha256_of(SID_JSON),
            "train_parquet_sha256": sha256_of(TRAIN),
            "mechanism": "F0 readout q; alpha mixes Euclidean^2 and hyperbolic^2/4; c on the hyperbolic half",
        },
    }

    generators = np.random.default_rng(SEED + 1)

    def evaluate(mask: np.ndarray, name: str):
        rows = np.flatnonzero(mask)
        if rows.size < 200:
            return None
        if rows.size > SAMPLE_PER_BIN:
            rows = generators.choice(rows, size=SAMPLE_PER_BIN, replace=False)
        entry = {"edges": int(rows.size)}
        entry.update({branch: float(values[rows].mean()) for branch, values in branch_hits.items()})
        entry["alpha_auc"] = {key: float(values[rows].mean()) for key, values in alpha_hits.items()}
        entry["curvature_auc"] = {key: float(values[rows].mean())
                                  for key, values in curvature_hits.items()}
        best_alpha = max(entry["alpha_auc"], key=lambda key: entry["alpha_auc"][key])
        entry["best_alpha"] = float(best_alpha)
        entry["best_alpha_auc"] = entry["alpha_auc"][best_alpha]
        entry["best_curvature"] = float(max(entry["curvature_auc"],
                                            key=lambda key: entry["curvature_auc"][key]))

        # Paired edge bootstrap: Poisson(1) weights over the sampled edges.
        indicators = np.stack([alpha_hits[f"{alpha:.1f}"][rows] for alpha in ALPHAS], axis=1)
        weights = generators.poisson(1.0, size=(BOOTSTRAP, rows.size))
        means = (weights @ indicators) / rows.size       # [B, |alphas|]
        winners = np.argmax(means, axis=1)
        entry["alpha_bootstrap_share"] = {
            f"{ALPHAS[i]:.1f}": float((winners == i).mean()) for i in range(len(ALPHAS))}
        entry["auc_bootstrap_ci"] = {
            f"{ALPHAS[i]:.1f}": [float(np.quantile(means[:, i], 0.025)),
                                 float(np.quantile(means[:, i], 0.975))]
            for i in range(len(ALPHAS))}
        report["classes"][name] = entry
        print(f"{name:14s} n={rows.size:5d} E={entry['euclidean']:.5f} "
              f"H={entry['hyperbolic']:.5f} S={entry['spherical']:.5f} "
              f"best_alpha={entry['best_alpha']:.1f} auc={entry['best_alpha_auc']:.5f} "
              f"best_c={entry['best_curvature']:.2f} "
              f"alpha_share={max(entry['alpha_bootstrap_share'].items(), key=lambda kv: kv[1])}")
        return entry

    evaluate(np.ones(len(edges), dtype=bool), "all_edges")
    for (low, high), label in zip(TRIANGLE_BINS, TRIANGLE_LABELS):
        evaluate((triangle_count >= low) & (triangle_count <= high), f"triangles_{label}")
    evaluate(triangle_count >= 1, "triangles_1plus")

    # Two-group fit with a held-out check: the alpha of each group is chosen on
    # one half of its edges and scored on the other half.  A group's preference is
    # only reported as measured when it survives that split.
    split = np.array([int(hashlib.sha256(str(int(edge)).encode()).hexdigest()[:8], 16) % 2
                      for edge in range(len(edges))], dtype=bool)
    holdout = {}
    for name, mask in {"triangles_0": triangle_count == 0,
                       "triangles_1plus": triangle_count >= 1}.items():
        fit_rows = np.flatnonzero(mask & split)
        eval_rows = np.flatnonzero(mask & ~split)
        if fit_rows.size > SAMPLE_PER_BIN:
            fit_rows = generators.choice(fit_rows, size=SAMPLE_PER_BIN, replace=False)
        if eval_rows.size > SAMPLE_PER_BIN:
            eval_rows = generators.choice(eval_rows, size=SAMPLE_PER_BIN, replace=False)
        fit_curve = {f"{alpha:.1f}": float(alpha_hits[f"{alpha:.1f}"][fit_rows].mean())
                     for alpha in ALPHAS}
        best = max(fit_curve, key=lambda key: fit_curve[key])
        eval_curve = {f"{alpha:.1f}": float(alpha_hits[f"{alpha:.1f}"][eval_rows].mean())
                      for alpha in ALPHAS}
        holdout[name] = {
            "fit_edges": int(fit_rows.size), "eval_edges": int(eval_rows.size),
            "fit_best_alpha": float(best), "fit_curve": fit_curve,
            "eval_curve": eval_curve,
            "eval_auc_at_fit_best": eval_curve[best],
            "eval_auc_at_population": eval_curve["0.6"],
            "eval_gain": eval_curve[best] - eval_curve["0.6"],
        }
        print(f"holdout {name:16s} fit_alpha={best} eval_gain={holdout[name]['eval_gain']:+.5f}")
    report["two_group_holdout"] = holdout

    # The claim's own test: the *difference* in how much the two graphlet
    # classes like the hyperbolic-dominant bias, as one paired interaction.
    sparse_rows = np.flatnonzero(triangle_count == 0)
    dense_rows = np.flatnonzero(triangle_count >= 1)
    if sparse_rows.size > SAMPLE_PER_BIN:
        sparse_rows = generators.choice(sparse_rows, size=SAMPLE_PER_BIN, replace=False)
    if dense_rows.size > SAMPLE_PER_BIN:
        dense_rows = generators.choice(dense_rows, size=SAMPLE_PER_BIN, replace=False)
    for low, high, label in ((0.1, 0.6, "0.1_minus_0.6"), (0.1, 1.0, "0.1_minus_1.0")):
        sparse_delta = alpha_hits[f"{low:.1f}"][sparse_rows] - alpha_hits[f"{high:.1f}"][sparse_rows]
        dense_delta = alpha_hits[f"{low:.1f}"][dense_rows] - alpha_hits[f"{high:.1f}"][dense_rows]
        interaction = float(sparse_delta.mean() - dense_delta.mean())
        weights_sparse = generators.poisson(1.0, size=(BOOTSTRAP, sparse_rows.size))
        weights_dense = generators.poisson(1.0, size=(BOOTSTRAP, dense_rows.size))
        boot = ((weights_sparse @ sparse_delta) / sparse_rows.size
                - (weights_dense @ dense_delta) / dense_rows.size)
        report.setdefault("interaction", {})[label] = {
            "sparse_delta": float(sparse_delta.mean()),
            "dense_delta": float(dense_delta.mean()),
            "interaction": interaction,
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "share_positive": float((boot > 0).mean()),
            "sparse_edges": int(sparse_rows.size), "dense_edges": int(dense_rows.size),
        }
        print(f"interaction {label}: sparse_delta={sparse_delta.mean():+.5f} "
              f"dense_delta={dense_delta.mean():+.5f} interaction={interaction:+.5f} "
              f"CI=[{np.quantile(boot,0.025):+.5f},{np.quantile(boot,0.975):+.5f}]")

    hub = np.maximum(degree[edges[:, 0]], degree[edges[:, 1]])
    hub_bin = np.searchsorted(np.quantile(hub, [0.2, 0.4, 0.6, 0.8]), hub, side="right")
    for b in range(5):
        evaluate(hub_bin == b, f"hub_degree_q{b}")

    # Does the best alpha shift with density?  Paired contrast between the two
    # extreme triangle bins on the same alpha grid.
    sparse = np.flatnonzero(triangle_count == 0)
    dense = np.flatnonzero(triangle_count >= 6)
    contrast = {}
    for alpha in ALPHAS:
        key = f"{alpha:.1f}"
        contrast[key] = float(alpha_hits[key][dense].mean() - alpha_hits[key][sparse].mean())
    report["density_contrast"] = {
        "definition": "AUC(triangle>=6) - AUC(triangle==0), same alpha",
        "edges_sparse": int(sparse.size),
        "edges_dense": int(dense.size),
        "delta": contrast,
    }
    print("density contrast:", {key: round(value, 5) for key, value in contrast.items()})

    out = HERE / "edge_class_geometry.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
