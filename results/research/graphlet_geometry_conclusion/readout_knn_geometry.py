"""Which geometry puts real behaviour edges into the top-k neighbourhood?

The recall-paired probe and the per-edge AUC probe disagree on how much a
graphlet class cares about the geometric bias.  This probe uses the repository's
own structural criterion instead of a pairwise AUC: for a fixed SID (F0's
exported tokens) it builds the global k-nearest-neighbour graph under the mixed
distance

    d_alpha^2 = alpha * d_E^2 + (1 - alpha) * d_H^2 / 4,

and measures, per behaviour-edge graphlet class, how many of that class's real
behaviour edges survive in the graph (`recovery`).  Nothing is retrained: the
tokens are F0's exported SID, and only the readout geometry moves.

The two questions:

1. is there one `alpha` that maximises overall recovery (mixed curvature beats
   the pure halves), and
2. do the graphlet classes have *different* best `alpha` -- measured on one half
   of the behaviour edges and confirmed on the other half.

The second is the claim under test.  A shared optimum is the null; a class whose
best `alpha` moves and keeps its gain out of sample is a real geometric bias.
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
K = 10
CHUNK = 256
BOOTSTRAP = 1000
SEED = 42
CLASSES = {
    "triangles_0": (0, 0),
    "triangles_1_2": (1, 2),
    "triangles_3_5": (3, 5),
    "triangles_6plus": (6, 10 ** 9),
}


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


def hyperbolic_squared_matrix(ball: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Squared Poincare distance of a row block to every item, curvature -1.

    ``cosh d = 1 + 2|u-v|^2 / ((1-|u|^2)(1-|v|^2))``; the rank order of ``d`` and
    ``d^2`` is the same, and this form avoids the (block, n, dim) intermediates
    that the elementwise Mobius path allocates.
    """
    norm2 = ball.pow(2).sum(-1)
    denominator = (1.0 - norm2).clamp_min(1e-12)
    difference = torch.cdist(ball[start:stop], ball).pow(2)
    argument = 1.0 + 2.0 * difference / (denominator[start:stop, None] * denominator[None, :])
    return torch.acosh(argument.clamp_min(1.0 + 1e-9)).pow(2) / 4.0


def build_graphs(euclidean: torch.Tensor, ball: torch.Tensor, hyp_sq,
                 alphas) -> dict:
    """One symmetrised k-NN graph per alpha, from a single pass over the items."""
    n_items = euclidean.shape[0]
    graphs = {alpha: set() for alpha in alphas}
    rows = torch.arange(n_items)
    for start in range(0, n_items, CHUNK):
        stop = min(start + CHUNK, n_items)
        d_e = torch.cdist(euclidean[start:stop], euclidean).pow(2)
        d_h = hyperbolic_squared_matrix(ball, start, stop)
        for alpha in alphas:
            if alpha == 0.0:
                distance = d_h
            elif alpha == 1.0:
                distance = d_e
            else:
                distance = alpha * d_e + (1.0 - alpha) * d_h
            distance[rows[:stop - start], rows[start:stop]] = float("inf")
            index = torch.topk(distance, K, dim=1, largest=False).indices.cpu().tolist()
            graph = graphs[alpha]
            for offset, row in enumerate(index):
                left = start + offset
                for right in row:
                    graph.add((left, right) if left < right else (right, left))
        print(f"  scored rows {stop}/{n_items}", flush=True)
    return graphs


def check_distance_formula(ball, hyp_sq, samples=8):
    """The closed form must agree with the arm's own Mobius implementation."""
    rng = np.random.default_rng(SEED)
    left = rng.integers(0, ball.shape[0], size=samples)
    right = rng.integers(0, ball.shape[0], size=samples)
    elementwise = hyp_sq(ball[left], ball[right], 1.0) / 4.0
    norm2 = ball.pow(2).sum(-1)
    difference = (ball[left] - ball[right]).pow(2).sum(-1)
    closed = torch.acosh((1 + 2 * difference / ((1 - norm2[left]) * (1 - norm2[right])))
                         .clamp_min(1 + 1e-9)).pow(2) / 4.0
    if not torch.allclose(elementwise, closed, rtol=1e-4, atol=1e-6):
        raise SystemExit(f"distance formula disagrees: {elementwise.tolist()} vs {closed.tolist()}")


def main():
    expmap0, hyp_sq, sphere_expmap0, sphere_sq = load_contracts()
    payload = np.load(VECTORS)
    q = torch.from_numpy(payload["q"].astype(np.float32)).cuda()
    n_items = q.shape[0]
    edges = load_graph()
    print(f"items={n_items} behaviour edges={len(edges)}")

    adjacency = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_items, n_items)
    ).tocsr()
    adjacency = (adjacency + adjacency.T).tocsr()
    triangle_count = np.rint(np.asarray(
        (adjacency @ adjacency)[edges[:, 0], edges[:, 1]]).ravel()).astype(np.int64)
    print(f"triangle multiplicity: mean={triangle_count.mean():.2f} "
          f"zero-share={(triangle_count == 0).mean():.3f}")

    # Fit / evaluate split by a deterministic hash of the edge's endpoints.
    split = np.array([int(hashlib.sha256(f"{int(a)}:{int(b)}".encode()).hexdigest()[:8], 16) % 2
                      for a, b in edges], dtype=bool)

    euclidean = q[:, :EUCLIDEAN_DIM]
    ball = expmap0(q[:, EUCLIDEAN_DIM:], 1.0)
    check_distance_formula(ball.cpu(), hyp_sq)
    print("distance formula agrees with the arm's Mobius implementation", flush=True)

    edge_list = [(int(a), int(b)) for a, b in edges]
    class_mask = {}
    for name, (low, high) in CLASSES.items():
        class_mask[name] = (triangle_count >= low) & (triangle_count <= high)

    graphs = build_graphs(euclidean, ball, hyp_sq, ALPHAS)
    recovery = {}
    for alpha in ALPHAS:
        graph = graphs[alpha]
        row = {}
        for name, mask in class_mask.items():
            hit = np.array([(int(edges[i][0]), int(edges[i][1])) in graph
                            for i in np.flatnonzero(mask)], dtype=np.float64)
            row[name] = {"all": float(hit.mean()), "n": int(hit.size)}
            row[name]["fit"] = float(hit[split[mask]].mean())
            row[name]["eval"] = float(hit[~split[mask]].mean())
            row[name]["_hits"] = hit
        overall = np.array([(int(a), int(b)) in graph for a, b in edge_list],
                           dtype=np.float64)
        row["all_edges"] = {"all": float(overall.mean()), "n": int(overall.size)}
        row["_overall"] = overall
        recovery[f"{alpha:.1f}"] = row
        print(f"alpha={alpha:.1f}  overall={overall.mean():.5f}  "
              + "  ".join(f"{name}={row[name]['all']:.5f}" for name in CLASSES), flush=True)

    report = {
        "schema": "genrec.readout_knn_geometry.v1",
        "n_items": n_items,
        "n_edges": int(len(edges)),
        "k": K,
        "alphas": list(ALPHAS),
        "classes": list(CLASSES),
        "fit_edges": int(split.sum()),
        "eval_edges": int((~split).sum()),
        "recovery": {},
        "provenance": {
            "vectors": str(VECTORS.relative_to(REPO)), "vectors_sha256": sha256_of(VECTORS),
            "sid_sha256": sha256_of(SID_JSON),
            "train_parquet_sha256": sha256_of(TRAIN),
            "mechanism": "fixed F0 SID; only the readout distance alpha moves",
        },
    }
    for alpha in ALPHAS:
        key = f"{alpha:.1f}"
        report["recovery"][key] = {
            name: {k: v for k, v in row.items() if not k.startswith("_")}
            for name, row in recovery[key].items() if not name.startswith("_")}

    # Per-class best alpha on the fit half, scored on the eval half.
    rng = np.random.default_rng(SEED)
    report["per_class"] = {}
    overall_key = max((f"{a:.1f}" for a in ALPHAS),
                      key=lambda key: recovery[key]["all_edges"]["all"])
    report["overall_best_alpha"] = float(overall_key)
    for name in CLASSES:
        curve_fit = {f"{a:.1f}": recovery[f"{a:.1f}"][name]["fit"] for a in ALPHAS}
        curve_all = {f"{a:.1f}": recovery[f"{a:.1f}"][name]["all"] for a in ALPHAS}
        best = max(curve_fit, key=lambda key: curve_fit[key])
        hits = recovery[best][name]["_hits"]
        hit_overall = recovery[overall_key][name]["_hits"]
        eval_mask = ~split[class_mask[name]]
        delta = (hits[~split[class_mask[name]]] - hit_overall[~split[class_mask[name]]])
        weights = rng.poisson(1.0, size=(BOOTSTRAP, delta.size))
        boot = (weights @ delta) / delta.size
        report["per_class"][name] = {
            "fit_best_alpha": float(best),
            "fit_curve": curve_fit,
            "all_curve": curve_all,
            "eval_auc_at_fit_best": float(hits[eval_mask].mean()),
            "eval_auc_at_overall_best": float(hit_overall[eval_mask].mean()),
            "eval_gain": float(delta.mean()),
            "eval_gain_ci95": [float(np.quantile(boot, 0.025)),
                               float(np.quantile(boot, 0.975))],
        }
        print(f"{name:16s} fit_alpha={best} eval_gain={delta.mean():+.5f} "
              f"CI=[{np.quantile(boot,0.025):+.5f},{np.quantile(boot,0.975):+.5f}]")

    out = HERE / "readout_knn_geometry.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
