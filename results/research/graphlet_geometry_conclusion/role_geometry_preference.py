"""Which geometry does each frozen ORCA graphlet role actually prefer?

The role-routing arm of issue #215 assigns the six ORCA roles to the three
branches by hand (`role_geometry.ROLE_SPACES`).  This script measures the
assignment instead of asserting it: for every item it asks whether the
Euclidean, hyperbolic, or spherical branch ranks the item's *training* behaviour
neighbours above degree-matched non-neighbours, and then aggregates that per
role.

Inputs are frozen and train-only:

* `results/research/tiger_cluster_graph/mixed_vectors.npz` -- the F0 arm's
  encoder output `z` and its quantised readout `q` (32 = 16 Euclidean + 16
  tangent), produced by `mixed_vectors.py` and pinned to the arm's checkpoint.
* `results/research/issue215/orca_roles_frozen.csv` -- the frozen role table,
  sha256 checked against the value the arms load.
* `results/stage0_build_parquet/train.parquet` -- the behaviour graph, built
  exactly as the offline reports build it (consecutive history pairs plus
  history-tail to target, undirected: 315,391 edges).

The distance functions are the arms' own `poincare_contract` / `spherical_contract`,
imported rather than re-derived.  The geometric bias is the F0 cost's `alpha`:
`d = alpha * d_E^2 + (1 - alpha) * d_H^2 / 4`, so `alpha = 1` is pure Euclidean
and `alpha = 0` is pure hyperbolic.

The heterogeneity test is a role-label permutation: item-level AUCs are fixed by
the geometry, so permuting the role labels and recomputing the per-role best
`alpha` asks whether the frozen roles group items with different geometric
preferences more than chance would.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VECTORS = REPO / "results" / "research" / "tiger_cluster_graph" / "mixed_vectors.npz"
SID_JSON = REPO / "results" / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0" / "item_sids.json"
ROLE_CSV = REPO / "results" / "research" / "issue215" / "orca_roles_frozen.csv"
ROLE_SHA = "c7a29e6559aed1385c5ce514e506be22d88bf6e85eb61120b6dbf69375ea251c"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
TREE = REPO / "analysis_sid_recall" / "r2" / "19" / "TIGER_mixed_rq_F0"
#: The spherical branch was authored with the role arm; the three-branch cost is
#: the only place the F0 recipe does not carry it, and the file is not duplicated
#: into every arm.
SPHERE_TREE = REPO / "analysis_sid_recall" / "r2" / "70" / "TIGER_role_geom_rq"

EUCLIDEAN_DIM = 16
CURVATURE = 1.0
ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))
POSITIVES = 20
NEGATIVES = 20
CHUNK = 64
SAMPLE_SEED = 42
PERMUTATIONS = 500

sys.path.insert(0, str(TREE))


def load_contracts():
    """The arms' own geometry, imported as packages so the relative imports hold."""
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


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_roles(n_items: int) -> np.ndarray:
    """The frozen role of every item; -1 for the 31 items that are not training targets."""
    digest = sha256_of(ROLE_CSV)
    if digest != ROLE_SHA:
        raise SystemExit(f"role table sha256 {digest} != frozen {ROLE_SHA}")
    table = pd.read_csv(ROLE_CSV)
    if list(table.columns) != ["item_id", "role"]:
        raise SystemExit(f"unexpected role table columns {list(table.columns)}")
    roles = np.full(n_items, -1, dtype=np.int64)
    roles[table["item_id"].to_numpy(dtype=np.int64)] = table["role"].to_numpy(dtype=np.int64)
    return roles


def load_graph(n_items: int):
    train = pd.read_parquet(TRAIN)
    pairs = set()
    for history, target in zip(train["history"].to_numpy(), train["target"].to_numpy()):
        history = list(history)
        for left, right in zip(history, history[1:]):
            pairs.add((left, right) if left < right else (right, left))
        if history:
            tail = history[-1]
            pairs.add((tail, target) if tail < target else (target, tail))
    pairs = np.array(sorted(pairs), dtype=np.int64)
    if pairs.shape[0] != 315391:
        raise SystemExit(f"behaviour graph has {pairs.shape[0]} edges, expected 315391")
    if pairs.max() >= n_items:
        raise SystemExit("behaviour graph references an item outside the vector file")
    neighbour = [set() for _ in range(n_items)]
    for left, right in pairs:
        neighbour[int(left)].add(int(right))
        neighbour[int(right)].add(int(left))
    degree = np.array([len(row) for row in neighbour], dtype=np.int64)
    return pairs, neighbour, degree


def branch_distances(query: torch.Tensor, all_items: torch.Tensor,
                     expmap0, hyp_sq, sphere_expmap0, sphere_sq):
    """Distances from every query row to every item, one matrix per branch."""
    euclidean_q = query[:, :EUCLIDEAN_DIM]
    euclidean_a = all_items[:, :EUCLIDEAN_DIM]
    tangent_q = query[:, EUCLIDEAN_DIM:]
    tangent_a = all_items[:, EUCLIDEAN_DIM:]
    ball_q, ball_a = expmap0(tangent_q, CURVATURE), expmap0(tangent_a, CURVATURE)
    sphere_q, sphere_a = sphere_expmap0(tangent_q), sphere_expmap0(tangent_a)
    d_e = torch.cdist(euclidean_q, euclidean_a).pow(2)
    d_h = hyp_sq(ball_q.unsqueeze(1), ball_a.unsqueeze(0), CURVATURE)
    d_s = sphere_sq(sphere_q.unsqueeze(1), sphere_a.unsqueeze(0))
    return d_e, d_h / 4.0, d_s


def item_auc(distance: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> np.ndarray:
    """AUC per row: P(positive distance < negative distance), ties count one half."""
    positive = distance.gather(1, positives)
    negative = distance.gather(1, negatives)
    compare = positive.unsqueeze(2) - negative.unsqueeze(1)
    return (compare < 0).float().add((compare == 0).float() * 0.5).mean(dim=(1, 2)).numpy()


def main():
    expmap0, hyp_sq, sphere_expmap0, sphere_sq = load_contracts()
    payload = np.load(VECTORS)
    q = torch.from_numpy(payload["q"].astype(np.float32))
    n_items = q.shape[0]
    roles = load_roles(n_items)
    print(f"items={n_items} labeled={int((roles >= 0).sum())} "
          f"roles={np.bincount(roles[roles >= 0]).tolist()}")
    pairs, neighbour, degree = load_graph(n_items)
    print(f"behaviour edges={len(pairs)}")

    # Degree quintiles: a negative must look like a neighbour in popularity.
    edges_sorted = np.sort(degree)
    cuts = np.quantile(edges_sorted, [0.2, 0.4, 0.6, 0.8])
    degree_bin = np.searchsorted(cuts, degree, side="right")
    by_bin = [np.flatnonzero(degree_bin == b) for b in range(5)]

    rng = np.random.default_rng(SAMPLE_SEED)
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
        pool = pool[(pool != item)]
        pool = pool[~np.isin(pool, neighbours)]
        if pool.size == 0:
            continue
        negatives[item] = rng.choice(pool, size=NEGATIVES, replace=False)
        usable[item] = True
    print(f"items with positives and negatives: {int(usable.sum())}")

    positive_rows = np.flatnonzero(usable)
    auc = {name: np.full(n_items, np.nan, dtype=np.float64)
           for name in [f"alpha={a}" for a in ALPHAS] + ["euclidean", "hyperbolic", "spherical"]}
    index_positive = torch.from_numpy(positives[positive_rows])
    index_negative = torch.from_numpy(negatives[positive_rows])

    for start in range(0, positive_rows.size, CHUNK):
        stop = min(start + CHUNK, positive_rows.size)
        rows = positive_rows[start:stop]
        d_e, d_h, d_s = branch_distances(q[rows], q, expmap0, hyp_sq, sphere_expmap0, sphere_sq)
        pos = index_positive[start:stop]
        neg = index_negative[start:stop]
        auc["euclidean"][rows] = item_auc(d_e, pos, neg)
        auc["hyperbolic"][rows] = item_auc(d_h, pos, neg)
        auc["spherical"][rows] = item_auc(d_s, pos, neg)
        for alpha in ALPHAS:
            mixed = alpha * d_e + (1.0 - alpha) * d_h
            auc[f"alpha={alpha}"][rows] = item_auc(mixed, pos, neg)
        if (start // CHUNK) % 40 == 0:
            print(f"  scored {stop}/{positive_rows.size}", flush=True)

    names = [f"alpha={a}" for a in ALPHAS]
    report = {"roles": {}, "alphas": list(ALPHAS), "n_items": n_items,
                "n_edges": int(len(pairs)), "usable_items": int(usable.sum()),
                "sample_seed": SAMPLE_SEED, "positives": POSITIVES,
                "negatives": NEGATIVES, "curvature": CURVATURE}
    for name in names + ["euclidean", "hyperbolic", "spherical"]:
        report.setdefault("overall", {})[name] = float(np.nanmean(auc[name]))

    overall_best = max(names, key=lambda k: report["overall"][k])
    overall_alpha = float(overall_best.split("=")[1])
    report["overall_best"] = {"name": overall_best, "alpha": overall_alpha,
                                "auc": report["overall"][overall_best]}
    print(f"overall best {overall_best} auc={report['overall'][overall_best]:.5f}")

    role_of_item = roles
    for role in range(int(roles.max()) + 1):
        mask = (role_of_item == role) & usable
        if mask.sum() < 50:
            report["roles"][str(role)] = {"items": int(mask.sum()), "skipped": True}
            continue
        per_geometry = {name: float(np.nanmean(auc[name][mask])) for name in auc}
        best = max(names, key=lambda k: per_geometry[k])
        alpha_r = float(best.split("=")[1])
        gain = per_geometry[best] - per_geometry[overall_best]
        report["roles"][str(role)] = {
            "items": int(mask.sum()),
            "auc": per_geometry,
            "best": {"name": best, "alpha": alpha_r, "auc": per_geometry[best]},
            "gain_over_global_alpha": float(gain),
        }
        print(f"role {role} n={int(mask.sum()):5d} best={best:9s} "
              f"auc={per_geometry[best]:.5f} global_alpha_auc={per_geometry[overall_best]:.5f} "
              f"gain={gain:+.5f} | E={per_geometry['euclidean']:.5f} "
              f"H={per_geometry['hyperbolic']:.5f} S={per_geometry['spherical']:.5f}")

    # Heterogeneity: permute the role labels, keep the per-item AUCs.
    observed = [report["roles"][str(r)]["best"]["alpha"]
                for r in range(int(roles.max()) + 1)
                if not report["roles"][str(r)].get("skipped")]
    observed_spread = float(max(observed) - min(observed))
    rows = np.flatnonzero(usable)
    permutation_alphas = []
    permutation_spread = []
    ratios = np.stack([auc[name][rows] for name in names], axis=1)
    role_rows = roles[rows]
    sizes = [(role_rows == r).sum() for r in range(int(roles.max()) + 1)]
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(role_rows)
        best_alphas = []
        for role in range(int(roles.max()) + 1):
            mask = shuffled == role
            if mask.sum() < 50:
                continue
            means = ratios[mask].mean(axis=0)
            best_alphas.append(float(ALPHAS[int(np.argmax(means))]))
        permutation_alphas.append(best_alphas)
        permutation_spread.append(max(best_alphas) - min(best_alphas))
    permutation_spread = np.array(permutation_spread)
    report["heterogeneity"] = {
        "observed_best_alphas": observed,
        "observed_spread": observed_spread,
        "permutation_spread_mean": float(permutation_spread.mean()),
        "permutation_spread_ci95": [float(np.quantile(permutation_spread, 0.025)),
                                    float(np.quantile(permutation_spread, 0.975))],
        "p_spread_ge_observed": float((permutation_spread >= observed_spread).mean()),
        "permutations": PERMUTATIONS,
    }
    print(f"heterogeneity: observed spread {observed_spread:.2f}, "
          f"permutation mean {permutation_spread.mean():.2f}, "
          f"p={(permutation_spread >= observed_spread).mean():.3f}")

    report["provenance"] = {
        "vectors": str(VECTORS.relative_to(REPO)),
        "vectors_sha256": sha256_of(VECTORS),
        "sid_sha256": sha256_of(SID_JSON),
        "role_table": str(ROLE_CSV.relative_to(REPO)),
        "role_table_sha256": ROLE_SHA,
        "train_parquet": str(TRAIN.relative_to(REPO)),
        "train_parquet_sha256": sha256_of(TRAIN),
    }
    out = HERE / "role_geometry_preference.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
