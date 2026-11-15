"""Does the quantised representation still carry the behaviour graph's local topology?

The owner's construction: build G_b from behaviour, run ORCA on it; build G_q from the quantised
vectors (k nearest neighbours, k matched to the behaviour graph's mean degree); run the same ORCA on
it; compare.  Two degrees of preservation are reported separately:

    role       A is still a hub in G_q -- its degree rank and its orbit profile survive
    structure  the very items A was connected to in G_b are among its neighbours in G_q

Every arm of the 67-arm mirror is measured with its own checkpoint and SID file, so the quantised
vectors are the model's own, and each arm's metric is paired with its `test_recall@10`.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
QGRAPHS = OUT / "structure_graphs_all"
ORCA = REPO / "results" / "research" / "issue213" / "orca" / "orca"
ORBIT_B5 = REPO / "results" / "research" / "issue213" / "orca" / "node5.out"
ORBIT_B4 = REPO / "results" / "research" / "issue213" / "orca" / "node4.out"

spec = importlib.util.spec_from_file_location(
    "analysis_sid_recall_analysis", REPO / "analysis_sid_recall" / "analysis.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)

# Each direction takes 13 neighbours, so the union's mean degree is about 26 -- the behaviour graph's
# mean degree (315,391 edges over 24,556 items).  Taking 26 per direction doubled both the density and
# ORCA's cost, and made the two graphs unmatched.
K = 13
STARS = 400
CHUNK = 2048


def knn_edges(vectors, k):
    """Union of the directed k-NN lists, as the graph files ORCA reads."""
    n = vectors.shape[0]
    squared = np.einsum("ij,ij->i", vectors, vectors)
    neighbours = np.full((n, k), -1, dtype=np.int64)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        block = squared[start:stop, None] + squared[None, :] - 2.0 * vectors[start:stop] @ vectors.T
        np.fill_diagonal(block[:, start:stop], np.inf) if stop - start > 1 else None
        block[np.arange(stop - start), np.arange(start, stop)] = np.inf
        neighbours[start:stop] = np.argpartition(block, k, axis=1)[:, :k]
    edges = set()
    for node in range(n):
        for other in neighbours[node]:
            a, b = (node, int(other)) if node < other else (int(other), node)
            edges.add((a, b))
    return np.asarray(sorted(edges), dtype=np.int64)


def write_graph(path, n, edges):
    with path.open("w") as handle:
        handle.write(f"{n} {len(edges)}\n")
        for a, b in edges:
            handle.write(f"{a} {b}\n")


def orbits(in_path, out_path):
    subprocess.run([str(ORCA), "node", "5", str(in_path), str(out_path)], check=True,
                   capture_output=True)
    table = np.loadtxt(out_path, dtype=np.float64)
    return table if table.ndim == 2 else table.reshape(1, -1)


def spearman_rows(first, second):
    """Per-column Spearman over items, returned as a vector (nan where a column is constant)."""
    out = np.full(first.shape[1], np.nan)
    for column in range(first.shape[1]):
        a, b = first[:, column], second[:, column]
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            continue
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        denominator = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        if denominator:
            out[column] = float((ra * rb).sum() / denominator)
    return out


def cosine_rows(first, second):
    left = np.log1p(np.clip(first, 0, None))
    right = np.log1p(np.clip(second, 0, None))
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def sample_stars(adjacency, n, rng, count):
    stars = []
    nodes = np.array([node for node in range(n) if len(adjacency[node]) >= 4])
    tries = 0
    while len(stars) < count and tries < count * 50:
        tries += 1
        centre = int(rng.choice(nodes))
        neighbours = list(adjacency[centre])
        if len(neighbours) < 4:
            continue
        leaves = rng.choice(neighbours, size=4, replace=False)
        if any(second in adjacency[int(first)] for i, first in enumerate(leaves)
               for second in leaves[i + 1:]):
            continue
        stars.append((centre, [int(leaf) for leaf in leaves]))
    return stars


def rank_columns(matrix):
    """Average ranks per column, the input of a Spearman correlation matrix."""
    from scipy.stats import rankdata
    return np.apply_along_axis(rankdata, 0, matrix)


def gcd(first, second):
    """Graphlet Correlation Distance: Euclidean distance between the orbit correlation matrices.

    Both matrices are Spearman correlations across items of the ORCA orbit counts, so the distance
    compares the *dependence pattern* of the orbits -- two graphs can share every marginal count and
    still fail this; and it says nothing about which item kept which role.
    """
    keep = np.array([column for column in range(first.shape[1])
                     if first[:, column].std() > 0 and second[:, column].std() > 0])
    if keep.size < 2:
        return float("nan")
    first_ranks = rank_columns(first[:, keep])
    second_ranks = rank_columns(second[:, keep])
    with np.errstate(invalid="ignore"):
        first_corr = np.corrcoef(first_ranks.T)
        second_corr = np.corrcoef(second_ranks.T)
    mask = np.triu_indices(first_corr.shape[0], 1)
    return float(np.sqrt(((first_corr[mask] - second_corr[mask]) ** 2).sum()))


def deltacon(first_edges, second_edges, n, seeds=10, seed=42):
    """Approximate DeltaCon: distance between the square roots of the two affinities.

    The exact form needs S^-1 twice, and S is dense at 24,587 nodes.  The approximation samples a few
    columns of S^-1 with a sparse solve (DeltaCon's own fast variant does the same thing with a
    truncated series); the sampled columns are then compared entry by entry.
    """
    from scipy.sparse import coo_matrix, diags, identity
    from scipy.sparse.linalg import spsolve

    def affinity(edges):
        adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        adjacency = (adjacency + adjacency.T).tocsr().astype(np.float64)
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        epsilon = 1.0 / (1.0 + degree.max())
        return identity(n, format="csr") + epsilon ** 2 * diags(degree) - epsilon * adjacency

    first_affinity = affinity(first_edges)
    second_affinity = affinity(second_edges)
    rng = np.random.default_rng(seed)
    picks = rng.choice(n, size=min(seeds, n), replace=False)
    total = 0.0
    for pick in picks:
        target = np.zeros(n)
        target[pick] = 1.0
        left = spsolve(first_affinity, target)
        right = spsolve(second_affinity, target)
        total += float(((np.sqrt(np.clip(left, 0, None)) - np.sqrt(np.clip(right, 0, None))) ** 2).sum())
    return float(np.sqrt(total / len(picks)))


def edge_overlap(first_edges, second_edges):
    first = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in first_edges.tolist()}
    second = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in second_edges.tolist()}
    intersection = len(first & second)
    union = len(first | second)
    return {"recall": intersection / max(len(first), 1),
            "precision": intersection / max(len(second), 1),
            "jaccard": intersection / max(union, 1)}


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 67
    QGRAPHS.mkdir(parents=True, exist_ok=True)
    n, adjacency, behaviour_edges = study.load_graph()
    orbit_b5 = np.loadtxt(ORBIT_B5, dtype=np.float64)
    orbit_b4 = np.loadtxt(ORBIT_B4, dtype=np.float64)
    degrees_b = np.array([len(adjacency[i]) for i in range(n)], dtype=np.float64)
    rng = np.random.default_rng(42)
    stars = sample_stars(adjacency, n, rng, STARS)
    print(f"behaviour graph: {n} items, {len(behaviour_edges)} edges, {len(stars)} sampled stars",
          flush=True)

    rows = []
    for arm in study.load_arms(limit):
        tokens = study.tokens_of(arm["sid_path"], n)
        vectors, geometry = study.quantised_vectors(arm["number"], tokens, n)
        if vectors is None:
            rows.append({"number": arm["number"], "reason": geometry.get("reason", "no codebook")})
            print(f"  {arm['number']}: skipped ({geometry.get('reason')})", flush=True)
            continue
        vectors = vectors[:n]
        edges_q = knn_edges(vectors, K)
        in_path = QGRAPHS / f"{arm['number']}.in"
        out_path = QGRAPHS / f"{arm['number']}.out"
        write_graph(in_path, n, edges_q)
        orbit_q5 = orbits(in_path, out_path)
        orbit_in4 = QGRAPHS / f"{arm['number']}_4.in"
        orbit_out4 = QGRAPHS / f"{arm['number']}_4.out"
        write_graph(orbit_in4, n, edges_q)
        subprocess.run([str(ORCA), "node", "4", str(orbit_in4), str(orbit_out4)], check=True,
                       capture_output=True)
        orbit_q4 = np.loadtxt(orbit_out4, dtype=np.float64).reshape(n, -1)

        degree_q = np.bincount(edges_q.ravel(), minlength=n).astype(np.float64)
        both = (degrees_b > 0) & (degree_q > 0)
        degree_rank = float(spearman_rows(degrees_b[:, None], degree_q[:, None])[0])
        orbit_rho = spearman_rows(orbit_b5, orbit_q5)
        profile_cosine = cosine_rows(orbit_b5, orbit_q5)
        overlap = edge_overlap(behaviour_edges, edges_q)

        knn_pairs = {(int(a), int(b)) for a, b in edges_q.tolist()}
        recovered_leaves, hub_kept = [], []
        for centre, leaves in stars:
            in_knn = [leaf in knn_pairs or (centre, leaf) in knn_pairs or (leaf, centre) in knn_pairs
                      for leaf in leaves]
            recovered_leaves.append(sum(in_knn) / 4.0)
            hub_kept.append(float(degree_q[centre] > max(degree_q[leaf] for leaf in leaves)))

        row = {"number": arm["number"], "recall": arm["recall"],
               "geometry": {key: geometry[key] for key in ("mixed", "width") if key in geometry},
               "gcd73": gcd(orbit_b5, orbit_q5),
               "gcd15": gcd(orbit_b4, orbit_q4),
               "deltacon": deltacon(behaviour_edges, edges_q, n),
               "edge_recall": overlap["recall"], "edge_precision": overlap["precision"],
               "edge_jaccard": overlap["jaccard"],
               "degree_rank": degree_rank,
               "orbit_rho_median": float(np.nanmedian(orbit_rho)),
               "profile_cosine": float(np.nanmean(profile_cosine[both])),
               "star_role_kept": float(np.mean(hub_kept)),
               "star_leaves_recovered": float(np.mean(recovered_leaves)),
               "star_full_kept": float(np.mean([value == 1.0 for value in recovered_leaves]))}
        rows.append(row)
        print(f"  {arm['number']} recall {arm['recall']:.5f} | gcd73 {row['gcd73']:.3f} "
              f"gcd15 {row['gcd15']:.3f} | dc {row['deltacon']:.3f} | edge rec {row['edge_recall']:.3f} "
              f"jac {row['edge_jaccard']:.3f} | orbit rho {row['orbit_rho_median']:+.3f} | "
              f"star role {row['star_role_kept']:.3f} full {row['star_full_kept']:.3f}", flush=True)

    path = OUT / "quantised_structure_all.json"
    path.write_text(json.dumps({"k": K, "stars": len(stars), "rows": rows}, indent=1, default=float))
    print(f"\nwritten {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
