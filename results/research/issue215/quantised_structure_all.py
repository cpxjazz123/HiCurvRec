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

# The behaviour graph has mean degree 25.7.  A single k cannot match that for every arm: arms whose
# quantised vectors collapse to a few distinct points produce one-sided ties and, with k fixed, a hub
# of degree 22k (arm 37).  The k is therefore chosen per arm from this grid so the union's mean degree
# lands closest to the behaviour graph, and the chosen k and the resulting degrees are recorded.
# Mutual k-NN: a pair is an edge only when each side picks the other.  Under the union an item with
# thousands of exact ties (identical quantised vectors) became a degree-22,307 hub and ORCA never
# finished; mutual selection bounds every degree by the k, at the cost of a lower density, so k is
# chosen per arm from this grid to come as close to the behaviour graph's mean degree as the cap allows.
K_GRID = (5, 8, 10, 13, 16, 20, 26, 32, 40, 60)
K_MAX = max(K_GRID)
K = 13                      # fallback; each arm records the k it actually used
TARGET_DEGREE = 25.7
STARS = 400
CHUNK = 2048


def nearest(vectors, k):
    """The k nearest neighbours of every item (self excluded), as an (n, k) index array."""
    n = vectors.shape[0]
    squared = np.einsum("ij,ij->i", vectors, vectors)
    neighbours = np.full((n, k), -1, dtype=np.int64)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        block = squared[start:stop, None] + squared[None, :] - 2.0 * vectors[start:stop] @ vectors.T
        block[np.arange(stop - start), np.arange(start, stop)] = np.inf
        neighbours[start:stop] = np.argpartition(block, k, axis=1)[:, :k]
    return neighbours


def mutual_edges(neighbours, k, n):
    """Pairs that select each other within k, as ORCA's graph file needs."""
    rows = np.repeat(np.arange(n), k)
    columns = neighbours[:, :k].ravel()
    forward = np.unique(rows.astype(np.int64) * n + columns)
    backward = np.unique(columns.astype(np.int64) * n + rows)
    codes = np.intersect1d(forward, backward)
    low = codes // n
    high = codes % n
    keep = low < high
    return np.stack([low[keep], high[keep]], axis=1).astype(np.int64)


def knn_edges(vectors, target=TARGET_DEGREE):
    """Pick the k whose union matches the behaviour graph's mean degree, and return its edges.

    Records the chosen k, the resulting mean and maximum degree -- a collapsed arm will still show a
    huge hub, and that has to be visible rather than smoothed away.
    """
    n = vectors.shape[0]
    neighbours = nearest(vectors, K_MAX)
    best = None
    for k in K_GRID:
        edges = mutual_edges(neighbours, k, n)
        degree = np.bincount(edges.ravel(), minlength=n)
        mean_degree = float(degree.mean())
        entry = (abs(mean_degree - target), k, edges, mean_degree, int(degree.max()))
        if best is None or entry[0] < best[0]:
            best = entry
    _, k, edges, mean_degree, max_degree = best
    return edges, {"k": k, "mean_degree": mean_degree, "max_degree": max_degree}


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
    from scipy.sparse.linalg import cg

    def affinity(edges):
        adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        adjacency = (adjacency + adjacency.T).tocsr().astype(np.float64)
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        epsilon = 1.0 / (1.0 + degree.max())
        return identity(n, format="csr") + epsilon ** 2 * diags(degree) - epsilon * adjacency

    # sparse Cholesky/LU on a 24,587-node system fills in badly (minutes), while conjugate gradients
    # on this well-conditioned affinity converge in well under a second per seed
    first_affinity = affinity(first_edges)
    second_affinity = affinity(second_edges)
    rng = np.random.default_rng(seed)
    picks = rng.choice(n, size=min(seeds, n), replace=False)
    total = 0.0
    for pick in picks:
        target = np.zeros(n)
        target[pick] = 1.0
        left, info_left = cg(first_affinity, target, rtol=1e-8, maxiter=500)
        right, info_right = cg(second_affinity, target, rtol=1e-8, maxiter=500)
        if info_left or info_right:
            raise RuntimeError(f"DeltaCon solve did not converge ({info_left}, {info_right})")
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


# per-worker globals, filled by the initialiser so the big tables are shared once, not per task
WORKER = {}


def _initialise():
    import os
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    n, adjacency, behaviour_edges = study.load_graph()
    WORKER.update(n=n, adjacency=adjacency, behaviour_edges=behaviour_edges,
                  orbit_b5=np.loadtxt(ORBIT_B5, dtype=np.float64),
                  orbit_b4=np.loadtxt(ORBIT_B4, dtype=np.float64),
                  degrees_b=np.array([len(adjacency[i]) for i in range(n)], dtype=np.float64),
                  stars=sample_stars(adjacency, n, np.random.default_rng(42), STARS),
                  behaviour_pairs={(int(a), int(b)) if a < b else (int(b), int(a))
                                   for a, b in behaviour_edges.tolist()})
    print(f"worker ready: {n} items, {len(behaviour_edges)} behaviour edges", flush=True)


def _one_arm(task):
    number, sid_path, recall = task
    n = WORKER["n"]
    cache = QGRAPHS / f"{number}.npz"
    tokens = study.tokens_of(sid_path, n)
    vectors, geometry = study.quantised_vectors(number, tokens, n)
    if vectors is None:
        return {"number": number, "reason": geometry.get("reason", "no codebook")}
    vectors = vectors[:n]
    edges_q, graph_stats = knn_edges(vectors)
    if cache.exists():
        stored = np.load(cache)
        orbit_q5, orbit_q4 = stored["orbit5"], stored["orbit4"]
    else:
        in_path = QGRAPHS / f"{number}.in"
        out_path = QGRAPHS / f"{number}.out"
        write_graph(in_path, n, edges_q)
        orbit_q5 = orbits(in_path, out_path)
        in4 = QGRAPHS / f"{number}_4.in"
        out4 = QGRAPHS / f"{number}_4.out"
        write_graph(in4, n, edges_q)
        subprocess.run([str(ORCA), "node", "4", str(in4), str(out4)], check=True, capture_output=True)
        orbit_q4 = np.loadtxt(out4, dtype=np.float64).reshape(n, -1)
        np.savez(cache, orbit5=orbit_q5, orbit4=orbit_q4)

    degree_q = np.bincount(edges_q.ravel(), minlength=n).astype(np.float64)
    both = (WORKER["degrees_b"] > 0) & (degree_q > 0)
    orbit_rho = spearman_rows(WORKER["orbit_b5"], orbit_q5)
    cosine = cosine_rows(WORKER["orbit_b5"], orbit_q5)
    overlap = edge_overlap(WORKER["behaviour_edges"], edges_q)
    knn_pairs = {(int(a), int(b)) for a, b in edges_q.tolist()}
    recovered, hub = [], []
    for centre, leaves in WORKER["stars"]:
        hits = [leaf in knn_pairs or (centre, leaf) in knn_pairs or (leaf, centre) in knn_pairs
                for leaf in leaves]
        recovered.append(sum(hits) / 4.0)
        hub.append(float(degree_q[centre] > max(degree_q[leaf] for leaf in leaves)))
    row = {"number": number, "recall": recall, "graph": graph_stats,
           "geometry": {key: geometry[key] for key in ("mixed", "width") if key in geometry},
           "gcd73": gcd(WORKER["orbit_b5"], orbit_q5),
           "gcd15": gcd(WORKER["orbit_b4"], orbit_q4),
           "deltacon": deltacon(WORKER["behaviour_edges"], edges_q, n),
           "edge_recall": overlap["recall"], "edge_precision": overlap["precision"],
           "edge_jaccard": overlap["jaccard"],
           "degree_rank": float(spearman_rows(WORKER["degrees_b"][:, None], degree_q[:, None])[0]),
           "orbit_rho_median": float(np.nanmedian(orbit_rho)),
           "profile_cosine": float(np.nanmean(cosine[both])),
           "star_role_kept": float(np.mean(hub)),
           "star_leaves_recovered": float(np.mean(recovered)),
           "star_full_kept": float(np.mean([value == 1.0 for value in recovered]))}
    return row


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 67
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    QGRAPHS.mkdir(parents=True, exist_ok=True)
    tasks = [(arm["number"], arm["sid_path"], arm["recall"]) for arm in study.load_arms(limit)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise) as pool:
        futures = {pool.submit(_one_arm, task): task[0] for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if "reason" in row:
                print(f"  {row['number']}: skipped ({row['reason']})", flush=True)
                continue
            print(f"  {row['number']} recall {row['recall']:.5f} | gcd73 {row['gcd73']:.3f} "
                  f"gcd15 {row['gcd15']:.3f} | dc {row['deltacon']:.3f} | "
                  f"edge rec {row['edge_recall']:.3f} jac {row['edge_jaccard']:.3f} | "
                  f"orbit rho {row['orbit_rho_median']:+.3f} | cos {row['profile_cosine']:.3f} | "
                  f"star role {row['star_role_kept']:.3f} full {row['star_full_kept']:.3f}", flush=True)
    rows.sort(key=lambda row: row["number"])
    path = OUT / "quantised_structure_all.json"
    path.write_text(json.dumps({"k": K, "stars": STARS, "rows": rows}, indent=1, default=float))
    print(f"\nwritten {path} ({len(rows)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
