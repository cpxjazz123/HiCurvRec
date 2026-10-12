"""Library for the behavior-graph geometry analysis. Imported by the tests first.

Everything here is a pure function of its arguments, so the unit tests can drive it
with hand-built trees, cycles and community graphs before any real data is touched.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque

import numpy as np
import networkx as nx


# --------------------------------------------------------------- sequence extraction
def load_user_sequences(frame) -> tuple[dict, dict]:
    """Ordered item sequence per user, from the real columns.

    The parquet has NO timestamp: the temporal order is carried by
    len(seen_history), because each row appends one item. The order is therefore
    recovered by sorting each user's rows on len(seen_history), and the function
    also reports whether the file was already in that order, so the recovery is
    verified rather than assumed.

    Returns (sequences, audit). sequences[user] is a list of item ids in purchase
    order; the first row's seen_history supplies the items bought before the first
    recorded target.
    """
    users = frame["user"].to_numpy()
    seen = frame["seen_history"].to_numpy()
    target = frame["target"].to_numpy()

    rows_by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    already_sorted = 0
    for i in range(len(users)):
        u = int(users[i])
        rows_by_user[u].append((len(seen[i]), i))

    sequences: dict[int, list[int]] = {}
    duplicates_in_history = 0
    for u, rows in rows_by_user.items():
        order = [idx for _, idx in sorted(rows, key=lambda t: (t[0], t[1]))]
        if order == [idx for _, idx in rows]:
            already_sorted += 1
        seq: list[int] = []
        if order:
            first_seen = seen[order[0]]
            seq.extend(int(x) for x in first_seen)
        for idx in order:
            seq.append(int(target[idx]))
        # the same item can be bought twice by a user; that is real, and is kept
        if len(set(seq)) != len(seq):
            duplicates_in_history += 1
        sequences[u] = seq

    audit = {
        "rows": int(len(frame)),
        "users": int(len(rows_by_user)),
        "users_already_in_history_length_order": int(already_sorted),
        "users_with_a_repeated_item": int(duplicates_in_history),
        "history_column_used": "seen_history",
        "order_source": "len(seen_history) ascending, ties broken by row index",
        "timestamp_column_present": False,
    }
    return sequences, audit


def transitions_from_sequences(sequences: dict) -> tuple[Counter, Counter]:
    """Directed weighted transitions, plus self-loops counted separately.

    Only ADJACENT items inside ONE user's sequence produce an edge. Items are never
    ordered by id and different users are never concatenated.
    """
    directed: Counter = Counter()
    self_loops: Counter = Counter()
    for _u, seq in sequences.items():
        for a, b in zip(seq[:-1], seq[1:]):
            if a == b:
                self_loops[a] += 1
            else:
                directed[(a, b)] += 1
    return directed, self_loops


# --------------------------------------------------------------------------- graphs
def directed_graph(directed: Counter) -> nx.DiGraph:
    g = nx.DiGraph()
    for (a, b), w in directed.items():
        g.add_edge(a, b, weight=int(w))
    return g


def undirected_graph(directed: Counter) -> nx.Graph:
    """Symmetric version: weight(a,b) = count(a->b) + count(b->a).

    A behaviour relation is symmetric for the purpose of grouping and for any
    distance-based analysis, so the directed counts are summed. The direction
    asymmetry that this loses is measured separately and reported.
    """
    g = nx.Graph()
    for (a, b), w in directed.items():
        if g.has_edge(a, b):
            g[a][b]["weight"] += int(w)
            g[a][b]["fwd"] += int(w) if g[a][b]["dir"] == (a, b) else 0
        else:
            g.add_edge(a, b, weight=int(w), dir=(a, b), fwd=int(w))
    # normalise the fwd/dir bookkeeping
    for a, b, d in g.edges(data=True):
        fwd = directed.get((a, b), 0)
        rev = directed.get((b, a), 0)
        d["weight"] = fwd + rev
        d["fwd"] = fwd
        d["rev"] = rev
    return g


def graph_stats(g: nx.Graph) -> dict:
    n, m = g.number_of_nodes(), g.number_of_edges()
    deg = np.array([d for _, d in g.degree()], dtype=np.float64) if n else np.zeros(0)
    isolates = int(sum(1 for _, d in g.degree() if d == 0))
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    lcc_nodes = max(nx.connected_components(g), key=len) if n else set()
    lcc = g.subgraph(lcc_nodes).copy()
    return {
        "nodes": int(n),
        "edges": int(m),
        "mean_degree": float(deg.mean()) if n else 0.0,
        "median_degree": float(np.median(deg)) if n else 0.0,
        "isolated_nodes": isolates,
        "connected_components": int(len(comps)),
        "component_sizes_top10": comps[:10],
        "lcc_nodes": int(lcc.number_of_nodes()),
        "lcc_edges": int(lcc.number_of_edges()),
        "lcc_fraction_of_nodes": float(lcc.number_of_nodes() / n) if n else 0.0,
        "density": float(2 * m / (n * (n - 1))) if n > 1 else 0.0,
        "clustering": average_clustering_fast(g) if n else {"average_clustering": 0.0},
        "assortativity_degree": float(nx.degree_assortativity_coefficient(g)) if m else float("nan"),
        # Path metrics on a large component are sampled, not enumerated: exact
        # average shortest path on ~20k nodes needs one BFS per node and is not a
        # feasible Python loop. The sample is seeded and its size is recorded, and
        # the estimate is reported as an estimate.
        "path_metrics_sampled": path_metrics_sampled(lcc, seed=42, sources=400),
        "is_connected": bool(nx.is_connected(g)) if n else False,
    }


def average_clustering_fast(g: nx.Graph) -> dict:
    """Exact average local clustering coefficient, via sparse triangle counting.

    NetworkX's average_clustering enumerates every neighbour pair of every node in
    Python, which is O(sum d^2) and takes tens of minutes on a 20k-node heavy-tailed
    graph. The number of triangles through each node is instead
    (A^2 .* A).sum(axis=1) / 2 with A the sparse adjacency, which is a couple of
    sparse products, and the local coefficient is triangles / C(d, 2). Same value,
    seconds instead of minutes.
    """
    from scipy import sparse
    nodes = list(g.nodes())
    n = len(nodes)
    if n < 3:
        return {"nodes": n, "average_clustering": 0.0, "global_clustering_transitivity": 0.0,
                "method": "exact sparse triangle count"}
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols = [], []
    for a, b in g.edges():
        rows.append(idx[a]); cols.append(idx[b])
        rows.append(idx[b]); cols.append(idx[a])
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A2 = A @ A
    triangles = np.asarray(A2.multiply(A).sum(axis=1)).ravel() / 2.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    denom = deg * (deg - 1) / 2.0
    local = np.divide(triangles, denom, out=np.zeros_like(triangles), where=denom > 0)
    total_denom = denom.sum()
    return {
        "nodes": int(n),
        "average_clustering": float(local.mean()),
        "global_clustering_transitivity": float(triangles.sum() / total_denom) if total_denom > 0 else 0.0,
        "method": "exact sparse triangle count (A^2 .* A)",
    }


def bfs_distances_scipy(g: nx.Graph, sources: list) -> np.ndarray:
    """Same distances as bfs_distances, computed by scipy's C shortest_path.

    NetworkX's BFS is a Python loop; 2000 sources on a 24.5k-node, 315k-edge component
    is minutes of pure Python. scipy.sparse.csgraph does the same unweighted shortest
    paths in C. A unit test pins the two implementations against each other.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import shortest_path
    nodes = list(g.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols = [], []
    for a, b in g.edges():
        rows.append(idx[a]); cols.append(idx[b])
        rows.append(idx[b]); cols.append(idx[a])
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    src = np.array([idx[s] for s in sources], dtype=np.int64)
    # scipy returns the rows in the order the indices were passed; D[i, j] is the
    # distance from sources[i] to the node at POSITION j of g.nodes(), which is NOT
    # the same as the distance to sources[j] unless the node list is index-aligned.
    # Callers that work in source space must go through source_submatrix first.
    D = shortest_path(A, method="D", unweighted=True, directed=False, indices=src)
    return np.asarray(D, dtype=np.int32)


def source_submatrix(D: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    """Restrict a distance matrix to a set of sources, in SOURCE order.

    bfs_distances_scipy returns rows in the order the sources were passed but keeps
    EVERY node as a column. A quadruple of source indices therefore cannot be looked
    up in D directly: D[a, b] would read the distance to the node at POSITION b, not
    to source b. This returns M[i, j] = distance from source i to source j, which is
    what gromov_delta_matrix needs.
    """
    return D[:, source_positions]


def path_metrics_sampled(g: nx.Graph, seed: int = 42, sources: int = 200) -> dict:
    """Diameter and mean shortest path from a seeded BFS sample.

    Exact values would need one BFS per node. The sample is drawn with a fixed seed
    so the numbers are reproducible, and the sample size is reported with them.
    """
    n = g.number_of_nodes()
    if n == 0:
        return {"nodes": 0}
    if n == 1:
        return {"nodes": 1, "diameter_estimate": 0, "mean_distance_estimate": 0.0,
                "sources_sampled": 1, "exact": n <= 1}
    nodes = list(g.nodes())
    rng = np.random.default_rng(seed)
    k = min(sources, n)
    src = [nodes[i] for i in rng.choice(n, size=k, replace=False)]
    eccs, means = [], []
    for s in src:
        d = nx.single_source_shortest_path_length(g, s)
        # exclude the source itself: a node's distance to itself is 0 and would bias
        # the mean down by exactly the 1/n a naive average over the row carries
        vals = np.fromiter((v for k, v in d.items() if k != s), dtype=np.int64)
        if vals.size == 0:
            continue          # isolated source inside a disconnected subgraph
        eccs.append(int(vals.max()))
        means.append(float(vals.mean()))
    if not eccs:
        return {"nodes": int(n), "diameter_estimate": 0, "mean_distance_estimate": 0.0,
                "sources_sampled": 0, "exact": False}
    eccs = np.array(eccs)
    return {
        "nodes": int(n),
        "sources_sampled": int(k),
        "diameter_estimate": int(eccs.max()),
        "eccentricity_mean": float(eccs.mean()),
        "eccentricity_p95": float(np.quantile(eccs, 0.95)),
        "mean_distance_estimate": float(np.mean(means)),
        "mean_distance_se": float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0,
        "exact": bool(k == n),
    }


def bfs_distances(g: nx.Graph, sources: list, nodes_index: dict | None = None) -> np.ndarray:
    """Unweighted shortest-path distances from each source, one row per source.

    Unweighted is the primary analysis, as required: the four-point condition needs
    a metric, and a symmetric unweighted graph distance satisfies symmetry and the
    triangle inequality by construction.
    """
    order = nodes_index if nodes_index is not None else {v: i for i, v in enumerate(g.nodes())}
    out = np.full((len(sources), len(order)), -1, dtype=np.int32)
    for r, s in enumerate(sources):
        dist = nx.single_source_shortest_path_length(g, s)
        for v, dv in dist.items():
            out[r, order[v]] = dv
    return out


# --------------------------------------------------------- Gromov hyperbolicity
def gromov_delta_from_sums(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> np.ndarray:
    """delta = (largest - middle) / 2 for the three pair-sum combinations."""
    stacked = np.stack([s1, s2, s3], axis=0)
    stacked.sort(axis=0)
    return (stacked[2] - stacked[1]) / 2.0


def gromov_delta_matrix(d: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """d is (S, N) distances from S sources; quads is (Q, 4) index quadruples."""
    a, b, c, e = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    s1 = d[a, b] + d[c, e]
    s2 = d[a, c] + d[b, e]
    s3 = d[a, e] + d[b, c]
    return gromov_delta_from_sums(s1, s2, s3)


def delta_summary(delta: np.ndarray) -> dict:
    delta = np.asarray(delta, dtype=np.float64)
    return {
        "n_quads": int(delta.size),
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "p90": float(np.quantile(delta, 0.90)),
        "p95": float(np.quantile(delta, 0.95)),
        "p99": float(np.quantile(delta, 0.99)),
        "max": float(delta.max()),
        "std": float(delta.std()),
        "bootstrap_se_of_mean": float(delta.std(ddof=1) / math.sqrt(delta.size))
        if delta.size > 1 else 0.0,
    }


# --------------------------------------------------------------------- tree-likeness
def tree_likeness(g: nx.Graph, root, sources: int = 200, seed: int = 42) -> dict:
    """How close the graph is to a tree, measured against its own BFS tree.

    A BFS tree has n-1 edges; every remaining edge of the graph is a non-tree edge.
    The distortion is the fraction of node pairs whose BFS-tree distance differs from
    their graph distance.
    """
    if g.number_of_nodes() == 0:
        return {"nodes": 0}
    bfs = nx.bfs_tree(g, root)
    non_tree = g.number_of_edges() - bfs.number_of_edges()
    # sampled sources: one BFS per node is not feasible on a 24.5k-node component
    nodes = list(g.nodes())
    rng = np.random.default_rng(seed)
    k = min(sources, len(nodes))
    sample_nodes = [nodes[i] for i in rng.choice(len(nodes), size=k, replace=False)]
    dg = bfs_distances(g, sample_nodes)
    dt = bfs_distances(bfs, sample_nodes)
    finite = (dg >= 0) & (dt >= 0)
    diff = np.abs(dg[finite] - dt[finite])
    return {
        "root": int(root),
        "nodes": int(g.number_of_nodes()),
        "tree_edges": int(bfs.number_of_edges()),
        "non_tree_edges": int(non_tree),
        "non_tree_edge_fraction": float(non_tree / max(g.number_of_edges(), 1)),
        "mean_tree_distance_excess": float(diff.mean()) if diff.size else 0.0,
        "p95_tree_distance_excess": float(np.quantile(diff, 0.95)) if diff.size else 0.0,
        "share_of_pairs_exactly_preserved": float((diff == 0).mean()) if diff.size else 0.0,
        "sources_sampled": int(k),
        "exact": bool(k == len(nodes)),
    }


def degree_preserving_null(g: nx.Graph, swaps_per_edge: int, seed: int, replicates: int = 1):
    """Degree-preserving randomised graphs by double-edge swaps.

    The configuration model is avoided because it creates multi-edges and
    self-loops; double-edge swaps keep the degree sequence EXACTLY while rewiring
    which nodes are connected, which is what a null model for "is the structure
    beyond the degree distribution" needs.
    """
    out = []
    for r in range(replicates):
        h = g.copy()
        rng = np.random.default_rng(seed + r)
        nx.double_edge_swap(h, nswap=swaps_per_edge * g.number_of_edges(),
                            max_tries=100 * swaps_per_edge * max(g.number_of_edges(), 1),
                            seed=int(rng.integers(1 << 31)))
        out.append(h)
    return out


# ------------------------------------------------------------- power-law fitting
def ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = 1.0 - np.arange(len(x)) / len(x)
    return x, y


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def powerlaw_mle(x: np.ndarray, xmin: float) -> tuple[float, float]:
    """Clauset-Shalizi-Newman MLE for a discrete power law above xmin.

    alpha = 1 + n / sum(ln(x_i / (xmin - 0.5))); the 0.5 correction is the standard
    discrete approximation. Returns (alpha, standard_error).
    """
    tail = x[x >= xmin]
    n = tail.size
    if n < 2:
        return float("nan"), float("nan")
    denom = np.log(tail / (xmin - 0.5)).sum()
    alpha = 1.0 + n / denom
    se = (alpha - 1.0) / math.sqrt(n)
    return float(alpha), float(se)


def ks_distance_powerlaw(x: np.ndarray, xmin: float, alpha: float) -> float:
    """KS distance between the empirical tail CDF and the fitted power law."""
    tail = np.sort(x[x >= xmin])
    n = tail.size
    if n < 2:
        return float("nan")
    cdf_emp = np.arange(1, n + 1) / n
    # discrete power-law CDF: P(X <= k) = 1 - zeta(alpha,k)/zeta(alpha,xmin)
    from scipy.special import zeta
    k = np.unique(tail)
    cdf_fit = 1.0 - np.array([zeta(alpha, kk) / zeta(alpha, xmin) for kk in k])
    # evaluate the fitted CDF at the empirical points
    fit_at_tail = 1.0 - np.array([zeta(alpha, kk) / zeta(alpha, xmin) for kk in tail])
    return float(np.max(np.abs(cdf_emp - fit_at_tail)))


def ks_distance_powerlaw_fast(x: np.ndarray, xmin: float, alpha: float) -> float:
    """Discrete power-law KS distance by a cumulative table.

    The reference implementation calls scipy's incomplete zeta once per distinct tail
    value, which makes an xmin scan of a few hundred candidates unusable. The pmf is
    k^-alpha normalised over [xmin, kmax] and cumulated once, so the CDF at every
    observation is a lookup. Same statistic, O(kmax) instead of O(tail * zeta).
    """
    tail = np.sort(np.asarray(x, dtype=np.float64))
    tail = tail[tail >= xmin]
    if tail.size < 2:
        return float("nan")
    kmin, kmax = int(xmin), int(tail.max())
    ks = np.arange(kmin, kmax + 1, dtype=np.float64)
    pmf = ks ** (-alpha)
    total = pmf.sum()
    if total <= 0:
        return float("nan")
    cdf = np.cumsum(pmf / total)
    cdf_at = cdf[(tail - kmin).astype(np.int64)]
    emp = np.arange(1, tail.size + 1) / tail.size
    return float(np.max(np.abs(emp - cdf_at)))


def vuong_like_test(loglik_1: float, loglik_2: float, per_point_diff: np.ndarray) -> dict:
    """Clauset-style normalised log-likelihood ratio between two fitted models.

    R = (l1 - l2) / (sqrt(n) * sigma) with sigma the standard deviation of the
    per-point log-likelihood difference; under the null the two models are
    indistinguishable and R is approximately standard normal.
    """
    d = np.asarray(per_point_diff, dtype=np.float64)
    n = d.size
    if n < 2:
        return {"R": float("nan"), "p_two_sided": float("nan"), "n": int(n)}
    sigma = d.std(ddof=1)
    if sigma <= 0:
        return {"R": float("inf") if loglik_1 > loglik_2 else 0.0, "p_two_sided": 0.0, "n": int(n)}
    from scipy import stats
    R = float((loglik_1 - loglik_2) / (math.sqrt(n) * sigma))
    p = float(2 * (1 - stats.norm.cdf(abs(R))))
    return {"R": R, "p_two_sided": p, "n": int(n),
            "favours": "model_1" if R > 0 else "model_2"}


def fit_xmin(x: np.ndarray, min_tail: int, max_candidates: int) -> dict:
    """Choose xmin by minimising the KS distance, as Clauset et al. prescribe."""
    xs = np.sort(np.asarray(x, dtype=np.float64))
    candidates = np.unique(xs[xs < np.quantile(xs, 0.9)])
    if candidates.size > max_candidates:
        candidates = candidates[np.linspace(0, candidates.size - 1, max_candidates).astype(int)]
    best = {"xmin": float("nan"), "alpha": float("nan"), "ks": float("inf"), "n_tail": 0}
    for xmin in candidates:
        tail_n = int((xs >= xmin).sum())
        if tail_n < min_tail:
            continue
        alpha, _ = powerlaw_mle(xs, float(xmin))
        if not np.isfinite(alpha) or alpha <= 1.0:
            continue
        ks = ks_distance_powerlaw_fast(xs, float(xmin), alpha)
        if np.isfinite(ks) and ks < best["ks"]:
            best = {"xmin": float(xmin), "alpha": alpha, "ks": ks, "n_tail": tail_n}
    return best


def fit_alternative(x: np.ndarray, xmin: float, family: str) -> dict:
    """Lognormal / exponential fit above xmin by CONDITIONAL (truncated) MLE.

    The power law is defined above xmin, so the comparison has to be on the same
    conditional support. Fitting an untruncated lognormal to a truncated sample is
    biased against it and would make the power law look better than it is, so the
    truncation is carried in the likelihood:

        log L = sum log pdf(x_i) - n * log S(xmin)

    The KS distance is then measured against the conditional CDF
    (F(x) - F(xmin)) / (1 - F(xmin)).
    """
    from scipy import stats
    from scipy.optimize import minimize
    tail = np.asarray(x, dtype=np.float64)
    tail = tail[tail >= xmin]
    if tail.size < 5:
        return {"family": family, "ks": float("nan"), "loglik": float("nan"), "params": None}
    if family == "lognormal":
        def nll(p):
            mu, sigma = float(p[0]), float(p[1])
            if sigma <= 1e-9:
                return 1e12
            lp = stats.lognorm.logpdf(tail, s=sigma, scale=math.exp(mu))
            sf = float(stats.lognorm.sf(xmin, s=sigma, scale=math.exp(mu)))
            return float(-(lp.sum() - tail.size * math.log(max(sf, 1e-300))))
        res = minimize(nll, x0=[float(np.log(np.median(tail))), float(np.std(np.log(tail)) or 1.0)],
                       method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
        mu, sigma = float(res.x[0]), float(res.x[1])
        params = {"mu": mu, "sigma": sigma}
        sf_min = float(stats.lognorm.sf(xmin, s=sigma, scale=math.exp(mu)))
        cdf = 1.0 - stats.lognorm.sf(tail, s=sigma, scale=math.exp(mu)) / max(sf_min, 1e-300)
        loglik = float(-res.fun)
    elif family == "exponential":
        scale = float((tail - xmin).mean())
        if scale <= 1e-12:
            return {"family": family, "ks": float("nan"), "loglik": float("nan"), "params": None}
        params = {"scale": scale}
        cdf = 1.0 - np.exp(-(tail - xmin) / scale)
        loglik = float(stats.expon.logpdf(tail, loc=xmin, scale=scale).sum())
    else:
        raise ValueError(family)
    order = np.argsort(tail)
    cdf_emp = np.arange(1, tail.size + 1) / tail.size
    ks = float(np.max(np.abs(cdf_emp - cdf[order])))
    return {"family": family, "ks": ks, "loglik": loglik, "params": params}


def powerlaw_loglik(x: np.ndarray, xmin: float, alpha: float) -> float:
    from scipy.special import zeta
    tail = np.asarray(x, dtype=np.float64)
    tail = tail[tail >= xmin]
    if tail.size == 0:
        return float("nan")
    return float((-alpha * np.log(tail) - math.log(float(zeta(alpha, xmin)))).sum())


# ------------------------------------------------------- embedding distortion
def normalised_stress(true_d: np.ndarray, emb_d: np.ndarray) -> float:
    num = ((true_d - emb_d) ** 2).sum()
    den = (true_d ** 2).sum()
    return float(math.sqrt(num / den)) if den > 0 else float("nan")


def mean_relative_distortion(true_d: np.ndarray, emb_d: np.ndarray) -> float:
    m = true_d > 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(emb_d[m] - true_d[m]) / true_d[m]))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats
    if a.size < 3:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def fit_scale(true_d: np.ndarray, emb_d: np.ndarray) -> float:
    """The single global positive scale the comparison is allowed to learn."""
    m = true_d > 0
    if not m.any():
        return 1.0
    return float((emb_d[m] * true_d[m]).sum() / max((emb_d[m] ** 2).sum(), 1e-12))


def neighbourhood_preservation(true_neighbours: list, emb_order: np.ndarray, k: int) -> float:
    """Share of each node's k nearest graph neighbours that are within its k nearest
    embedding neighbours."""
    hits = 0
    total = 0
    order = np.array(emb_order, dtype=np.float64, copy=True)
    np.fill_diagonal(order, np.inf)          # a node is not its own neighbour
    emb_rank = np.argsort(order, axis=1)[:, :k]
    for i, nb in enumerate(true_neighbours):
        if not nb:
            continue
        total += 1
        hits += len(set(nb[:k]) & set(emb_rank[i].tolist()))
    return float(hits / (total * k)) if total and k else float("nan")
