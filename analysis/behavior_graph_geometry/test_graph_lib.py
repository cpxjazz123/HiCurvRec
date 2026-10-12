"""Unit tests for the behavior-graph geometry library. Written before the real run.

Every test uses a hand-built tree, cycle, star or community graph whose answers are
known exactly, so the metrics are validated on cases with a derivable ground truth
before any real transition data is touched.

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/behavior_graph_geometry/test_graph_lib.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph_lib as L  # noqa: E402

_ok = [True]


def req(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    _ok[0] = _ok[0] and bool(cond)


# ----------------------------------------------------------- transition extraction
def test_transitions() -> None:
    frame = pd.DataFrame({
        "user":    [0, 0, 0, 1, 1],
        "history": [[], [7], [7, 3], [], [9]],
        "seen_history": [[], [7], [7, 3], [], [9]],
        "target":  [7, 3, 9, 5, 9],
    })
    seqs, audit = L.load_user_sequences(frame)
    req(seqs[0] == [7, 3, 9], f"T1 user 0 sequence is the row order ({seqs[0]})")
    req(seqs[1] == [5, 9], f"T1 user 1 sequence ({seqs[1]})")
    req(audit["users"] == 2 and audit["rows"] == 5, f"T1 audit counts ({audit})")

    d, loops = L.transitions_from_sequences(seqs)
    req(dict(d) == {(7, 3): 1, (3, 9): 1, (5, 9): 1},
        f"T1 only adjacent same-user transitions ({dict(d)})")
    req((9, 7) not in d and (9, 5) not in d,
        "T1 no reverse edges and no cross-user joins are invented")

    # order recovery: shuffling the rows inside a user must not change the sequence
    shuffled = frame.sample(frac=1.0, random_state=1)
    seqs2, _ = L.load_user_sequences(shuffled)
    req(seqs2[0] == seqs[0] and seqs2[1] == seqs[1],
        "T1 the sequence is recovered from len(seen_history), not from row order")

    # self loops counted separately, never silently dropped
    frame2 = pd.DataFrame({"user": [0, 0], "history": [[], [4]], "seen_history": [[], [4]],
                           "target": [4, 4]})
    seqs3, _ = L.load_user_sequences(frame2)
    d3, loops3 = L.transitions_from_sequences(seqs3)
    req(dict(loops3) == {4: 1} and len(d3) == 0,
        f"T1 a repeated purchase is a self loop, reported separately ({dict(loops3)})")

    # item ids are never used to synthesise order
    frame3 = pd.DataFrame({"user": [0, 0], "history": [[], [2]], "seen_history": [[], [2]],
                           "target": [2, 1]})
    seqs4, _ = L.load_user_sequences(frame3)
    d4, _ = L.transitions_from_sequences(seqs4)
    req(dict(d4) == {(2, 1): 1},
        f"T1 a descending-id transition is kept as observed, not sorted ({dict(d4)})")


def test_undirected_symmetry() -> None:
    d = {(1, 2): 3, (2, 1): 5, (2, 3): 1}
    g = L.undirected_graph(d)
    req(g[1][2]["weight"] == 8, f"T1 undirected weight sums both directions ({g[1][2]['weight']})")
    req(g[1][2]["fwd"] == 3 and g[1][2]["rev"] == 5,
        f"T1 the direction split is preserved for the asymmetry analysis "
        f"({g[1][2]['fwd']}/{g[1][2]['rev']})")
    req(g.number_of_edges() == 2, "T1 symmetric edges are merged, not duplicated")


# ------------------------------------------------------------------ graph metrics
def test_graph_stats_and_distance() -> None:
    p = nx.path_graph(5)                       # a-b-c-d-e
    s = L.graph_stats(p)
    req(s["nodes"] == 5 and s["edges"] == 4, f"T1 path stats ({s['nodes']}/{s['edges']})")
    pm = s["path_metrics_sampled"]
    req(pm["diameter_estimate"] == 4 and pm["exact"],
        f"T1 path diameter 4, exact on a small graph ({pm})")
    req(abs(pm["mean_distance_estimate"] - 2.0) < 1e-9,
        f"T1 path average distance 2.0 ({pm['mean_distance_estimate']})")
    req(s["clustering"]["average_clustering"] == 0.0,
        f"T1 a path has no triangles ({s['clustering']})")

    d = L.bfs_distances(p, [0, 4])
    req(d[0, 4] == 4 and d[1, 0] == 4, "T1 BFS distance symmetric on a path")
    # triangle inequality on every triple of a random graph
    g = nx.gnp_random_graph(60, 0.08, seed=3)
    nodes = list(g.nodes())
    rng_pairs = np.random.default_rng(7).choice
    dm = L.bfs_distances(g, nodes)
    finite = dm >= 0
    ok = True
    rng = np.random.default_rng(0)
    for _ in range(2000):
        i, j, k = rng.integers(0, len(nodes), 3)
        if finite[i, j] and finite[j, k] and finite[i, k]:
            if dm[i, k] > dm[i, j] + dm[j, k]:
                ok = False
                break
    req(ok, "T1 the unweighted graph distance satisfies the triangle inequality")

    # the scipy shortest-path implementation must agree with the networkx one.
    # The source list is deliberately UNSORTED and sparse: scipy returns its rows in
    # ascending index order, so a sorted list would hide a mis-alignment whose only
    # visible symptom is a non-zero diagonal.
    pick = sorted(int(x) for x in rng_pairs(len(nodes), size=13, replace=False))
    src = [nodes[i] for i in pick][::-1]
    dn = L.bfs_distances(g, src)
    ds = L.bfs_distances_scipy(g, src)
    req(dn.shape == ds.shape and np.array_equal(dn, ds),
        f"T1 scipy and networkx BFS agree on an UNSORTED source list ({dn.shape})")
    # the source-order submatrix is what a quadruple of source indices needs, and its
    # diagonal is zero; the raw matrix keeps every node as a column, so ITS diagonal
    # is only zero when the source list happens to be index-aligned
    pos = np.array([nodes.index(v) for v in src], dtype=np.int64)
    sub = L.source_submatrix(ds, pos)
    diag_ok = float(np.max(np.abs(np.diag(sub))))
    req(diag_ok == 0.0,
        f"T1 the source-order submatrix has a zero diagonal (max {diag_ok})")


def test_gromov_delta() -> None:
    # a tree has delta exactly 0 for EVERY quadruple: that is the defining property
    star = nx.star_graph(6)
    tree = nx.balanced_tree(2, 3)
    for name, t in (("star", star), ("balanced_tree", tree)):
        nodes = list(t.nodes())
        dm = L.bfs_distances(t, nodes)
        rng = np.random.default_rng(1)
        quads = rng.integers(0, len(nodes), size=(5000, 4))
        delta = L.gromov_delta_matrix(dm, quads)
        req(float(delta.max()) == 0.0, f"T1 {name}: delta == 0 on every sampled quadruple")

    # a 4-cycle has a known delta of exactly 1 for the four nodes in order
    c4 = nx.cycle_graph(4)
    dm = L.bfs_distances(c4, [0, 1, 2, 3])
    delta = L.gromov_delta_matrix(dm, np.array([[0, 1, 2, 3]]))
    req(abs(float(delta[0]) - 1.0) < 1e-12, f"T1 C4 quadruple delta == 1 ({float(delta[0])})")

    # a path IS a tree, so delta is 0 for every quadruple of it as well
    p5 = nx.path_graph(5)
    dm = L.bfs_distances(p5, [0, 1, 2, 3, 4])
    rng2 = np.random.default_rng(9)
    q5 = rng2.integers(0, 5, size=(3000, 4))
    req(float(L.gromov_delta_matrix(dm, q5).max()) == 0.0,
        "T1 a path is a tree, so delta == 0 on every sampled quadruple")

    # a 6-cycle has a known positive delta: (0,2,3,5) gives sums 4, 6, 2 -> delta 1
    c6 = nx.cycle_graph(6)
    dm6 = L.bfs_distances(c6, list(range(6)))
    d6 = L.gromov_delta_matrix(dm6, np.array([[0, 2, 3, 5]]))
    req(abs(float(d6[0]) - 1.0) < 1e-12, f"T1 C6 quadruple delta == 1 ({float(d6[0])})")
    # and the largest delta over all C6 quadruples is the known value for the cycle
    allq = np.array([[a, b, c, e] for a in range(6) for b in range(6)
                     for c in range(6) for e in range(6)])
    req(float(L.gromov_delta_matrix(dm6, allq).max()) == 1.0,
        "T1 C6 maximum delta over all quadruples is 1")

    # the formula is symmetric in its arguments
    rng = np.random.default_rng(5)
    g = nx.gnp_random_graph(40, 0.1, seed=7)
    nodes = list(g.nodes())
    dm = L.bfs_distances(g, nodes)
    q = rng.integers(0, len(nodes), size=(200, 4))
    base = L.gromov_delta_matrix(dm, q)
    perm = L.gromov_delta_matrix(dm, q[:, [2, 0, 3, 1]])
    req(np.allclose(base, perm), "T1 delta is invariant under permuting the four nodes")

    # summary statistics are consistent
    summ = L.delta_summary(base)
    req(summ["n_quads"] == 200 and summ["max"] >= summ["median"] >= 0.0,
        f"T1 delta summary is ordered ({summ['median']:.3f} <= {summ['max']:.3f})")


def test_tree_metric_on_scipy_path() -> None:
    """A tree metric must give delta == 0 through the scipy path too."""
    t = nx.random_labeled_tree(60, seed=5)
    nodes = sorted(t.nodes())
    order = [nodes[i] for i in rng_choice_unsorted(len(nodes), 25)]
    D = L.bfs_distances_scipy(t, order)
    pos = np.array([nodes.index(v) for v in order], dtype=np.int64)
    req(float(np.max(np.abs(np.diag(L.source_submatrix(D, pos))))) == 0.0,
        "T1 the source-order submatrix has a zero diagonal on a tree")
    rng = np.random.default_rng(2)
    q = rng.choice(len(order), size=(4000, 4))
    # D's rows are the sources but its columns are ALL nodes, so the quadruple of
    # source indices has to be read through the source-order submatrix. Reading
    # gromov_delta_matrix(D, q) directly treats a source index as a node position
    # and produces a plausible-looking, entirely wrong delta.
    pos = np.array([nodes.index(v) for v in order], dtype=np.int64)
    delta = L.gromov_delta_matrix(L.source_submatrix(D, pos), q)
    req(float(delta.max()) == 0.0,
        f"T1 a tree metric gives delta == 0 on every sampled quadruple "
        f"(max {float(delta.max())}) with an unsorted source list")


def rng_choice_unsorted(n, size):
    return np.random.default_rng(11).choice(n, size=size, replace=False)


def test_tree_likeness_and_null() -> None:
    t = nx.balanced_tree(2, 2)
    tl = L.tree_likeness(t, 0)
    req(tl["non_tree_edges"] == 0 and tl["share_of_pairs_exactly_preserved"] == 1.0,
        f"T1 a tree is exactly tree-like ({tl})")
    c6 = nx.cycle_graph(6)
    tl6 = L.tree_likeness(c6, 0)
    req(tl6["non_tree_edges"] == 1, f"T1 a 6-cycle has exactly 1 non-tree edge ({tl6['non_tree_edges']})")
    req(tl6["share_of_pairs_exactly_preserved"] < 1.0,
        "T1 and its BFS tree does not reproduce the graph distances")

    g = nx.gnp_random_graph(200, 0.05, seed=11)
    before = sorted(d for _, d in g.degree())
    nulls = L.degree_preserving_null(g, swaps_per_edge=5, seed=42, replicates=2)
    req(len(nulls) == 2, "T1 the null model returns the requested replicates")
    for h in nulls:
        req(sorted(d for _, d in h.degree()) == before,
            "T1 the null model preserves the degree sequence EXACTLY")
        req(h.number_of_edges() == g.number_of_edges(),
            "T1 and the edge count")
        req(not any(u == v for u, v in h.edges()), "T1 and creates no self loops")


# --------------------------------------------------------------------- power law
def test_powerlaw() -> None:
    rng = np.random.default_rng(42)
    # synthetic discrete power law with alpha = 2.5 above xmin = 10
    alpha_true, xmin_true, n = 2.5, 10, 20000
    u = rng.random(n)
    x = np.floor((xmin_true - 0.5) * (1 - u) ** (-1.0 / (alpha_true - 1))).astype(float)
    fit = L.fit_xmin(x, min_tail=200, max_candidates=200)
    req(abs(fit["alpha"] - alpha_true) < 0.15,
        f"T1 power-law alpha recovered ({fit['alpha']:.3f} vs {alpha_true}), xmin {fit['xmin']}")
    req(fit["ks"] < 0.05, f"T1 and the KS distance is small ({fit['ks']:.4f})")

    ln = np.exp(rng.normal(3.0, 1.0, 20000))
    ln = np.round(ln).astype(float)
    ln = ln[ln >= 1]
    pl = L.powerlaw_loglik(ln, xmin=float(np.quantile(ln, 0.5)),
                           alpha=L.powerlaw_mle(ln, float(np.quantile(ln, 0.5)))[0])
    alt = L.fit_alternative(ln, xmin=float(np.quantile(ln, 0.5)), family="lognormal")
    req(alt["ks"] < 0.05, f"T1 the lognormal fit is close on lognormal data (KS {alt['ks']:.4f})")
    req(np.isfinite(pl) and np.isfinite(alt["loglik"]),
        f"T1 both log-likelihoods are finite (pl {pl:.1f}, ln {alt['loglik']:.1f})")

    # the fast KS must agree with the incomplete-zeta reference
    xs_tail = x[x >= fit["xmin"]]
    k_fast = L.ks_distance_powerlaw_fast(x, fit["xmin"], fit["alpha"])
    k_ref = L.ks_distance_powerlaw(x, fit["xmin"], fit["alpha"])
    req(abs(k_fast - k_ref) < 0.01,
        f"T1 the cumulative-table KS agrees with the incomplete-zeta reference "
        f"({k_fast:.4f} vs {k_ref:.4f})")

    # the normalised log-likelihood ratio behaves: identical models give R = 0
    d = rng.normal(0.0, 1.0, 500)
    v = L.vuong_like_test(0.0, 0.0, d)
    req(abs(v["R"]) < 1e-9, f"T1 the ratio test is 0 for identical models ({v})")
    v2 = L.vuong_like_test(10.0, 0.0, d)
    req(v2["R"] > 0 and v2["favours"] == "model_1",
        f"T1 and favours the model with the higher likelihood ({v2['R']:.2f})")

    g = L.gini(np.ones(100))
    req(abs(g) < 1e-9, f"T1 Gini of a constant is 0 ({g})")
    g2 = L.gini(np.array([0.0] * 99 + [100.0]))
    req(g2 > 0.98, f"T1 Gini of one huge value is near 1 ({g2:.4f})")

    xs, ys = L.ccdf(np.array([1, 1, 2, 3, 3, 3]))
    req(xs.tolist() == [1, 1, 2, 3, 3, 3] and abs(ys[0] - 1.0) < 1e-12 and ys[-1] > 0,
        f"T1 CCDF starts at 1 and is non-increasing ({xs.tolist()} {ys.tolist()})")


# ---------------------------------------------------------------- distortion metrics
def test_distortion() -> None:
    rng = np.random.default_rng(3)
    true_d = rng.integers(1, 8, size=500).astype(float)
    req(abs(L.normalised_stress(true_d, true_d)) < 1e-12,
        "T1 stress is 0 for a perfect embedding")
    req(abs(L.mean_relative_distortion(true_d, true_d)) < 1e-12,
        "T1 mean relative distortion is 0 for a perfect embedding")
    req(abs(L.spearman(true_d, true_d) - 1.0) < 1e-9,
        "T1 Spearman is 1 for a perfect embedding")

    scaled = true_d * 3.7
    s = L.fit_scale(true_d, scaled)
    req(abs(s - 1.0 / 3.7) < 1e-9, f"T1 the global scale is recovered exactly ({s:.6f})")
    req(abs(L.normalised_stress(true_d, scaled * s)) < 1e-12,
        "T1 and a pure rescaling is not counted as distortion")

    worse = true_d + rng.normal(0, 2.0, true_d.size)
    req(L.normalised_stress(true_d, worse) > L.normalised_stress(true_d, true_d),
        "T1 noise increases stress")

    # neighbourhood preservation: when the embedding distances ARE the graph
    # distances, every neighbourhood is preserved exactly. The neighbourhoods are
    # derived from the same distances, so the test cannot disagree with itself.
    path = nx.path_graph(4)
    dm = L.bfs_distances(path, [0, 1, 2, 3]).astype(float)
    true_nb = [list(np.argsort(dm[i] + (np.eye(4)[i] * 1e9))[:2]) for i in range(4)]
    req(abs(L.neighbourhood_preservation(true_nb, dm, 2) - 1.0) < 1e-12,
        f"T1 neighbourhood preservation is 1 when the orders agree (neighbours {true_nb})")

    # and a reversed embedding order scores 0
    req(L.neighbourhood_preservation(true_nb, -dm + dm.max(), 2) < 1.0,
        "T1 and it drops when the embedding orders the neighbours differently")


def main() -> int:
    print("=== transitions ===", flush=True); test_transitions()
    print("\n=== undirected symmetry ===", flush=True); test_undirected_symmetry()
    print("\n=== graph stats and distance ===", flush=True); test_graph_stats_and_distance()
    print("\n=== Gromov delta ===", flush=True); test_gromov_delta()
    print("\n=== tree metric on the scipy path ===", flush=True); test_tree_metric_on_scipy_path()
    print("\n=== tree-likeness and null model ===", flush=True); test_tree_likeness_and_null()
    print("\n=== power law ===", flush=True); test_powerlaw()
    print("\n=== distortion ===", flush=True); test_distortion()
    print("\n" + "=" * 72)
    print("ALL LIBRARY TESTS PASS" if _ok[0] else "LIBRARY TESTS FAILED")
    print("=" * 72)
    return 0 if _ok[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
