"""Step 4, RQ3: Gromov hyperbolicity of the real behaviour graph.

Four-point delta on the unweighted undirected largest connected component, by seeded
Monte Carlo sampling, against a degree-preserving randomised null and a tree reference.
The absolute delta is reported together with delta/diameter, because a random graph with
a small diameter also produces a small absolute delta and the two must not be confused.

Strata: uniform anchors, anchors on the most popular items, anchors on the least
popular ones, and a subgraph-size sensitivity sweep.

Run: /home/wlia0047/ar57/scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/behavior_graph_geometry/step4_gromov.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import graph_lib as L  # noqa: E402


def delta_for_graph(g: nx.Graph, nodes: np.ndarray, seed: int, n_sources: int,
                    n_quads: int, popularity: np.ndarray | None = None,
                    label: str = "uniform") -> dict:
    rng = np.random.default_rng(seed)
    if popularity is None:
        src_idx = rng.choice(len(nodes), size=min(n_sources, len(nodes)), replace=False)
    else:
        # anchors restricted to a popularity stratum, taken from the FULL item universe
        order = np.argsort(popularity)
        if label == "popular":
            pool = order[-max(1, len(order) // 100):]          # top 1%
        elif label == "low_popularity":
            pool = order[: len(order) // 2]                     # bottom half
        else:
            pool = order
        chosen = rng.choice(pool, size=min(n_sources, len(pool)), replace=False)
        src_idx = np.searchsorted(nodes, chosen).clip(0, len(nodes) - 1)

    D = L.bfs_distances_scipy(g, nodes[src_idx].tolist())
    # D's rows are the sampled sources but its columns are EVERY node, so a
    # quadruple of source indices has to be read through the source-order
    # submatrix; gromov_delta_matrix(D, q) directly would treat a source index as a
    # node position and return a plausible-looking, entirely wrong delta.
    M = L.source_submatrix(D, src_idx)
    q = rng.choice(M.shape[0], size=(n_quads, 4), replace=True)
    delta = L.gromov_delta_matrix(M, q)
    finite = np.isfinite(delta)
    # quadruples that are disconnected inside the sampled sources are not comparable
    summ = L.delta_summary(delta[finite])
    summ["stratum"] = label
    summ["mean_over_finite_quarter"] = float(np.mean(delta[finite] > 0)) if finite.any() else 0.0
    summ["n_sources"] = int(D.shape[0])
    summ["n_quads_requested"] = int(n_quads)
    summ["n_quads_finite"] = int(finite.sum())
    summ["n_quads_disconnected"] = int((~finite).sum())
    return summ


def diameter_of(g: nx.Graph, seed: int = 42, sources: int = 400) -> int:
    pm = L.path_metrics_sampled(g, seed=seed, sources=sources)
    return int(pm["diameter_estimate"])


def main() -> None:
    t0 = time.time()
    lcc = np.load(C.OUT_DIR / "lcc_edges.npz")
    nodes = lcc["nodes"]
    edges = lcc["edges"]
    arrays = np.load(C.OUT_DIR / "graph_arrays.npz")
    popularity = arrays["interactions"].astype(np.float64)

    # The graph keeps the ORIGINAL item ids, so `nodes` indexes into node list()
    # directly and every lookup stays in the item id space.
    g = nx.Graph()
    g.add_nodes_from(nodes.tolist())
    g.add_edges_from((int(nodes[a]), int(nodes[b])) for a, b in edges)

    diam = diameter_of(g, C.SEED, sources=400)
    print(f"[t={time.time()-t0:.1f}s] LCC {g.number_of_nodes()}/{g.number_of_edges()}, "
          f"diameter {diam}", flush=True)

    block = {
        "graph": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                  "diameter_estimate": diam, "diameter_method": "seeded 400-source BFS"},
        "seed": C.SEED,
        "method": {
            "distance": "unweighted shortest path on the undirected LCC; symmetric and "
                        "triangle-inequality satisfying by construction, which the "
                        "four-point condition requires",
            "delta": "(largest - second largest) / 2 of the three pair sums",
            "sampling": "seeded Monte Carlo over quadruples of a seeded source sample; the "
                        "sampled estimate is NOT the exact hyperbolicity constant",
            "normalisation": "delta/diameter, because a small-diameter random graph also "
                             "gives a small absolute delta",
            "n_quads": C.GROMOV_QUADS,
            "n_sources": 2000,
        },
        "real": {},
        "null": [],
        "tree_reference": {},
        "sensitivity": {},
        "subgraph_size_sweep": [],
    }

    for label in ("uniform", "popular", "low_popularity"):
        s = delta_for_graph(g, nodes, C.SEED, 2000, C.GROMOV_QUADS,
                            popularity=popularity, label=label)
        s["delta_over_diameter"] = s["median"] / max(diam, 1)
        block["real"][label] = s
        print(f"[t={time.time()-t0:.1f}s] real/{label}: median delta={s['median']:.4f} "
              f"mean={s['mean']:.4f} p95={s['p95']:.4f} delta/diam={s['delta_over_diameter']:.4f} "
              f"(disconnected quads {s['n_quads_disconnected']})", flush=True)

    # sample-size sensitivity
    for nq in (25_000, 50_000, 200_000):
        s = delta_for_graph(g, nodes, C.SEED, 2000, nq, label=f"quads_{nq}")
        s["delta_over_diameter"] = s["median"] / max(diam, 1)
        block["sensitivity"][f"n_quads_{nq}"] = s
        print(f"[t={time.time()-t0:.1f}s] sensitivity n_quads={nq}: median={s['median']:.4f} "
              f"se={s['bootstrap_se_of_mean']:.5f}", flush=True)

    # subgraph size sensitivity
    rng = np.random.default_rng(C.SEED)
    for frac in (0.05, 0.20, 0.50):
        keep = np.sort(rng.choice(len(nodes), size=max(50, int(frac * len(nodes))),
                                   replace=False))
        sg = g.subgraph(keep.tolist())
        # an induced subgraph is usually disconnected; distances are only a metric
        # inside a connected component, so the sweep runs on its largest one
        if sg.number_of_nodes() > 1:
            sg = sg.subgraph(max(nx.connected_components(sg), key=len)).copy()
        sg_nodes = np.array(sorted(sg.nodes()))
        sd = diameter_of(sg, C.SEED, sources=200)
        s = delta_for_graph(sg, sg_nodes, C.SEED, 800, 50_000, label=f"subgraph_{frac}")
        s["subgraph_diameter"] = sd
        s["delta_over_diameter"] = s["median"] / max(sd, 1)
        block["subgraph_size_sweep"].append({"fraction": frac, **s})
        print(f"[t={time.time()-t0:.1f}s] subgraph {frac}: n={sg.number_of_nodes()} "
              f"diam={sd} median delta={s['median']:.4f}", flush=True)

    # degree-preserving nulls
    nulls = L.degree_preserving_null(g, swaps_per_edge=C.NULL_SWAPS_PER_EDGE,
                                     seed=C.SEED, replicates=C.NULL_REPLICATES)
    print(f"[t={time.time()-t0:.1f}s] null graphs built", flush=True)
    for i, h in enumerate(nulls):
        nd = diameter_of(h, C.SEED, sources=200)
        s = delta_for_graph(h, np.array(sorted(h.nodes())), C.SEED, 1500, 100_000,
                            label=f"null_{i}")
        s["diameter_estimate"] = nd
        s["delta_over_diameter"] = s["median"] / max(nd, 1)
        block["null"].append(s)
        print(f"[t={time.time()-t0:.1f}s] null {i}: diameter={nd} median delta={s['median']:.4f} "
              f"delta/diam={s['delta_over_diameter']:.4f}", flush=True)

    # tree references: the graph's own BFS tree, and a random tree of the same size
    root = int(max(g.degree, key=lambda kv: kv[1])[0])
    bfs = nx.bfs_tree(g, root)
    td = diameter_of(bfs, C.SEED, sources=400)
    s = delta_for_graph(bfs, np.array(sorted(bfs.nodes())), C.SEED, 1500, 100_000,
                        label="bfs_tree_of_real_graph")
    s["diameter_estimate"] = td
    s["delta_over_diameter"] = s["median"] / max(td, 1)
    block["tree_reference"]["bfs_tree"] = s
    rt = nx.random_labeled_tree(n=min(g.number_of_nodes(), 5000), seed=C.SEED)
    rtd = diameter_of(rt, C.SEED, sources=300)
    s2 = delta_for_graph(rt, np.array(sorted(rt.nodes())), C.SEED, 1500, 100_000,
                         label="random_labeled_tree_5000")
    s2["diameter_estimate"] = rtd
    s2["delta_over_diameter"] = s2["median"] / max(rtd, 1)
    block["tree_reference"]["random_tree_5000"] = s2
    print(f"[t={time.time()-t0:.1f}s] tree refs: bfs diameter={td} median delta={s['median']:.4f}; "
          f"random tree diameter={rtd} median delta={s2['median']:.4f}", flush=True)

    (C.OUT_DIR / "gromov_hyperbolicity.json").write_text(json.dumps(block, indent=2),
                                                        encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    rng = np.random.default_rng(0)
    nodes_l = list(g.nodes())[:2000]
    D = L.bfs_distances_scipy(g, nodes_l)
    M = L.source_submatrix(D, np.arange(len(nodes_l), dtype=np.int64))
    d = L.gromov_delta_matrix(M, rng.choice(M.shape[0], size=(20000, 4)))
    ax.hist(d[np.isfinite(d)], bins=80, alpha=0.7, label="real graph")
    ax.set_title("four-point delta distribution (real graph)", fontsize=9)
    ax.set_xlabel("delta"); ax.set_ylabel("count"); ax.legend(fontsize=7)
    ax = axes[1]
    labels = ["real"] + [f"null {i}" for i in range(len(block["null"]))] + ["bfs tree"]
    vals = [block["real"]["uniform"]["median"]] \
        + [n["median"] for n in block["null"]] \
        + [block["tree_reference"]["bfs_tree"]["median"]]
    ax.bar(labels, vals)
    ax.set_title("median delta: real vs degree-preserving null vs tree", fontsize=9)
    ax.set_ylabel("median delta")
    ax = axes[2]
    labels2 = ["real"] + [f"null {i}" for i in range(len(block["null"]))] + ["bfs tree"]
    vals2 = [block["real"]["uniform"]["delta_over_diameter"]] \
        + [n["delta_over_diameter"] for n in block["null"]] \
        + [block["tree_reference"]["bfs_tree"]["delta_over_diameter"]]
    ax.bar(labels2, vals2)
    ax.set_title("delta / diameter (diameter-normalised)", fontsize=9)
    ax.set_ylabel("median delta / diameter")
    fig.tight_layout()
    fig.savefig(C.OUT_DIR / "gromov_delta.png", dpi=C.FIG_DPI)

    print(f"\nwritten: {C.OUT_DIR/'gromov_hyperbolicity.json'}")
    print(f"written: {C.OUT_DIR/'gromov_delta.png'}")
    print(f"[t={time.time()-t0:.1f}s] total", flush=True)


if __name__ == "__main__":
    main()