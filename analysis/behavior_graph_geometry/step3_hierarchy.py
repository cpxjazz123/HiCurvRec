"""Step 3, RQ2: community, branching and hierarchical structure of the real graph.

Communities come from Louvain at several resolutions (networkx, fixed seed), and the
same measurements are taken on a DEGREE-PRESERVING randomised graph, because a
community partition on its own is not evidence of hierarchy: the null model is what
says whether the partition is beyond what the degree sequence already implies.

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/behavior_graph_geometry/step3_hierarchy.py
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

RESOLUTIONS = (0.5, 1.0, 1.5, 2.0, 3.0)


def louvain(g: nx.Graph, resolution: float, seed: int) -> tuple[list, float]:
    parts = nx.community.louvain_communities(g, weight=None, resolution=resolution,
                                            seed=seed)
    mod = nx.community.modularity(g, parts)
    return parts, float(mod)


def community_block(g: nx.Graph, seed: int) -> dict:
    out = {}
    for res in RESOLUTIONS:
        t = time.time()
        parts, mod = louvain(g, res, seed)
        sizes = np.array(sorted((len(p) for p in parts), reverse=True))
        comm_of = {}
        for i, p in enumerate(parts):
            for v in p:
                comm_of[v] = i
        cross = 0
        for a, b in g.edges():
            if comm_of[a] != comm_of[b]:
                cross += 1
        out[f"resolution_{res}"] = {
            "n_communities": int(len(parts)),
            "modularity": mod,
            "largest_community": int(sizes[0]),
            "largest_community_fraction": float(sizes[0] / g.number_of_nodes()),
            "sizes_top10": sizes[:10].tolist(),
            "size_mean": float(sizes.mean()),
            "size_median": float(np.median(sizes)),
            "size_gini": L.gini(sizes.astype(np.float64)),
            "cross_community_edge_fraction": float(cross / max(g.number_of_edges(), 1)),
            "seconds": round(time.time() - t, 2),
        }
    return out


def main() -> None:
    t0 = time.time()
    lcc = np.load(C.OUT_DIR / "lcc_edges.npz")
    nodes = lcc["nodes"]
    edges = lcc["edges"]
    g = nx.Graph()
    g.add_nodes_from(range(len(nodes)))
    g.add_edges_from(map(tuple, edges))
    arrays = np.load(C.OUT_DIR / "graph_arrays.npz")

    block = {
        "graph": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                  "note": "undirected largest connected component, unweighted"},
        "seed": C.SEED,
    }
    block["communities_real"] = community_block(g, C.SEED)
    print(f"[t={time.time()-t0:.1f}s] real communities done", flush=True)

    nulls = L.degree_preserving_null(g, swaps_per_edge=C.NULL_SWAPS_PER_EDGE,
                                     seed=C.SEED, replicates=C.NULL_REPLICATES)
    print(f"[t={time.time()-t0:.1f}s] null graphs built", flush=True)
    block["communities_null"] = []
    for i, h in enumerate(nulls):
        cb = community_block(h, C.SEED + i)
        block["communities_null"].append(cb)
        r = cb["resolution_1.0"]
        print(f"[t={time.time()-t0:.1f}s] null {i}: communities={r['n_communities']} "
              f"modularity={r['modularity']:.4f}", flush=True)

    # branching
    out_nb = arrays["out_neighbours"]
    pos = out_nb[out_nb > 0]
    block["branching"] = {
        "distinct_successors_mean": float(pos.mean()),
        "distinct_successors_median": float(np.median(pos)),
        "distinct_successors_p90": float(np.quantile(pos, 0.90)),
        "distinct_successors_p99": float(np.quantile(pos, 0.99)),
        "distinct_successors_max": int(pos.max()),
        "share_with_single_successor": float((pos == 1).mean()),
        "share_with_at_least_10_successors": float((pos >= 10).mean()),
        "top1pct_share_of_all_successor_slots": float(
            np.sort(pos)[::-1][: max(1, int(0.01 * pos.size))].sum() / pos.sum()),
    }

    # tree-likeness on the real graph and on the null models
    root = int(max(g.degree, key=lambda kv: kv[1])[0])
    block["tree_likeness_real"] = L.tree_likeness(g, root, sources=200, seed=C.SEED)
    print(f"[t={time.time()-t0:.1f}s] tree-likeness real: non_tree_edges="
          f"{block['tree_likeness_real']['non_tree_edges']} "
          f"preserved={block['tree_likeness_real']['share_of_pairs_exactly_preserved']:.4f}",
          flush=True)
    block["tree_likeness_null"] = [
        L.tree_likeness(h, int(max(h.degree, key=lambda kv: kv[1])[0]), sources=200, seed=C.SEED + i)
        for i, h in enumerate(nulls)
    ]

    # local structure and direction asymmetry summary
    block["local_structure"] = {
        "average_clustering": L.average_clustering_fast(g)["average_clustering"],
        "transitivity": L.average_clustering_fast(g)["global_clustering_transitivity"],
        "degree_assortativity": float(nx.degree_assortativity_coefficient(g)),
        "path_metrics_sampled": L.path_metrics_sampled(g, seed=C.SEED, sources=400),
    }
    stats = json.loads((C.OUT_DIR / "behavior_graph_statistics.json").read_text(encoding="utf-8"))
    block["direction_asymmetry"] = stats["direction_asymmetry"]

    stats["hierarchy_analysis"] = block
    (C.OUT_DIR / "behavior_graph_statistics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    (C.OUT_DIR / "behavior_hierarchy_analysis.json").write_text(
        json.dumps(block, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    real = block["communities_real"]["resolution_1.0"]
    ax = axes[0]
    ax.bar(range(len(real["sizes_top10"])), real["sizes_top10"])
    ax.set_title(f"largest communities (real, res 1.0)\n{real['n_communities']} communities, "
                 f"modularity {real['modularity']:.3f}", fontsize=9)
    ax.set_xlabel("rank"); ax.set_ylabel("size")
    ax = axes[1]
    for name, cb, c in (("real", block["communities_real"], "C0"),):
        xs = [cb[f"resolution_{r}"]["n_communities"] for r in RESOLUTIONS]
        ys = [cb[f"resolution_{r}"]["modularity"] for r in RESOLUTIONS]
        ax.plot(xs, ys, "o-", label=name, color=c)
    for i, ncb in enumerate(block["communities_null"]):
        xs = [ncb[f"resolution_{r}"]["n_communities"] for r in RESOLUTIONS]
        ys = [ncb[f"resolution_{r}"]["modularity"] for r in RESOLUTIONS]
        ax.plot(xs, ys, "s--", alpha=0.5, label=f"degree-preserving null {i}")
    ax.set_title("modularity vs number of communities\nreal against degree-preserving null", fontsize=9)
    ax.set_xlabel("#communities"); ax.set_ylabel("modularity"); ax.legend(fontsize=7)
    ax = axes[2]
    tr = block["tree_likeness_real"]
    tn = [x["share_of_pairs_exactly_preserved"] for x in block["tree_likeness_null"]]
    ax.bar(["real"] + [f"null {i}" for i in range(len(tn))],
           [tr["share_of_pairs_exactly_preserved"]] + tn)
    ax.set_title("BFS-tree distance exactly preserved\nhigher = more tree-like", fontsize=9)
    ax.set_ylabel("share of sampled pairs")
    fig.tight_layout()
    fig.savefig(C.OUT_DIR / "hierarchy_communities.png", dpi=C.FIG_DPI)

    print(f"\nwritten: {C.OUT_DIR/'behavior_hierarchy_analysis.json'}")
    print(f"written: {C.OUT_DIR/'hierarchy_communities.png'}")
    print(f"[t={time.time()-t0:.1f}s] total", flush=True)


if __name__ == "__main__":
    main()