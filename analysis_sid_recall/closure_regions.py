"""Do the high-closure nodes sit together, or is closure just spread out?

Step 1 picks the top 1%, 5% and 10% of nodes by local clustering (degree < 2
counted as 0).  Step 2 asks whether those nodes form a region: how many edges
run inside the selection, whether they fall in a few communities, and how much
their neighbourhoods overlap.  Step 3 puts the same question to two nulls --
degree-matched random nodes, and a degree-preserving rewiring of the same graph
-- because high-clustering nodes are also high-degree nodes, and a selection
that is simply "the busy nodes" would look concentrated for free.

    python3 analysis_sid_recall/closure_regions.py
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
OUT = Path(__file__).resolve().parent / "closure_regions.json"
SEED = 42
SHARES = (0.01, 0.05, 0.10)
SWAP_FACTOR = 2


def edges() -> np.ndarray:
    frame = pd.read_parquet(TRAIN)
    chunks = []
    for history in frame["history"]:
        if history is None or len(history) < 2:
            continue
        items = np.asarray([int(x) for x in history])
        chunks.append(np.stack([items[:-1], items[1:]], axis=1))
    pairs = np.unique(np.sort(np.concatenate(chunks), axis=1), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def graph(pairs: np.ndarray):
    nodes, inverse = np.unique(pairs, return_inverse=True)
    index = inverse.reshape(-1, 2)
    A = sparse.coo_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])),
                          shape=(len(nodes), len(nodes))).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    return nodes, index, A


def clustering(A: sparse.csr_matrix, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deg = np.asarray(A.sum(1)).ravel()
    S = A @ A
    node_tri = np.asarray(S.multiply(A).sum(1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(deg >= 2, node_tri / (deg * (deg - 1.0)), 0.0)
    return deg, c


def rewire(index: np.ndarray, n: int, factor: int = SWAP_FACTOR, seed: int = SEED) -> np.ndarray:
    """Degree-preserving double-edge swaps."""
    rng = np.random.default_rng(seed)
    e = index.copy()
    present = {tuple(sorted(pair)) for pair in map(tuple, e)}
    swaps = 0
    target = factor * len(e)
    while swaps < target:
        i, j = rng.integers(0, len(e), 2)
        a, b = e[i]
        c, d = e[j]
        if len({a, b, c, d}) < 4:
            continue
        new1 = tuple(sorted((a, c)))
        new2 = tuple(sorted((b, d)))
        if new1 in present or new2 in present or new1[0] == new1[1] or new2[0] == new2[1]:
            new1 = tuple(sorted((a, d)))
            new2 = tuple(sorted((b, c)))
            if new1 in present or new2 in present or new1[0] == new1[1] or new2[0] == new2[1]:
                continue
        present.discard(tuple(sorted((a, b))))
        present.discard(tuple(sorted((c, d))))
        present.add(new1)
        present.add(new2)
        e[i] = new1
        e[j] = new2
        swaps += 1
    return e


def induced_stats(A, selected: np.ndarray, deg: np.ndarray, communities: np.ndarray) -> dict:
    n = A.shape[0]
    mask = np.zeros(n, bool)
    mask[selected] = True
    sub = A[selected][:, selected]
    inside = int(sub.nnz // 2)
    possible = len(selected) * (len(selected) - 1) / 2
    # neighbour share: of a selected node's neighbours, how many are selected too
    neighbour_share = np.asarray(sub.sum(1)).ravel() / np.maximum(deg[selected], 1)
    labels, counts = np.unique(communities[selected], return_counts=True)
    return {
        "selected": int(len(selected)),
        "mean_degree_of_selected": float(deg[selected].mean()),
        "edges_inside": inside,
        "density_inside": float(inside / possible) if possible else None,
        "mean_share_neighbours_selected": float(neighbour_share.mean()),
        "communities_touched": int(len(labels)),
        "share_in_top3_communities": float(np.sort(counts)[::-1][:3].sum() / len(selected)),
        "share_in_largest_community": float(counts.max() / len(selected)),
    }


def hop_distance(G: nx.Graph, selected: np.ndarray, sample: int, seed: int) -> float | None:
    """Mean hops from a selected node to its nearest other selected node."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(selected, size=min(sample, len(selected)), replace=False)
    targets = set(int(x) for x in selected)
    distances = []
    for start in picks:
        seen = {int(start)}
        frontier = [int(start)]
        depth = 0
        found = None
        while frontier and found is None:
            depth += 1
            nxt = []
            for node in frontier:
                for neighbour in G[node]:
                    if neighbour in seen:
                        continue
                    if neighbour in targets:
                        found = depth
                        break
                    seen.add(neighbour)
                    nxt.append(neighbour)
                if found is not None:
                    break
            frontier = nxt
        if found is not None:
            distances.append(found)
    return float(np.mean(distances)) if distances else None


def degree_matched_sample(deg: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Nodes drawn so the degree multiset matches the selected one."""
    pool = np.argsort(deg)
    out = set()
    for value in np.sort(deg[np.random.default_rng(SEED).choice(len(deg), size, replace=False)]):
        lo = np.searchsorted(deg[pool], value, side="left")
        hi = np.searchsorted(deg[pool], value, side="right")
        candidates = pool[lo:hi]
        pick = int(rng.choice(candidates))
        while pick in out:
            pick = int(rng.choice(candidates))
        out.add(pick)
    return np.array(sorted(out))


def main() -> None:
    pairs = edges()
    nodes, index, A = graph(pairs)
    deg, c = clustering(A, index)
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(map(tuple, index))
    communities_list = nx.community.louvain_communities(G, seed=SEED)
    communities = np.zeros(len(nodes), dtype=int)
    for i, block in enumerate(communities_list):
        communities[list(block)] = i
    order = np.lexsort((np.arange(len(nodes)), -deg, -c))
    rng = np.random.default_rng(SEED)
    null_pairs = rewire(index, len(nodes))
    _, null_index, null_A = graph(np.vstack([null_pairs]))
    null_deg, null_c = clustering(null_A, null_index)
    report = {"nodes": int(len(nodes)), "edges": int(len(index)),
              "clustering_real": {"mean": float(c.mean()),
                                   "share_ge_0_3": float((c >= 0.3).mean()),
                                   "triangles": int(np.asarray(A.multiply(A @ A).sum()) / 6)},
              "clustering_rewired": {"mean": float(null_c.mean()),
                                      "share_ge_0_3": float((null_c >= 0.3).mean()),
                                      "triangles": int(np.asarray(null_A.multiply(null_A @ null_A).sum()) / 6)},
              "communities": int(len(communities_list)),
              "modularity": float(nx.community.modularity(G, communities_list)),
              "shares": {}}
    for share in SHARES:
        k = max(10, int(share * len(nodes)))
        selected = order[:k]
        real = induced_stats(A, selected, deg, communities)
        matched = degree_matched_sample(deg, k, rng)
        random_stats = induced_stats(A, matched, deg, communities)
        # the same nodes evaluated on the rewired graph: closure there is gone by construction,
        # so the interesting comparison is their induced density and neighbour overlap
        null_stats = induced_stats(null_A, selected, null_deg, communities)
        hops_real = hop_distance(G, selected, 100, SEED)
        hops_rand = hop_distance(G, matched, 100, SEED)
        report["shares"]["top%g%%" % (share * 100)] = {
            "real": real, "degree_matched_random_nodes": random_stats,
            "same_nodes_on_rewired_graph": null_stats,
            "mean_hops_between_selected": hops_real,
            "mean_hops_between_matched": hops_rand,
            "density_ratio_vs_matched": (real["density_inside"] / random_stats["density_inside"]
                                         if random_stats["density_inside"] else None),
            "neighbour_share_ratio_vs_matched": (real["mean_share_neighbours_selected"]
                                                 / random_stats["mean_share_neighbours_selected"]),
        }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("nodes %d edges %d communities %d modularity %.3f"
          % (report["nodes"], report["edges"], report["communities"], report["modularity"]))
    print("clustering  real mean %.4f share>=0.3 %.4f triangles %d | rewired mean %.4f share>=0.3 %.4f triangles %d"
          % (report["clustering_real"]["mean"], report["clustering_real"]["share_ge_0_3"],
             report["clustering_real"]["triangles"], report["clustering_rewired"]["mean"],
             report["clustering_rewired"]["share_ge_0_3"], report["clustering_rewired"]["triangles"]))
    for name, block in report["shares"].items():
        real, rand = block["real"], block["degree_matched_random_nodes"]
        print("%s k=%d  density %.3f vs matched %.3f (x%.2f)  neighbour-share %.3f vs %.3f (x%.2f)"
              % (name, real["selected"], real["density_inside"], rand["density_inside"],
                 block["density_ratio_vs_matched"], real["mean_share_neighbours_selected"],
                 rand["mean_share_neighbours_selected"], block["neighbour_share_ratio_vs_matched"]))
        print("     top-3 communities hold %.2f of the selection (matched %.2f); largest %.2f"
              % (real["share_in_top3_communities"], rand["share_in_top3_communities"],
                 real["share_in_largest_community"]))
        few = int((deg[selected] <= 3).sum())
        print("     degree of the selection: mean %.1f, %d of %d have degree <= 3"
              % (real["mean_degree_of_selected"], few, real["selected"]))
        print("     mean hops to nearest other selected %.2f (matched %.2f); inside edges %d"
              % (block["mean_hops_between_selected"], block["mean_hops_between_matched"],
                 real["edges_inside"]))


if __name__ == "__main__":
    main()
