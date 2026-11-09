"""Behaviour graph and its communities, for step 4 of the programme.

The behaviour graph is built from the training transitions only: an undirected
edge between a and b whenever either bought the other after the other, with the
weight equal to the number of such transitions. Undirected because a behaviour
relation is symmetric for the purpose of grouping, and weighted because some
pairs are purchased together far more often than others.

Communities come from Louvain with the seed fixed, so the grouping is a function
of the data and the seed alone. If the community count comes out above the L1
codebook capacity, the resolution is raised until it fits, and the count that was
actually used is recorded, because the cluster-to-codeword mapping is one to one
and cannot use more communities than there are codewords.

The resulting assignment is what the L1 consistency term supervises. Unlike the
soft-assignment overlap that failed in step 3, every item here has a non-degenerate
target, so nothing pushes the assignment toward a single codeword.

Written once and committed: the trainer loads the saved assignment rather than
recomputing it, so the grouping cannot drift between the diagnostic and the run.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_prefix_behavior as AP  # noqa: E402
import monotonicity_metrics as MM  # noqa: E402

SEED = 42
OUT_DIR = MM.OUT_DIR
OUT_NPZ = OUT_DIR / "behaviour_clusters_s42.npz"
OUT_META = OUT_DIR / "behaviour_clusters_s42.meta.json"
MAX_CLUSTERS = 256


def build_graph(n_items: int) -> nx.Graph:
    frame = pd.read_parquet(AP.DATA / "train.parquet")
    succ = AP.successors(frame, n_items)
    g = nx.Graph()
    # Isolated items are added explicitly: an item with no observed transition
    # still has to receive a cluster, otherwise it would be silently dropped from
    # the supervision.
    g.add_nodes_from(range(n_items))
    for a, targets in succ.items():
        for b in targets:
            if g.has_edge(a, b):
                g[a][b]["weight"] += 1.0
            else:
                g.add_edge(a, b, weight=1.0)
    return g


def louvain_within_capacity(g: nx.Graph) -> tuple[list[set[int]], dict]:
    """Louvain, resolution raised until the community count fits the codebook.

    The resolution search ascends and keeps the LARGEST community count that still
    fits inside the L1 codebook. Taking the first resolution that fits would pick
    the coarsest partition, which here is 39 communities and would leave 217 of
    the 256 codewords permanently unused: a codebook collapse by construction,
    which the programme explicitly forbids.
    """
    log: list[dict] = []
    best = None
    for resolution in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        comms = nx.community.louvain_communities(
            g, weight="weight", resolution=resolution, seed=SEED
        )
        log.append({"resolution": resolution, "n_communities": len(comms)})
        if len(comms) <= MAX_CLUSTERS and (best is None or len(comms) > len(best[1])):
            best = (resolution, [set(int(x) for x in c) for c in comms])
        if len(comms) > MAX_CLUSTERS and best is not None:
            break
    if best is not None:
        return best[1], {
            "chosen_resolution": best[0],
            "resolution_search": log,
            "selection_rule": "largest community count not exceeding the codebook capacity",
        }
    comms = nx.community.louvain_communities(
        g, weight="weight", resolution=64.0, seed=SEED
    )
    return [set(int(x) for x in c) for c in comms], {
        "chosen_resolution": 64.0,
        "resolution_search": log,
        "note": "no resolution kept the community count inside the codebook capacity",
    }


def main() -> None:
    g = build_graph(MM.N_ITEMS)
    nodes_in_graph = g.number_of_nodes()
    n_edges = g.number_of_edges()
    total_weight = float(sum(d["weight"] for _, _, d in g.edges(data=True)))

    communities, info = louvain_within_capacity(g)
    communities.sort(key=lambda c: -len(c))

    assignment = np.full(MM.N_ITEMS, -1, dtype=np.int32)
    for cid, members in enumerate(communities):
        for item in members:
            assignment[item] = cid
    unassigned = int((assignment < 0).sum())
    if unassigned:
        raise AssertionError(f"{unassigned} items received no cluster")

    sizes = np.bincount(assignment, minlength=len(communities))
    np.savez_compressed(OUT_NPZ, cluster=assignment)

    # Diagnostics the programme will need: does the clustering actually align with
    # behaviour, or is it just a partition of the popularity distribution?
    succ = AP.successors(pd.read_parquet(AP.DATA / "train.parquet"), MM.N_ITEMS)
    same, tot = 0, 0
    for a, targets in succ.items():
        for b in targets:
            tot += 1
            if assignment[a] == assignment[b]:
                same += 1
    l1 = Counter(assignment.tolist())
    meta = {
        "graph": {
            "nodes": nodes_in_graph,
            "edges": n_edges,
            "total_edge_weight": total_weight,
            "source": "train.parquet transitions, symmetrised, weight = transition count",
            "isolated_items_added_explicitly": int(MM.N_ITEMS - len(succ)),
        },
        "clustering": info,
        "n_clusters": len(communities),
        "l1_codebook_capacity": MAX_CLUSTERS,
        "seed": SEED,
        "cluster_size": {
            "min": int(sizes.min()), "max": int(sizes.max()),
            "median": float(np.median(sizes)),
            "mean": float(sizes.mean()),
            "singletons": int((sizes == 1).sum()),
        },
        "behaviour_alignment": {
            "transitions_total": tot,
            "transitions_within_cluster": same,
            "fraction_within": same / max(tot, 1),
        },
        "note": (
            "the L1 consistency term maps cluster c to L1 codeword index c, so the "
            "cluster count must not exceed the codebook capacity"
        ),
        "reproduce": (
            "cd /home/wlia0047/ar57/wenyu/GeneRec && OMP_NUM_THREADS=8 "
            "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 "
            "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts/build_behaviour_clusters.py"
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "resolution_search"}, indent=2))
    print("resolution search:", info.get("resolution_search"))
    print(f"written: {OUT_NPZ}\nwritten: {OUT_META}")


if __name__ == "__main__":
    main()