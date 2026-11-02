#!/usr/bin/env python3
"""The corrected curvature profile of the frozen baseline's prefixes.

Two nulls, because they answer different questions:

* the SIZE-PRESERVING SID-label shuffle: the items are relabelled into the
  prefixes with the group sizes kept, so it tests the alignment of the
  prefixes with the behaviour;
* the DEGREE-PRESERVING edge-rewired graph: the behaviour graph's edges are
  swapped, so it tests the structure BEYOND the degree.

The 1/2-hop ground metric is used for the sweep and is verified against the
exact metric on a sample of edges; the verification numbers travel with the
profile.

    python3 tools/research/science/curvature_profile.py --out <path>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO))

from tools.research.science import quotient_curvature as qc   # noqa: E402

TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
SID = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"
SEED = 20261102
NULLS = 2
EDGES = 16


def rewired_graph(graph, rng):
    """A degree-preserving rewiring of the behaviour graph."""
    import networkx as nx

    network = nx.from_scipy_sparse_array(graph)
    swaps = min(3 * network.number_of_edges(), 300_000)
    nx.double_edge_swap(network, nswap=swaps, max_tries=swaps * 10, seed=int(rng.integers(1 << 30)))
    return nx.to_scipy_sparse_array(network, format="csr")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exact", action="store_true",
                        help="use the exact shortest-path metric (slow)")
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_parquet(TRAIN, columns=["history", "target"])
    nodes, index, graph = qc.build_behaviour_graph(frame)
    sid = json.loads(SID.read_text())
    tokens = np.array([sid[str(item)][:3] for item in range(len(sid))])
    rng = np.random.default_rng(SEED)

    rewired = rewired_graph(graph, rng)
    levels = []
    for level in range(3):
        groups = {}
        assignment = np.full(len(nodes), -1, dtype=np.int64)
        for item in range(len(tokens)):
            node = index.get(item)
            if node is None:
                continue
            key = tuple(int(value) for value in tokens[item][:level + 1])
            groups.setdefault(key, len(groups))
            assignment[node] = groups[key]
        n_groups = len(groups)
        started = time.time()
        result = qc.standardised_curvature(graph, assignment, n_groups, rng,
                                           nulls=NULLS, edges=EDGES, exact=args.exact)
        verification = qc.verify_bounded_metric(
            qc.quotient_graph(graph, assignment, n_groups), rng, edges=4)
        rewired_result = qc.standardised_curvature(rewired, assignment, n_groups, rng,
                                                  nulls=NULLS, edges=EDGES, exact=args.exact)
        levels.append({
            "level": level,
            "prefix_groups": n_groups,
            "kappa": result["kappa"],
            "kappa_hat": result["kappa_hat"],
            "null_mean": result["null_mean"],
            "null_std": result["null_std"],
            "status": result["status"],
            "rewired_kappa_hat": rewired_result["kappa_hat"],
            "rewired_status": rewired_result["status"],
            "bounded_vs_exact": verification,
            "seconds": round(time.time() - started, 1),
        })
        print("level %d: kappa_hat %s (status %s), rewired %s, verification %s, %.1fs" % (
            level, result["kappa_hat"], result["status"],
            rewired_result["kappa_hat"], verification.get("mean_abs_difference"),
            time.time() - started), flush=True)

    values = np.array([level["kappa_hat"] if level["kappa_hat"] is not None else 0.0
                       for level in levels])
    delta = np.clip(values - values.mean(), -1.0, 1.0)
    curvature = np.exp(delta)
    payload = {
        "kappa_hat": [float(value) for value in values],
        "delta": [float(value) for value in delta],
        "curvature": [float(value) for value in curvature],
        "levels": levels,
        "source": ("the frozen baseline TIGER_RQ-VAE prefixes, measured on the UNDIRECTED "
                   "training behaviour graph with the 1/2-hop ground metric (verified "
                   "against the exact metric per level)"),
        "nulls": "size-preserving member shuffle (%d relabellings) and a "
                 "degree-preserving rewiring of the behaviour graph" % NULLS,
        "edges_per_level": EDGES,
        "seed": SEED,
        "estimator": {"ground_metric": "1/2-hop bound, verified against the exact "
                                        "unweighted shortest path",
                      "transport": "exact linear program",
                      "graph": "undirected (symmetrised), no self-loops"},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
