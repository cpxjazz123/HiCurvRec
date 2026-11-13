"""What did the duplicated orbit columns do to the roles?

The five-node run used to concatenate ORCA's 15 four-node columns with the 73 five-node ones, even
though the 15 four-node columns are byte-identical to the first 15 of the five-node output (checked on
all 24,587 nodes).  That made the standardised feature matrix rank deficient and weighted the
four-node information twice.

This script fits the roles both ways on the same graph and reports how far the two partitions are
apart -- so the change is a measured one rather than an assumption.

    python3 results/research/issue213/compare_feature_dims.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("graphlet_roles", HERE / "graphlet_roles.py")
gr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gr)


def main() -> int:
    import networkx as nx
    import pandas as pd

    edges = gr.edges_of()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    adj = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    degrees = dict(graph.degree())
    orbits4, used4 = gr.orca_orbits(gr.ORCA_NODE4, 15)
    orbits5, used5 = gr.orca_orbits(gr.ORCA_NODE5, 73)
    print(f"orbits: {used4} + {used5}", flush=True)

    identical = sum(1 for i in range(15) if np.array_equal(orbits4[:, i], orbits5[:, i]))
    print(f"four-node columns identical to the first 15 five-node columns: {identical}/15", flush=True)

    # the old matrix: 15 four-node columns followed by the 73 five-node ones, duplicates included
    stacked = np.hstack([orbits4, orbits5])
    nodes, role_88, _ = gr.features_and_roles(graph, degrees, adj, stacked)
    nodes2, role_73, _ = gr.features_and_roles(graph, degrees, adj, orbits5)

    a = np.asarray([role_88[node] for node in nodes])
    b = np.asarray([role_73[node] for node in nodes2])
    ari = float(adjusted_rand_score(a, b))
    print(f"adjusted Rand index between the two partitions: {ari:.4f}", flush=True)
    for label, roles in (("88 columns", a), ("73 columns", b)):
        sizes = [int((roles == r).sum()) for r in range(gr.ROLES)]
        print(f"  {label}: sizes {sizes}", flush=True)
    print(f"same assignment for {float((a == b).mean()):.4f} of the nodes", flush=True)

    out = {"ari": ari,
           "identical_columns": identical,
           "sizes_88": [int((a == r).sum()) for r in range(gr.ROLES)],
           "sizes_73": [int((b == r).sum()) for r in range(gr.ROLES)],
           "same_assignment_share": float((a == b).mean())}
    (HERE / "feature_dims_comparison.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
