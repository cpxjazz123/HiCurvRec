"""Is the role assignment reproducible? Fit the same features twice and compare.

The role partition enters every retention number, so if two runs of the same code on the same graph
produce different partitions, the correlations that are read off a role are not attached to a stable
grouping.  This script fits the five-node features twice, in one process, and reports how much of
the assignment agrees.

    python3 results/research/issue213/check_role_determinism.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("graphlet_roles", HERE / "graphlet_roles.py")
gr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gr)


def main() -> int:
    edges = gr.edges_of()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    adj = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    degrees = dict(graph.degree())
    orbits5, used5 = gr.orca_orbits(gr.ORCA_NODE5, 73)
    print(f"orbits: {used5}", flush=True)

    nodes_a, role_a, _ = gr.features_and_roles(graph, degrees, adj, orbits5)
    nodes_b, role_b, _ = gr.features_and_roles(graph, degrees, adj, orbits5)
    a = np.asarray([role_a[node] for node in nodes_a])
    b = np.asarray([role_b[node] for node in nodes_b])
    same = float((a == b).mean())
    ari = float(adjusted_rand_score(a, b))
    print(f"identical assignment for {same:.4f} of the nodes; adjusted Rand {ari:.4f}", flush=True)
    for label, roles in (("first", a), ("second", b)):
        print(f"  {label}: sizes {[int((roles == r).sum()) for r in range(gr.ROLES)]}", flush=True)
    out = {"same_assignment_share": same, "adjusted_rand": ari,
           "sizes_first": [int((a == r).sum()) for r in range(gr.ROLES)],
           "sizes_second": [int((b == r).sum()) for r in range(gr.ROLES)]}
    (HERE / "role_determinism.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
