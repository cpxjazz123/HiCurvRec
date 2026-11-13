"""Freeze the ORCA role table for issue #215 and record its identity.

The roles are the only behaviour signal the arms may use, so they are fitted once, written to disk
with a sha256, and read from there by every arm -- training, validation and SID export alike.  The
freeze is also what makes the arms reproducible: NMF is not bit-reproducible (see
results/research/issue213/check_role_determinism.py), so a role table that is refitted at every run
would silently be a different table each time.

    python3 results/research/issue215/freeze_roles.py

Writes:
    results/research/issue215/orca_roles_frozen.csv    item_id, role
    results/research/issue215/orca_roles_meta.json      feature source, sha256s, sizes, centroids
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("graphlet_roles", REPO_ROOT / "results" / "research" / "issue213" / "graphlet_roles.py")
gr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gr)

OUT_CSV = HERE / "orca_roles_frozen.csv"
OUT_META = HERE / "orca_roles_meta.json"

#: the fixed role -> geometry mapping of issue #215, in the role order the audit reports
ROLE_SPACE = {0: "euclidean", 1: "spherical", 2: "hyperbolic",
              3: "euclidean", 4: "spherical", 5: "hyperbolic"}


def main() -> int:
    edges = gr.edges_of()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    adj = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    degrees = dict(graph.degree())
    orbits5, used5 = gr.orca_orbits(gr.ORCA_NODE5, 73)
    if used5 is None:
        raise SystemExit("no node5.out: build it with the accelerated ORCA first")
    print(f"orbits: {used5}", flush=True)

    nodes, role_of, profiles = gr.features_and_roles(graph, degrees, adj, orbits5)
    graph_nodes = set(graph.nodes())

    # every item the arms can see: the SID tables carry 24,587 items, the graph has 24,556 nodes
    # (the rest have no behaviour edge at all).  Those get the documented fallback role and are
    # counted, not dropped.
    import pandas as pd
    frame = pd.read_parquet(REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["seen_history", "target"])
    seen = set()
    for history, target in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)):
        if history is not None and len(history):
            seen.add(int(history[-1]))
        seen.add(int(target))
    seen |= graph_nodes
    missing = sorted(seen - graph_nodes)

    fallback_role = 0                                   # euclidean: the conservative route
    lines = ["item_id,role"]
    for item in sorted(seen):
        role = role_of.get(item, fallback_role)
        lines.append(f"{item},{role}")
    OUT_CSV.write_text("\n".join(lines) + "\n")
    digest = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()

    orbits_digest = hashlib.sha256((REPO_ROOT / used5).read_bytes()).hexdigest()
    meta = {
        "role_table": str(OUT_CSV.relative_to(REPO_ROOT)),
        "role_table_sha256": digest,
        "role_table_rows": len(seen),
        "items_without_a_role": len(missing),
        "fallback_role": fallback_role,
        "fallback_role_note": "items with no TRAIN behaviour edge keep the euclidean route; the count is above",
        "feature": {"source": used5, "sha256": orbits_digest, "columns": 73,
                    "note": "ORCA's five-node orbits only; the fifteen four-node columns are byte-identical "
                            "to the first fifteen of these, so they are not concatenated again"},
        "roles": gr.ROLES,
        "seed": gr.SEED,
        "role_sizes_in_graph": [int(sum(1 for v in role_of.values() if v == r)) for r in range(gr.ROLES)],
        "role_space_mapping": {str(r): ROLE_SPACE[r] for r in ROLE_SPACE},
        "role_profiles": profiles,
    }
    OUT_META.write_text(json.dumps(meta, indent=1, default=float))
    print(f"role table: {meta['role_table_rows']} rows, {meta['items_without_a_role']} without a role")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
