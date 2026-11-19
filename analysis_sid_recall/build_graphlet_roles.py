"""Issue #217 data: induced 3/4-node graphlet instances with per-node orbit roles.

Every instance is an induced subgraph (all required edges present, no other edge between
its nodes).  Roles are topological orbits inside the instance, so a leaf of a 4-cycle and a
leaf of a claw are different roles while the four C4 nodes share one role.

Writes results/research/graphlet_roles/{instances.npz,coverage.json}; the split is a stable
hash of the instance's unordered node set, so no instance (in any permutation) is on both
sides.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
METADATA = REPO / "results" / "research" / "behavior_graph" / "metadata.json"
OUT = REPO / "results" / "research" / "graphlet_roles"
HELDOUT_FRACTION = 0.2
PER_CENTRE_CAP = {"K3": 20, "P3": 20, "C4": 12, "P4": 12, "K1_3": 12}
TYPES = ("K3", "P3", "C4", "P4", "K1_3")
ROLES = {
    "K3": lambda n: 0,
    "P3": lambda n: {0: 0, 1: 1, 2: 0},
    "C4": lambda n: 0,
    "P4": lambda n: 0,
    "K1_3": lambda n: 0,
}


def verify_graph() -> str:
    metadata = json.loads(METADATA.read_text())
    record = (metadata.get("outputs") or {}).get("graph.npz") or {}
    digest = hashlib.sha256(GRAPH.read_bytes()).hexdigest()
    if record.get("sha256") != digest:
        raise RuntimeError("graph.npz does not match metadata.json")
    if (metadata.get("policy") or {}).get("single_undirected_graph") is not True:
        raise RuntimeError("the graph policy block is missing")
    return digest


def load():
    payload = np.load(GRAPH, allow_pickle=False)
    n = int(payload["nodes"])
    neighbours: list[set] = [set() for _ in range(n)]
    for a, b in zip(payload["rows"].tolist(), payload["cols"].tolist()):
        neighbours[a].add(b)
        neighbours[b].add(a)
    degree = np.array([len(s) for s in neighbours], dtype=np.int64)
    return neighbours, degree, n


def split_of(kind: str, nodes: tuple) -> str:
    key = kind + "|" + "-".join(str(v) for v in sorted(nodes))
    return "heldout" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100 < HELDOUT_FRACTION * 100 else "train"


def role_vector(kind: str, nodes: tuple, neighbours: list[set]) -> tuple:
    """Orbit label per position, computed from the induced degrees inside the instance."""
    inner = {}
    for node in nodes:
        inner[node] = sum(1 for other in neighbours[node] if other in nodes)
    if kind == "K3":
        return (0, 0, 0)
    if kind == "P3":
        centre = max(nodes, key=lambda v: inner[v])
        return tuple(1 if v == centre else 0 for v in nodes)
    if kind == "C4":
        return (0, 0, 0, 0)
    if kind == "P4":
        return tuple(1 if inner[v] == 2 else 0 for v in nodes)
    if kind == "K1_3":
        centre = max(nodes, key=lambda v: inner[v])
        return tuple(1 if v == centre else 0 for v in nodes)
    raise KeyError(kind)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    graph_sha = verify_graph()
    neighbours, degree, n = load()
    per_centre = {kind: {} for kind in TYPES}
    seen = {kind: set() for kind in TYPES}
    rows = []

    def add(kind: str, nodes: tuple, centre: int) -> None:
        key = tuple(sorted(nodes))
        if key in seen[kind]:
            return
        seen[kind].add(key)
        rows.append((kind, key, role_vector(kind, key, neighbours)))

    from itertools import combinations

    for a in range(n):
        members = sorted(neighbours[a])[:60]  # hubs are capped, not enumerated whole
        for index, b in enumerate(members):
            for c in members[index + 1:]:
                if c in neighbours[b]:
                    if per_centre["K3"].get(a, 0) < PER_CENTRE_CAP["K3"]:
                        add("K3", (a, b, c), a)
                        per_centre["K3"][a] = per_centre["K3"].get(a, 0) + 1
                    continue
                if per_centre["P3"].get(a, 0) < PER_CENTRE_CAP["P3"]:
                    add("P3", (a, b, c), a)
                    per_centre["P3"][a] = per_centre["P3"].get(a, 0) + 1
                # C4 = a-b-d-c-a: b and c share d, and d is not adjacent to a.  Without the
                # a-d check the induced subgraph is a diamond (five edges), not a 4-cycle.
                for d in sorted((neighbours[b] & neighbours[c]) - {a}):
                    if d in neighbours[a]:
                        continue
                    if per_centre["C4"].get(a, 0) >= PER_CENTRE_CAP["C4"]:
                        break
                    add("C4", (a, b, c, d), a)
                    per_centre["C4"][a] = per_centre["C4"].get(a, 0) + 1
        # K1,3 = centre a with three pairwise non-adjacent leaves
        if per_centre["K1_3"].get(a, 0) < PER_CENTRE_CAP["K1_3"]:
            for leaves in combinations(members[:24], 3):
                if per_centre["K1_3"].get(a, 0) >= PER_CENTRE_CAP["K1_3"]:
                    break
                if any(second in neighbours[first] for first, second in combinations(leaves, 2)):
                    continue
                add("K1_3", (a, *leaves), a)
                per_centre["K1_3"][a] = per_centre["K1_3"].get(a, 0) + 1
        # P4 = v1-v2-v3-v4 induced path, enumerated over the middle edge (v2, v3)
        if per_centre["P4"].get(a, 0) < PER_CENTRE_CAP["P4"]:
            for v2 in members:
                if per_centre["P4"].get(a, 0) >= PER_CENTRE_CAP["P4"]:
                    break
                for v3 in sorted(neighbours[v2]):
                    if v3 == a or v3 in neighbours[a]:
                        continue
                    for v1 in sorted(neighbours[v2] - {v3}):
                        if v1 == a or per_centre["P4"].get(a, 0) >= PER_CENTRE_CAP["P4"]:
                            continue
                        for v4 in sorted(neighbours[v3] - {v2}):
                            if v4 in (a, v1) or v4 in neighbours[v2] or v4 in neighbours[v1]:
                                continue
                            if v1 in neighbours[v3]:
                                continue
                            add("P4", (v1, v2, v3, v4), a)
                            per_centre["P4"][a] = per_centre["P4"].get(a, 0) + 1
                            break
                        if per_centre["P4"].get(a, 0) >= PER_CENTRE_CAP["P4"]:
                            break

    payload = {"K3": [], "P3": [], "C4": [], "P4": [], "K1_3": []}
    splits = {"K3": [], "P3": [], "C4": [], "P4": [], "K1_3": []}
    for kind, nodes, roles in rows:
        payload[kind].append((nodes, roles))
        splits[kind].append(split_of(kind, nodes))
    arrays = {}
    coverage = {"graph_sha256": graph_sha, "types": {}}
    for kind in TYPES:
        if not payload[kind]:
            coverage["types"][kind] = {"total": 0}
            continue
        nodes = np.array([entry[0] for entry in payload[kind]], dtype=np.int64)
        roles = np.array([entry[1] for entry in payload[kind]], dtype=np.int8)
        split = np.array([1 if value == "heldout" else 0 for value in splits[kind]], dtype=np.int8)
        arrays[f"{kind}_nodes"] = nodes
        arrays[f"{kind}_roles"] = roles
        arrays[f"{kind}_split"] = split
        coverage["types"][kind] = {
            "total": int(len(nodes)),
            "train": int((split == 0).sum()),
            "heldout": int((split == 1).sum()),
            "role_counts": {str(role): int((roles == role).sum()) for role in np.unique(roles)},
            "distinct_node_sets": int(len({tuple(row) for row in nodes.tolist()})),
            "sha256": hashlib.sha256(nodes.tobytes() + roles.tobytes()).hexdigest(),
        }
        print(kind, coverage["types"][kind]["total"], "heldout", coverage["types"][kind]["heldout"])
    np.savez_compressed(OUT / "instances.npz", **arrays)
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
