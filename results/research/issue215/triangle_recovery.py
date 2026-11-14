"""How many of a triangle's edges survive, in the graphs the vectors imply?

The owner's question, made concrete: take a triangle of the real behaviour graph -- three items
that are pairwise connected -- and ask how many of its three edges the model's own graph contains.
A model that recovers four of four edges around a neighbourhood keeps the local structure; one that
recovers two of four has lost half of it.

For every graph the vectors imply (one per arm and per vector space, plus the semantic reference)
this reports:

  * the mean number of a triangle's three edges that the graph contains (0 to 3);
  * the share of triangles that survive whole;
  * the same two numbers split by the structural roles the triangle's items carry, so it says which
    local structures are the hard ones to recover.

The behaviour graph is the answer key: on itself every number is 3 and 1.

    python3 results/research/issue215/triangle_recovery.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
GRAPHS = OUT / "graphs"
JOURNAL = OUT / "triangle_recovery.json"
ROLE_NAMES = {0: "dense hub", 1: "tight community", 2: "two-hop / fan", 3: "degree middle",
              4: "cycle / long path", 5: "star / chain"}
MAX_TRIANGLES = 300_000
SEED = 42


def load_edges(path: Path) -> tuple[int, np.ndarray]:
    """The edges as a sorted array of a*n+b keys, so membership is one vectorised lookup."""
    with path.open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    left = np.minimum(edges[:, 0], edges[:, 1])
    right = np.maximum(edges[:, 0], edges[:, 1])
    return n, np.sort(left * n + right)


def triangles_of(n: int, edge_keys: np.ndarray) -> np.ndarray:
    """Every triangle of the graph, found through the common neighbours of each edge."""
    adjacency: list[set[int]] = [set() for _ in range(n)]
    with open(GRAPHS / "behaviour.in") as handle:
        _, m = map(int, handle.readline().split())
        raw = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    for a, b in raw.tolist():
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    found = []
    for a, b in raw.tolist():
        a, b = int(a), int(b)
        for c in adjacency[a] & adjacency[b]:
            if c > b:
                found.append((a, b, c))
    return np.asarray(found, dtype=np.int64)


def load_roles() -> dict[int, int]:
    roles = {}
    for line in (OUT / "orca_roles_frozen.csv").read_text().splitlines()[1:]:
        item, role = line.split(",")
        roles[int(item)] = int(role)
    return roles


def evaluate(name: str, triangles: np.ndarray, edges: np.ndarray, roles: np.ndarray,
             n_items: int) -> dict:
    """Vectorised: the three edges of every triangle are looked up in the graph's edge array."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    found = np.zeros(len(triangles), dtype=np.int64)
    for left, right in ((a, b), (b, c), (a, c)):
        found += np.isin(left.astype(np.int64) * n_items + right.astype(np.int64),
                         edges.astype(np.int64))
    recovered = found.astype(np.float64)
    per_role = {}
    for role in sorted(ROLE_NAMES):
        picked = ((roles[a] == role) | (roles[b] == role) | (roles[c] == role))
        if not picked.any():
            continue
        values = recovered[picked]
        per_role[str(role)] = {
            "triangles": int(picked.sum()),
            "edges_recovered_mean": float(values.mean()),
            "triangle_survival": float((values == 3).mean()),
        }
    same_role = ((roles[a] == roles[b]) & (roles[b] == roles[c]))
    mixed_role = ~same_role
    block = {
        "triangles": int(len(triangles)),
        "edges_recovered_mean": float(recovered.mean()),
        "triangle_survival": float((recovered == 3).mean()),
        "all_three_fraction": float((recovered == 3).mean()),
        "none_fraction": float((recovered == 0).mean()),
        "per_role": per_role,
        "same_role_triangles": {
            "count": int(same_role.sum()),
            "edges_recovered_mean": float(recovered[same_role].mean()) if same_role.any() else None,
            "triangle_survival": float((recovered[same_role] == 3).mean()) if same_role.any() else None,
        },
        "mixed_role_triangles": {
            "count": int(mixed_role.sum()),
            "edges_recovered_mean": float(recovered[mixed_role].mean()) if mixed_role.any() else None,
            "triangle_survival": float((recovered[mixed_role] == 3).mean()) if mixed_role.any() else None,
        },
    }
    print(f"{name:<16} edges/triangle {block['edges_recovered_mean']:.3f} | triangles kept whole "
          f"{block['triangle_survival']:.4f} | none {block['none_fraction']:.3f}", flush=True)
    return block


def main() -> int:
    role_map = load_roles()
    n, behaviour_edges = load_edges(GRAPHS / "behaviour.in")
    triangles = triangles_of(n, behaviour_edges)
    roles = np.asarray([role_map.get(item, -1) for item in range(n)], dtype=np.int64)
    print(f"behaviour graph: {len(behaviour_edges)} edges, {len(triangles)} triangles", flush=True)
    if len(triangles) > MAX_TRIANGLES:
        rng = np.random.default_rng(SEED)
        picked = np.sort(rng.choice(len(triangles), size=MAX_TRIANGLES, replace=False))
        triangles = triangles[picked]
        print(f"sampled {len(triangles)} triangles for the evaluation", flush=True)

    report = {"items": n, "behaviour_edges": len(behaviour_edges), "triangles_evaluated": len(triangles),
              "graphs": {}}
    report["graphs"]["behaviour (answer key)"] = evaluate("behaviour (key)", triangles, behaviour_edges, roles, n)
    for path in sorted(GRAPHS.glob("*_rel.in")):
        name = path.name[:-len("_rel.in")]
        _, edges = load_edges(path)
        report["graphs"][name] = evaluate(name, triangles, edges, roles, n)
    for path in sorted(GRAPHS.glob("*_z.in")) + sorted(GRAPHS.glob("*_q.in")):
        name = path.name[:-len(".in")]
        if name in report["graphs"] or name == "behaviour":
            continue
        _, edges = load_edges(path)
        report["graphs"][name + " (knn)"] = evaluate(name + " (knn)", triangles, edges, roles)

    JOURNAL.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwritten: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
