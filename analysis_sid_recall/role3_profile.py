"""What are Role 3's items, structurally? A descriptive profile, no naming.

The independence test said Role 3 is the only role carrying information the others do not.  Before
anything is designed around it, this script describes what that group actually is, from data only:

    graph invariants     degree, triangles, four- and five-cliques per item, local clustering,
                         k-core, two-hop neighbourhood size, mean neighbour degree, neighbour spread
    ORCA signatures      the mean of each of the 73 five-node orbit counts, standardised, with the
                         orbits that separate the role from the other five by the largest effect size
    pattern participation  how often the role's items appear in each of the nine five-node patterns
                         that this issue scores (sampled instances, so these are rates per item)

Effect sizes are Cohen's d against the pooled other five roles, so a large number means the role really
differs on that axis; a small one means the axis does not separate it.

    python3 analysis_sid_recall/role3_profile.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orca5_role_preservation as m  # noqa: E402
import role_pattern_gain as rpg  # noqa: E402

ORBITS = m.REPO / "results" / "research" / "issue213" / "orca" / "node5.out"
OUT = HERE / "role3_profile.json"
TARGET = 2                 # the NMF role that the report calls Role 3 (one-based)


def cohens_d(values: np.ndarray, mask: np.ndarray) -> float:
    inside = values[mask]
    outside = values[~mask]
    if inside.size < 2 or outside.size < 2:
        return float("nan")
    pooled = np.sqrt((inside.var(ddof=1) + outside.var(ddof=1)) / 2.0)
    return float((inside.mean() - outside.mean()) / pooled) if pooled else float("nan")


def main() -> int:
    n, adjacency, edges = m.load_graph()
    role_of = np.full(n, -1, dtype=np.int64)
    cached = np.load(m.ROLE_CACHE5)
    for node, role in zip(cached["nodes"], cached["role"]):
        if 0 <= int(node) < n:
            role_of[int(node)] = int(role)
    members = {role: int((role_of == role).sum()) for role in range(6)}
    print("role sizes:", {f"Role {role + 1}": size for role, size in members.items()})

    orbits = np.loadtxt(ORBITS, dtype=np.float64)
    if orbits.shape[0] != n:
        raise SystemExit(f"orbit table has {orbits.shape[0]} rows for {n} items")
    orbit_z = (orbits - orbits.mean(axis=0)) / (orbits.std(axis=0) + 1e-12)

    print("counting cliques and triangles ...", flush=True)
    cliques4 = m.enumerate_cliques(adjacency, 4)
    cliques5 = m.enumerate_cliques(adjacency, 5)
    k4_count = np.zeros(n, dtype=np.int64)
    for clique in cliques4:
        for node in clique:
            k4_count[node] += 1
    k5_count = np.zeros(n, dtype=np.int64)
    for clique in cliques5:
        for node in clique:
            k5_count[node] += 1

    neighbours = [sorted(adjacency[i]) for i in range(n)]
    degree = np.array([len(w) for w in neighbours], dtype=np.float64)
    triangles = np.zeros(n, dtype=np.float64)
    overlap = np.zeros(n, dtype=np.float64)
    two_hop = np.zeros(n, dtype=np.float64)
    neighbour_degree = np.zeros(n, dtype=np.float64)
    for node in range(n):
        near = neighbours[node]
        if not near:
            continue
        common_total = 0
        reach: set[int] = set()
        for other in near:
            common_total += len(adjacency[node] & adjacency[other])
            reach.update(neighbours[other])
        triangles[node] = common_total / 2.0
        overlap[node] = float(np.mean([len(adjacency[node] & adjacency[other]) for other in near]))
        reach.discard(node)
        two_hop[node] = float(len(reach))
        neighbour_degree[node] = float(np.mean([len(adjacency[other]) for other in near]))

    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from(edges.tolist())
    core = nx.core_number(graph)
    k_core = np.array([core.get(node, 0) for node in range(n)], dtype=np.float64)
    clustering = nx.clustering(graph)
    local_clustering = np.array([clustering.get(node, 0.0) for node in range(n)], dtype=np.float64)

    features = {"degree": degree, "triangles": triangles, "k4": k4_count.astype(np.float64),
                "k5": k5_count.astype(np.float64), "clustering": local_clustering,
                "k_core": k_core, "two_hop": two_hop, "neighbour_overlap": overlap,
                "neighbour_degree": neighbour_degree}

    print("\n=== graph profile per role (mean; Cohen's d of the role against the other five) ===")
    table = {}
    header = f"  {'feature':<20}" + "".join(f"{f'R{role + 1}':>9}" for role in range(6)) + "   d(Role3)"
    print(header)
    target_mask = role_of == TARGET
    for name, values in features.items():
        row = [float(values[role_of == role].mean()) if (role_of == role).any() else float("nan")
               for role in range(6)]
        effect = cohens_d(values, target_mask)
        table[name] = {"means": row, "cohens_d_role3": effect}
        print(f"  {name:<20}" + "".join(f"{value:>9.2f}" for value in row) + f"{effect:>10.2f}")

    print("\n=== ORCA five-node orbits that separate Role 3 most (standardised counts) ===")
    orbit_rows = []
    for index in range(orbit_z.shape[1]):
        effect = cohens_d(orbit_z[:, index], target_mask)
        orbit_rows.append((float(effect), index,
                           float(orbit_z[target_mask, index].mean())))
    orbit_rows.sort(key=lambda row: -abs(row[0]))
    print(f"  {'orbit':<10}{'mean z (Role 3)':>17}{'d vs others':>13}")
    top_orbits = []
    for effect, index, mean_z in orbit_rows[:10]:
        top_orbits.append({"orbit": index, "mean_z": mean_z, "cohens_d": effect})
        print(f"  o5_{index:<7}{mean_z:>17.2f}{effect:>+13.2f}")
    table["orbits"] = top_orbits

    print("\nbuilding the nine pattern instance sets ...", flush=True)
    prepared = rpg.prepare()
    incidences = prepared["incidences"]
    print("\n=== pattern participation: share of the role's items that appear in the sampled instances ===")
    print(f"  {'pattern':<16}" + "".join(f"{f'R{role + 1}':>9}" for role in range(6)))
    participation = {}
    for name, incidence in incidences.items():
        row = []
        for role in range(6):
            of_role = np.flatnonzero(role_of == role)
            if of_role.size == 0:
                row.append(float("nan"))
                continue
            touched = sum(1 for node in of_role if node in incidence)
            row.append(touched / of_role.size)
        participation[name] = row
        print(f"  {name:<16}" + "".join(f"{value:>9.3f}" for value in row))
    table["participation"] = participation

    result = {"roles": {f"Role {role + 1}": members[role] for role in range(6)},
              "target_role": "Role 3", "profile": table,
              "note": "Cohen's d of the role against the pooled other five; orbit indices follow "
                      "ORCA's 73-orbit numbering for graphlets up to five nodes"}
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {OUT.relative_to(m.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
