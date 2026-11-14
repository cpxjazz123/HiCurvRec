"""How much of the behaviour graph's structure survives in the graphs the vectors imply.

For every predicted-relation graph (one per arm and per vector space) this compares its ORCA
features with the behaviour graph's, item by item:

  * the mean Spearman over ORCA's 73 orbit columns -- does the graph the vectors imply reproduce the
    behaviour graph's local structures, item by item?
  * the degree and clustering correlation;
  * per role: the mean degree the behaviour role's items get in the predicted graph, against the
    behaviour graph's own mean degree for that role -- do the hubs stay hubs?

    python3 results/research/issue215/relation_graph_structure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
GRAPHS = OUT / "graphs"
JOURNAL = OUT / "relation_graph_structure.json"
ROLE_NAMES = {0: "dense hub", 1: "tight community", 2: "two-hop / fan", 3: "degree middle",
              4: "cycle / long path", 5: "star / chain"}


def load_graph(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    with path.open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    degrees = np.bincount(edges.ravel(), minlength=n).astype(np.float64)
    return n, edges, degrees


def load_roles() -> dict[int, int]:
    roles = {}
    for line in (OUT / "orca_roles_frozen.csv").read_text().splitlines()[1:]:
        item, role = line.split(",")
        roles[int(item)] = int(role)
    return roles


def clustering_of(degrees: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    return np.where(degrees > 1, 2.0 * triangles / np.maximum(degrees * (degrees - 1.0), 1e-12), 0.0)


def main() -> int:
    roles = load_roles()
    n, behaviour_edges, behaviour_degrees = load_graph(GRAPHS / "behaviour.in")
    behaviour_orbits = np.loadtxt(GRAPHS / "behaviour5.out", dtype=np.float64)
    behaviour_clustering = clustering_of(behaviour_degrees, behaviour_orbits[:, 3])
    role_vector = np.asarray([roles.get(item, -1) for item in range(n)], dtype=np.int64)

    report = {"items": n, "behaviour_edges": int(len(behaviour_edges)), "graphs": {}}
    print(f"{'graph':<16} {'orbit rho':>10} {'degree rho':>11} {'clust rho':>10} {'per-role degree (behaviour -> predicted)':<44}")
    for path in sorted(GRAPHS.glob("*_rel.in")):
        name = path.name[:-len("_rel.in")]
        orbits_path = GRAPHS / f"{name}_rel.out"
        if not orbits_path.is_file():
            continue
        _, edges, degrees = load_graph(path)
        orbits = np.loadtxt(orbits_path, dtype=np.float64)
        clustering = clustering_of(degrees, orbits[:, 3])
        rho = [float(spearmanr(behaviour_orbits[:, column], orbits[:, column]).statistic)
               for column in range(behaviour_orbits.shape[1])]
        per_role = {}
        for role in sorted(ROLE_NAMES):
            items = np.flatnonzero(role_vector == role)
            if items.size == 0:
                continue
            per_role[str(role)] = {
                "items": int(items.size),
                "behaviour_degree": float(behaviour_degrees[items].mean()),
                "predicted_degree": float(degrees[items].mean()),
                "clustering_behaviour": float(behaviour_clustering[items].mean()),
                "clustering_predicted": float(clustering[items].mean()),
            }
        block = {
            "edges": int(len(edges)),
            "orbit_spearman_mean": float(np.nanmean(rho)),
            "orbit_spearman_min": float(np.nanmin(rho)),
            "orbit_spearman_max": float(np.nanmax(rho)),
            "degree_spearman": float(spearmanr(behaviour_degrees, degrees).statistic),
            "clustering_spearman": float(spearmanr(behaviour_clustering, clustering).statistic),
            "per_role": per_role,
        }
        report["graphs"][name] = block
        role_text = " ".join(f"{ROLE_NAMES[r][:4]}:{per_role[str(r)]['behaviour_degree']:.0f}->{per_role[str(r)]['predicted_degree']:.0f}"
                             for r in sorted(ROLE_NAMES) if str(r) in per_role)
        print(f"{name:<16} {block['orbit_spearman_mean']:>+10.4f} {block['degree_spearman']:>+11.3f} "
              f"{block['clustering_spearman']:>+10.3f} {role_text}", flush=True)

    JOURNAL.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwritten: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
