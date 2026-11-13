"""How much behaviour-graph structure do the pre- and post-quantization graphs keep?

For every arm this compares two k-NN graphs against the behaviour graph, on the same items and with
the same ORCA features:

    G_z   over the encoder's vector before quantization
    G_q   over the model's own quantized readout

The questions, in the owner's order:

  1. how close is G_z to G_b -- how much structure the semantic representation itself carries;
  2. does G_q move *towards* G_b (quantization preserves or even sharpens structure) or away
     from it (the codebook costs structure)?

Agreement is measured two ways, both on the ORCA features of each graph:

  * the Spearman correlation, across items, of every structural statistic (degree, clustering, and
    each of ORCA's orbit counts) between G_b and G_x, then averaged;
  * the edge-level overlap with the behaviour graph, which says how many real behaviour edges the
    k-NN graph actually contains.

    python3 results/research/issue215/compare_structures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
GRAPHS = OUT / "graphs"
ROLE_TABLE = OUT / "orca_roles_frozen.csv"
ROLE_SHA = "c7a29e6559aed1385c5ce514e506be22d88bf6e85eb61120b6dbf69375ea251c"
JOURNAL = OUT / "compare_structures.json"
ARMS = ("TIGER", "F0", "B", "C", "D")


def load_graph(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """The edge list and the degree of every node; no graph object, no Python-level traversal."""
    with path.open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    degrees = np.bincount(edges.ravel(), minlength=n).astype(np.float64)
    return n, edges, degrees


def load_orbits(path: Path) -> np.ndarray:
    """ORCA's per-node orbit counts; the width follows whichever graphlet size produced them."""
    counts = np.loadtxt(path, dtype=np.int64)
    return np.atleast_2d(counts)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.allclose(x, y) or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def main() -> int:
    n, behaviour_edges, degrees_b = load_graph(GRAPHS / "behaviour.in")
    behaviour_set = {tuple(sorted(pair)) for pair in behaviour_edges.tolist()}
    orbits_b = load_orbits(GRAPHS / "behaviour.out")
    # ORCA's orbit 3 is the number of triangles at the node, so the local clustering coefficient
    # is 2*T/(d(d-1)) without touching a graph library
    clustering_b = np.where(degrees_b > 1,
                            2.0 * orbits_b[:, 3] / np.maximum(degrees_b * (degrees_b - 1.0), 1e-12), 0.0)

    roles = {}
    with ROLE_TABLE.open() as handle:
        next(handle)
        for line in handle:
            item, role = line.strip().split(",")
            roles[int(item)] = int(role)

    report = {"items": n, "behaviour_edges": len(behaviour_set), "arms": {}}
    for arm in ARMS:
        block = {}
        for kind in ("z", "q"):
            path = GRAPHS / f"{arm}_{kind}.in"
            if not path.is_file() or not (GRAPHS / f"{arm}_{kind}.out").is_file():
                block[kind] = {"status": "missing"}
                continue
            _, edges, degrees_x = load_graph(path)
            edge_set = {tuple(sorted(pair)) for pair in edges.tolist()}
            overlap = len(edge_set & behaviour_set)
            orbits_x = load_orbits(GRAPHS / f"{arm}_{kind}.out")
            clustering_x = np.where(degrees_x > 1,
                                    2.0 * orbits_x[:, 3] / np.maximum(degrees_x * (degrees_x - 1.0), 1e-12), 0.0)
            corr_orbits = np.asarray([spearman(orbits_b[:, i], orbits_x[:, i]) for i in range(orbits_b.shape[1])])
            per_role = {}
            for role in sorted(set(roles.values())):
                items = np.asarray([item for item, value in roles.items() if value == role and item < n])
                per_role[str(role)] = {
                    "items": int(items.size),
                    "degree_b": float(degrees_b[items].mean()),
                    "degree_x": float(degrees_x[items].mean()),
                    "clustering_b": float(clustering_b[items].mean()),
                    "clustering_x": float(clustering_x[items].mean()),
                    "orbit_profile_cosine": float(
                        (orbits_b[items].mean(0) @ orbits_x[items].mean(0))
                        / max(np.linalg.norm(orbits_b[items].mean(0)) * np.linalg.norm(orbits_x[items].mean(0)), 1e-12)),
                }
            block[kind] = {
                "status": "ok",
                "edges": len(edge_set),
                "edges_recovered": overlap,
                "edge_recall_vs_behaviour": overlap / max(len(behaviour_set), 1),
                "edge_precision_vs_behaviour": overlap / max(len(edge_set), 1),
                "degree_spearman": spearman(degrees_b, degrees_x),
                "clustering_spearman": spearman(clustering_b, clustering_x),
                "orbit_spearman_mean": float(np.nanmean(corr_orbits)),
                "orbit_spearman_min": float(np.nanmin(corr_orbits)),
                "orbit_spearman_max": float(np.nanmax(corr_orbits)),
                "per_role": per_role,
            }
        report["arms"][arm] = block
        for kind in ("z", "q"):
            value = block.get(kind, {})
            if value.get("status") != "ok":
                continue
            print(f"{arm} {kind}: edges {value['edges']} | recovers {value['edges_recovered']} behaviour edges "
                  f"({value['edge_recall_vs_behaviour']:.4f}) | degree rho {value['degree_spearman']:+.3f} "
                  f"| clustering rho {value['clustering_spearman']:+.3f} | orbit rho {value['orbit_spearman_mean']:+.3f}",
                  flush=True)
    JOURNAL.write_text(json.dumps(report, indent=1, default=float))

    print("\nquantization effect (G_q minus G_z), averaged structure agreement:")
    print(f"{'arm':<5} {'orbit rho z':>12} {'orbit rho q':>12} {'delta':>8} {'degree dz->dq':>15} {'edge recall z->q':>18}")
    for arm in ARMS:
        block = report["arms"][arm]
        if block.get("z", {}).get("status") != "ok" or block.get("q", {}).get("status") != "ok":
            continue
        z, q = block["z"], block["q"]
        print(f"{arm:<5} {z['orbit_spearman_mean']:>12.4f} {q['orbit_spearman_mean']:>12.4f} "
              f"{q['orbit_spearman_mean'] - z['orbit_spearman_mean']:>+8.4f} "
              f"{z['degree_spearman']:+.3f}->{q['degree_spearman']:+.3f}   "
              f"{z['edge_recall_vs_behaviour']:.4f}->{q['edge_recall_vs_behaviour']:.4f}")
    print(f"\nwritten: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
