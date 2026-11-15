"""Three readings of preservation, computed inside the six fixed Roles.

The Roles come from the behaviour graph's NMF (K=6) and are used as given: re-clustering the quantised
graph and matching "Role 3" to "Role 3" by number would compare different item sets.

    distance   the existing structural score: within a Role's sampled structures, are the near/far
               orderings right?  (this is the per-role gain already reported, recomputed here on the
               same graph so the three readings are comparable arm by arm)
    gdv        GRAAL-style Graphlet Degree Vector similarity: for each item, compare its 73 orbit
               counts in the behaviour graph with the same item's counts in the quantised graph, per
               orbit as min/max and averaged over orbits (unweighted and with 1/log1p(count) weights)
    graphlet   for each of the nine structures, does the Role's items keep the behaviour edges that
               formed that structure?  (share of induced edges surviving, per Role, per structure)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
TOPO = OUT / "topology_graphs"
ORBIT_B = REPO / "results" / "research" / "issue213" / "orca" / "node5.out"

spec = importlib.util.spec_from_file_location(
    "graphlet_tools", OUT / "graphlet_preservation.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
study = tools.study

PATTERNS = ("K5", "double_triangle", "K4_tail", "K2_3", "P5_centre", "C5", "P5_hops", "K1_4", "T5")


def gdv_similarity(first, second, weights):
    """GRAAL's per-orbit relative similarity, averaged over orbits with the given weights."""
    both_zero = (first == 0) & (second == 0)
    maximum = np.maximum(first, second).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(maximum > 0, 1.0 - np.abs(first - second) / np.maximum(maximum, 1e-12), 1.0)
    weighted = (relative * weights).sum(axis=1) / weights.sum()
    plain = relative.mean(axis=1)
    weighted[both_zero.all(axis=1)] = 1.0
    plain[both_zero.all(axis=1)] = 1.0
    return plain, weighted


def main() -> int:
    n, adjacency, behaviour_edges = study.load_graph()
    role_of = study.load_roles(n, adjacency, "mean")
    orbit_b = np.loadtxt(ORBIT_B, dtype=np.float64)
    weights = 1.0 / np.log1p(orbit_b.sum(axis=0).clip(min=1.0))
    behaviour_set = {(int(a), int(b)) if a < b else (int(b), int(a))
                     for a, b in behaviour_edges.tolist()}
    instances = np.load(OUT / "graphlet_instances.npz", allow_pickle=True)
    sets = {name: [tuple(int(node) for node in row) for row in instances[name]] for name in PATTERNS}

    rows = []
    for arm in study.load_arms(67):
        number = arm["number"]
        stored = TOPO / f"{number}.npz"
        if not stored.exists():
            rows.append({"number": number, "reason": "no orbit table"})
            continue
        orbit_q = np.load(stored)["orbit5"]
        edges = np.load(stored)["edges"]
        edge_set = {(int(a), int(b)) for a, b in edges.tolist()}
        plain, weighted = gdv_similarity(orbit_b, orbit_q, weights)
        row = {"number": number, "recall": arm["recall"],
               "edge_recall": len(edge_set & behaviour_set) / max(len(behaviour_set), 1)}
        for role in range(6):
            mask = role_of == role
            key = f"role{role + 1}"
            row[f"{key}_gdv"] = float(np.nanmean(plain[mask]))
            row[f"{key}_gdv_weighted"] = float(np.nanmean(weighted[mask]))
            row[f"{key}_graphlet"] = {}
            for name in PATTERNS:
                shares = []
                for nodes in sets[name]:
                    if role_of[nodes[0]] != role:
                        continue
                    wanted = set()
                    for i in range(len(nodes)):
                        for j in range(i + 1, len(nodes)):
                            a, b = nodes[i], nodes[j]
                            wanted.add((a, b) if a < b else (b, a))
                    wanted &= behaviour_set
                    if wanted:
                        shares.append(len(wanted & edge_set) / len(wanted))
                row[f"{key}_graphlet"][name] = float(np.mean(shares)) if shares else None
        rows.append(row)
        print(f"  {number} recall {row['recall']:.5f} | " +
              " ".join(f"r{role + 1}:{row[f'role{role + 1}_gdv']:.3f}"
                       for role in range(6)), flush=True)
    (OUT / "role_topology.json").write_text(json.dumps({"rows": rows}, indent=1, default=float))
    print(f"written {OUT / 'role_topology.json'} ({len(rows)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
