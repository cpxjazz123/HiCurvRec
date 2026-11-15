"""GRAAL's orbit weights, derived from the orbit definition.

Milenković & Pržulj define ``w_i = 1 - log10(o_i)/log10(73)`` with ``o_i`` the number of orbits that
affect orbit i, and state the table is available on request -- neither paper prints it.  The relation is
derivable: an orbit i is defined by a graphlet G_i and a node class in it; the orbits that necessarily
appear whenever a node sits at orbit i are exactly the orbits that node occupies inside the induced
subgraphs of G_i that contain it.  ORCA is run on each of the 30 connected graphlets up to five nodes, and
each node's touched-orbit set is read off.  A node's own orbit is recovered by subtracting the sets of the
proper induced subgraphs, which leaves exactly the label of the graphlet itself.

This reproduces the paper's two worked statements: orbit 0 (the degree) has o_i = 1 and weight 1, and orbit
3 (a triangle corner) has a smaller o_i than the orbits that contain a triangle.

    python3 results/research/issue215/orbit_weights.py
"""
from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
ORCA = REPO / "results" / "research" / "issue213" / "orca" / "orca"
CACHE = OUT / "orbit_weights.json"
WORK = OUT / "orbit_weights_work"


def orca_sets(graph) -> dict[int, set[int]]:
    """Node -> set of orbits it touches, from ORCA on this single induced graphlet."""
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "g.in"
    out = WORK / "g.out"
    nodes = sorted(graph.nodes())
    with path.open("w") as handle:
        handle.write(f"{len(nodes)} {graph.number_of_edges()}\n")
        for a, b in graph.edges():
            handle.write(f"{a} {b}\n")
    subprocess.run([str(ORCA), "node", "5", str(path), str(out)], check=True, capture_output=True)
    table = np.loadtxt(out, dtype=np.float64).reshape(len(nodes), -1)
    return {node: set(int(c) for c in np.flatnonzero(table[position] > 0))
            for position, node in enumerate(nodes)}


def proper_subgraph_sets(graph, keep):
    union = set()
    for size in range(2, graph.number_of_nodes()):
        for subset in itertools.combinations(sorted(graph.nodes()), size):
            if keep not in subset:
                continue
            block = graph.subgraph(subset)
            if not nx.is_connected(block):
                continue
            relabelled = nx.convert_node_labels_to_integers(block.copy(), ordering="sorted")
            position = sorted(block.nodes()).index(keep)
            union |= orca_sets(relabelled)[position]
    return union


def derive():
    graphs = [g for g in nx.graph_atlas_g()
              if 2 <= g.number_of_nodes() <= 5 and nx.is_connected(g)]
    unique, seen = [], set()
    for graph in graphs:
        key = nx.weisfeiler_lehman_graph_hash(graph)
        if key in seen:
            continue
        seen.add(key)
        unique.append(graph)
    weights, labels_seen = {}, {}
    for graph in unique:
        touched = orca_sets(graph)
        for node in graph.nodes():
            own = touched[node] - proper_subgraph_sets(graph, node)
            if len(own) != 1:
                raise SystemExit(f"graphlet {graph.number_of_nodes()}n node {node}: "
                                 f"residual {sorted(own)}")
            orbit = own.pop()
            entry = {"o_i": len(touched[node]),
                     "w": 1.0 - np.log10(len(touched[node])) / np.log10(73),
                     "graphlet_nodes": graph.number_of_nodes(),
                     "orbits_touched": sorted(touched[node])}
            # several nodes of one graphlet legitimately share an orbit (the star's leaves); they must
            # produce the same dependency set, and a different graphlet must not claim the same label
            previous = labels_seen.get(orbit)
            if previous is not None:
                if previous["graphlet_nodes"] != entry["graphlet_nodes"] or \
                        previous["orbits_touched"] != entry["orbits_touched"]:
                    raise SystemExit(f"orbit {orbit} inconsistent between occurrences")
                continue
            labels_seen[orbit] = entry
            weights[orbit] = entry
    return {"graphlets": len(unique), "orbits": weights}


def main() -> int:
    data = derive()
    CACHE.write_text(json.dumps(data, indent=1, sort_keys=True))
    orbits = {int(key): value for key, value in data["orbits"].items()}
    print(f"graphlets {data['graphlets']}; orbits labelled {len(orbits)}")
    for orbit in (0, 3, 10, 14, 72):
        if orbit in orbits:
            entry = orbits[orbit]
            print(f"  orbit {orbit:>2}: o_i={entry['o_i']:>2} w={entry['w']:.4f} "
                  f"touched={entry['orbits_touched'][:8]}")
    print("missing:", [orbit for orbit in range(73) if orbit not in orbits])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
