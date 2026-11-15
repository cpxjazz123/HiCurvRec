"""Rebuild every cached orbit table from the current edge-budget graph.

The caches predate the exact edge budget (number 01 held 344,802 edges against the current 315,391), so a
reader had no way to know whether a table described the graph it was paired with.  Each arm is rebuilt from
its own quantised vectors, the graph is written with its edge fingerprint, and the table is stored beside it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
TOPO = OUT / "topology_graphs"

spec = importlib.util.spec_from_file_location("g", OUT / "graphlet_preservation.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
study = g.study


def rebuild(number):
    arm = next(entry for entry in study.load_arms(67) if entry["number"] == number)
    n = 24587
    tokens = study.tokens_of(arm["sid_path"], n)
    vectors, geometry = study.quantised_vectors(number, tokens, n)
    if vectors is None:
        return number, "no codebook"
    edges, stats = g.budget_edges(vectors[:n], len(g_behaviour), n)
    print(f"  {number}: graph built ({stats['edges']} edges, max degree {stats['max_degree']})", flush=True)
    if stats["max_degree"] > 600:
        # ORCA's triple enumeration grows with the cube of the maximum degree. At 677-1553 (arms 37, 53,
        # 54, 55) a single table did not finish in twenty minutes; 600 is the cut, and those arms are
        # reported as unavailable rather than silently dropped
        print(f"  {number}: max degree {stats['max_degree']} > 200, orbit table skipped", flush=True)
        return number, {**stats, "orbit_skipped": True}
    g.orbits_for(number, edges, n, TOPO)
    print(f"  {number}: orbit table written", flush=True)
    return number, stats


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor
    global g_behaviour
    import os
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    n, adjacency, behaviour = study.load_graph()
    g_behaviour = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in behaviour.tolist()}
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    numbers = [entry["number"] for entry in study.load_arms(67)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number, result in pool.map(rebuild, numbers):
            print(f"  {number}: {result}", flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
