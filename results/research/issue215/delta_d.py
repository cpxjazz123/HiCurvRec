"""Delta D = D(G_b, G_q) - D(G_b, G_z) for the arms that have both vector sets.

G_z and G_q graphs already exist for five arms (TIGER, F0, B, C, D) under graphs/, built by
structure_graphs.py with the same k.  Every distance is computed the same way for both sides, so the
difference isolates what quantization did, and the sign says whether the quantized graph moved towards
the behaviour graph or away from it.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
GRAPHS = REPO / "results" / "research" / "issue215" / "graphs"
ORCA = REPO / "results" / "research" / "issue213" / "orca" / "orca"
NAMES = ("TIGER", "F0", "B", "C", "D")

spec = importlib.util.spec_from_file_location(
    "graph_tools", REPO / "results" / "research" / "issue215" / "quantised_structure_all.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)


def read_edges(path):
    with path.open() as handle:
        n, m = map(int, handle.readline().split())
        rows = np.loadtxt(handle, dtype=np.int64, max_rows=m)
    return n, (rows if rows.ndim == 2 else rows.reshape(1, -1))


def orbits(path, size):
    out = path.with_suffix(f".{size}.out")
    subprocess.run([str(ORCA), "node", str(size), str(path), str(out)], check=True,
                   capture_output=True)
    return np.loadtxt(out, dtype=np.float64)


def main() -> int:
    n_b, edges_b = read_edges(GRAPHS / "behaviour.in")
    orbit_b5 = np.loadtxt(REPO / "results" / "research" / "issue213" / "orca" / "node5.out")
    orbit_b4 = np.loadtxt(REPO / "results" / "research" / "issue213" / "orca" / "node4.out")
    from concurrent.futures import ThreadPoolExecutor
    tasks = [(GRAPHS / f"{name}_{side}.in", size) for name in NAMES for side in ("z", "q")
             for size in (5, 4) if (GRAPHS / f"{name}_{side}.in").exists()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        tables = dict(zip(tasks, pool.map(lambda task: orbits(*task), tasks)))
    print(f"orbits computed for {len(tables)} graph files", flush=True)

    rows = []
    for name in NAMES:
        first, second = GRAPHS / f"{name}_z.in", GRAPHS / f"{name}_q.in"
        if not first.exists() or not second.exists():
            print(f"{name}: missing graphs"); continue
        _, edges_z = read_edges(first)
        _, edges_q = read_edges(second)
        orbit_z5, orbit_q5 = tables[(first, 5)], tables[(second, 5)]
        orbit_z4, orbit_q4 = tables[(first, 4)], tables[(second, 4)]
        row = {"name": name,
               "gcd73_before": tools.gcd(orbit_b5, orbit_z5),
               "gcd73_after": tools.gcd(orbit_b5, orbit_q5),
               "gcd15_before": tools.gcd(orbit_b4, orbit_z4),
               "gcd15_after": tools.gcd(orbit_b4, orbit_q4),
               "deltacon_before": tools.deltacon(edges_b, edges_z, n_b),
               "deltacon_after": tools.deltacon(edges_b, edges_q, n_b),
               "jaccard_before": tools.edge_overlap(edges_b, edges_z)["jaccard"],
               "jaccard_after": tools.edge_overlap(edges_b, edges_q)["jaccard"]}
        row["deltas"] = {key: row[f"{key}_after"] - row[f"{key}_before"]
                         for key in ("gcd73", "gcd15", "deltacon", "jaccard")}
        rows.append(row)
        print(f"{name}: GCD73 {row['gcd73_before']:.2f}->{row['gcd73_after']:.2f} "
              f"(d {row['deltas']['gcd73']:+.2f}) | GCD15 {row['gcd15_before']:.2f}->"
              f"{row['gcd15_after']:.2f} (d {row['deltas']['gcd15']:+.2f}) | DeltaCon "
              f"{row['deltacon_before']:.3f}->{row['deltacon_after']:.3f} "
              f"(d {row['deltas']['deltacon']:+.3f}) | Jaccard {row['jaccard_before']:.4f}->"
              f"{row['jaccard_after']:.4f} (d {row['deltas']['jaccard']:+.4f})", flush=True)
    path = REPO / "results" / "research" / "issue215" / "delta_d.json"
    path.write_text(json.dumps({"rows": rows}, indent=1, default=float))
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
