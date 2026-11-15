"""Same-item ORCA orbit similarity, and the same question split by graphlet type.

ORCA gives every item a 73-dimensional orbit count in each graph.  Comparing the *same item's* vector
between the behaviour graph and the quantised graph asks whether that item kept its local topological
role; grouping the 73 orbits by the graphlet they belong to asks which structures' signatures survive.

The orbit-to-graphlet grouping is not taken from a table: each template is handed to ORCA on its own and
the orbits that light up are that graphlet's orbits, so the grouping is derived from the same binary that
produces the counts.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
TOPO = OUT / "topology_graphs"
ORCA = REPO / "results" / "research" / "issue213" / "orca" / "orca"
ORBIT_B = REPO / "results" / "research" / "issue213" / "orca" / "node5.out"

spec = importlib.util.spec_from_file_location(
    "graphlet_tools", OUT / "graphlet_preservation.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
study = tools.study

TEMPLATES = {
    "K5": [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    "K4_tail": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)],
    "K2_3": [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)],
    "C5": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
    "T5": [(0, 1), (1, 2), (2, 0), (0, 3)],
    "P5": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "K1_4": [(0, 1), (0, 2), (0, 3), (0, 4)],
    "K4": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    "triangle": [(0, 1), (1, 2), (2, 0)],
}


def template_orbits(force: bool = False):
    """Orbits each template occupies, read back from ORCA itself."""
    cache = OUT / "orbit_groups.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    groups = {}
    for name, edges in TEMPLATES.items():
        n = 1 + max(max(a, b) for a, b in edges)
        in_path = OUT / "orbit_groups" / f"{name}.in"
        out_path = OUT / "orbit_groups" / f"{name}.out"
        in_path.parent.mkdir(parents=True, exist_ok=True)
        tools.tools.write_graph(in_path, n, np.asarray(edges, dtype=np.int64))
        subprocess.run([str(ORCA), "node", "5", str(in_path), str(out_path)], check=True,
                       capture_output=True)
        table = np.loadtxt(out_path, dtype=np.float64).reshape(n, -1)
        groups[name] = [int(column) for column in np.flatnonzero(table.sum(axis=0) > 0)]
        print(f"  {name}: orbits {groups[name]}", flush=True)
    cache.write_text(json.dumps(groups, indent=1))
    return groups


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    change = np.flatnonzero(np.diff(sorted_values) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(values)]]) - 1
    ranks = np.empty(len(values))
    ranks[order] = np.repeat((starts + ends) / 2.0, ends - starts + 1)
    return ranks


def spearman(first, second):
    a = average_ranks(first) - (len(first) - 1) / 2
    b = average_ranks(second) - (len(second) - 1) / 2
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def cosine_rows(first, second):
    numerator = (first * second).sum(axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def arm_metrics(orbit_b, orbit_q, groups, degree_b, degree_q):
    active = (degree_b > 0) & (degree_q > 0)
    log_b, log_q = np.log1p(orbit_b), np.log1p(orbit_q)
    node_cos = cosine_rows(log_b, log_q)[active]
    per_orbit_rho = np.array([spearman(orbit_b[active, column], orbit_q[active, column])
                              for column in range(orbit_b.shape[1])])
    row = {"node_orbit_cos": float(np.nanmean(node_cos)),
           "node_orbit_rho": float(np.nanmedian(per_orbit_rho)),
           "node_orbit_rho_mean": float(np.nanmean(per_orbit_rho))}
    for name, columns in groups.items():
        if not columns:
            continue
        group_cos = cosine_rows(log_b[:, columns], log_q[:, columns])[active]
        row[f"sim_{name}"] = float(np.nanmean(group_cos))
        # a cosine over log-counts is dominated by the many shared near-zero orbits; the per-orbit rank
        # correlation within the group is the discriminating reading of the same question
        row[f"rho_{name}"] = float(np.nanmedian(per_orbit_rho[columns]))
        argmax_b = np.argmax(orbit_b[:, columns], axis=1)
        argmax_q = np.argmax(orbit_q[:, columns], axis=1)
        row[f"role_{name}"] = float(np.mean(argmax_b[active] == argmax_q[active]))
    return row


def main() -> int:
    groups = template_orbits()
    orbit_b = np.loadtxt(ORBIT_B, dtype=np.float64)
    n, adjacency, behaviour_edges = study.load_graph()
    degree_b = np.array([len(adjacency[i]) for i in range(n)], dtype=np.float64)
    behaviour_set = {(int(a), int(b)) if a < b else (int(b), int(a))
                     for a, b in behaviour_edges.tolist()}
    rows = []
    for arm in study.load_arms(67):
        number = arm["number"]
        stored = np.load(TOPO / f"{number}.npz") if (TOPO / f"{number}.npz").exists() else None
        if stored is None:
            print(f"  {number}: no orbit table yet", flush=True)
            continue
        orbit_q = stored["orbit5"]
        edges = stored["edges"]
        edge_set = {(int(a), int(b)) for a, b in edges.tolist()}
        degree_q = np.bincount(edges.ravel(), minlength=n).astype(np.float64)
        row = {"number": number, "recall": arm["recall"],
               "edge_recall": len(edge_set & behaviour_set) / max(len(behaviour_set), 1),
               "mean_degree": float(degree_q.mean())}
        row.update(arm_metrics(orbit_b, orbit_q, groups, degree_b, degree_q))
        rows.append(row)
        print(f"  {number} recall {row['recall']:.5f} | node cos {row['node_orbit_cos']:.3f} "
              f"rho {row['node_orbit_rho']:+.3f} | " +
              " ".join(f"{name}:{row.get(f'sim_{name}', float('nan')):.2f}"
                       for name in TEMPLATES if f"sim_{name}" in row), flush=True)
    (OUT / "orbit_similarity.json").write_text(json.dumps({"rows": rows}, indent=1, default=float))
    print(f"written {OUT / 'orbit_similarity.json'} ({len(rows)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
