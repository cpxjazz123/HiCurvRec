"""ORCA orbit structure of the TIGER pre/post cluster graphs, relative to the behaviour graph.

The ORCA binary is rebuilt from the upstream source under results/orca_build/orca (not committed).
Inputs are the same budgeted graphs as structure_retention.py.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/orca_structure.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPHS = HERE / "cluster_graphs.npz"
ORCA = REPO / "results" / "orca_build" / "orca" / "orca"
RUN = REPO / "results" / "orca_build" / "runs"
OUT = HERE / "orca_structure.json"
MD = HERE / "orca_structure.md"


def write_graph(path: Path, nodes: int, edges: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write(f"{nodes} {len(edges)}\n")
        for left, right in edges.tolist():
            handle.write(f"{int(left)} {int(right)}\n")


def run_orca(name: str, nodes: int, edges: np.ndarray, size: int) -> np.ndarray:
    RUN.mkdir(parents=True, exist_ok=True)
    in_path = RUN / f"{name}_{size}.in"
    out_path = RUN / f"{name}_{size}.out"
    write_graph(in_path, nodes, edges)
    subprocess.run([str(ORCA), "node", str(size), str(in_path), str(out_path)], check=True)
    table = np.loadtxt(out_path, dtype=np.float64)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    return table


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    rows = min(len(reference), len(candidate))
    columns = min(reference.shape[1], candidate.shape[1])
    rhos = []
    for column in range(columns):
        if np.allclose(reference[:, column], reference[0, column]) or np.allclose(candidate[:, column], candidate[0, column]):
            continue
        rhos.append(float(spearmanr(reference[:rows, column], candidate[:rows, column]).statistic))
    left = np.log1p(np.clip(reference[:rows, :columns], 0, None))
    right = np.log1p(np.clip(candidate[:rows, :columns], 0, None))
    cosine = np.einsum("ij,ij->i", left, right) / np.clip(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12, None
    )
    return {
        "columns": columns,
        "median_spearman": float(np.nanmedian(rhos)),
        "mean_spearman": float(np.nanmean(rhos)),
        "p10_spearman": float(np.nanpercentile(rhos, 10)),
        "p90_spearman": float(np.nanpercentile(rhos, 90)),
        "profile_cosine_mean": float(cosine.mean()),
    }


def main() -> int:
    payload = np.load(GRAPHS)
    nodes = int(max(payload["behaviour"].max(), payload["z_edges"].max(), payload["q_edges"].max())) + 1
    graphs = {
        "behaviour": payload["behaviour"],
        "z": payload["z_edges"],
        "q": payload["q_edges"],
    }
    result = {
        "orca": str(ORCA.relative_to(REPO)),
        "nodes": nodes,
        "graphs": {},
        "sizes": {},
        "five_node_status": (
            "skipped: serial ORCA node-5 on the 315,391-edge behaviour graph was still running "
            "after 6 minutes with no output; the 4-node exact orbit comparison is reported instead"
        ),
    }
    for size in (4,):
        tables = {}
        for name, edges in graphs.items():
            print(f"ORCA node {size}: {name} ({len(edges)} edges)", flush=True)
            tables[name] = run_orca(name, nodes, edges, size)
        result["sizes"][str(size)] = {
            "z_vs_behaviour": compare(tables["behaviour"], tables["z"]),
            "q_vs_behaviour": compare(tables["behaviour"], tables["q"]),
            "z_vs_q": compare(tables["z"], tables["q"]),
        }
        for name, table in tables.items():
            result["graphs"].setdefault(name, {})[str(size)] = list(table.shape)

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# ORCA orbit comparison\n"]
    for size in (4,):
        block = result["sizes"][str(size)]
        lines.append(f"\n## node {size}-graphlet orbits\n\n")
        lines.append("| comparison | median Spearman | mean | p10 | p90 | profile cosine |\n")
        lines.append("|---|---:|---:|---:|---:|---:|\n")
        for label, key in (("z vs behaviour", "z_vs_behaviour"), ("q vs behaviour", "q_vs_behaviour"), ("z vs q", "z_vs_q")):
            row = block[key]
            lines.append(
                f"| {label} | {row['median_spearman']:.4f} | {row['mean_spearman']:.4f} | "
                f"{row['p10_spearman']:.4f} | {row['p90_spearman']:.4f} | {row['profile_cosine_mean']:.4f} |\n"
            )
    MD.write_text("".join(lines) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
