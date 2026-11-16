"""Run exact ORCA node-5 orbits on the GPU and compare TIGER z/q with behaviour.

The GPU binary is the rebuilt stage-1 port described in the historical issue213 GPU README.  Its
4-node output was first checked byte-for-byte against the stock CPU ORCA; only then is the 5-node
output used.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/orca5_structure.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPHS = HERE / "cluster_graphs.npz"
ORCA_GPU = REPO / "results" / "orca_build" / "orca" / "orca_gpu"
RUN = REPO / "results" / "orca_build" / "runs"
OUT = HERE / "orca5_structure.json"
MD = HERE / "orca5_structure.md"


def write_graph(path: Path, nodes: int, edges: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write(f"{nodes} {len(edges)}\n")
        for left, right in edges.tolist():
            handle.write(f"{int(left)} {int(right)}\n")


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
    graphs = {"behaviour": payload["behaviour"], "z": payload["z_edges"], "q": payload["q_edges"]}
    RUN.mkdir(parents=True, exist_ok=True)
    timings = {}
    for name, edges in graphs.items():
        in_path = RUN / f"{name}_5.in"
        out_path = RUN / f"{name}_gpu_5.out"
        if not in_path.exists():
            write_graph(in_path, nodes, edges)
        if not out_path.exists() or out_path.stat().st_size == 0:
            start = time.time()
            subprocess.run([str(ORCA_GPU), "node", "5", str(in_path), str(out_path)], check=True)
            timings[name] = time.time() - start
        else:
            timings[name] = None
        table = np.loadtxt(out_path, dtype=np.float64)
        if table.ndim == 1:
            table = table.reshape(1, -1)
        graphs[name] = table
        print(f"{name}: {table.shape}, {timings[name]}", flush=True)

    result = {
        "orca_gpu": str(ORCA_GPU.relative_to(REPO)),
        "nodes": nodes,
        "columns": int(graphs["behaviour"].shape[1]),
        "timings_seconds": timings,
        "z_vs_behaviour": compare(graphs["behaviour"], graphs["z"]),
        "q_vs_behaviour": compare(graphs["behaviour"], graphs["q"]),
        "z_vs_q": compare(graphs["z"], graphs["q"]),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# ORCA node-5 orbit comparison\n"]
    lines.append("| comparison | median Spearman | mean | p10 | p90 | profile cosine |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for label, key in (("z vs behaviour", "z_vs_behaviour"), ("q vs behaviour", "q_vs_behaviour"), ("z vs q", "z_vs_q")):
        row = result[key]
        lines.append(
            f"| {label} | {row['median_spearman']:.4f} | {row['mean_spearman']:.4f} | "
            f"{row['p10_spearman']:.4f} | {row['p90_spearman']:.4f} | {row['profile_cosine_mean']:.4f} |\n"
        )
    lines.append("\nGPU runtimes (seconds): " + json.dumps(timings) + "\n")
    MD.write_text("".join(lines) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
