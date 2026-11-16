"""Cache behaviour-graph graphlet instances for SID structural fidelity.

Uses the same pattern definitions as the previous structure analyses.  Only the instance list is
saved; no SID or arm is involved.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/sample_graphlets.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
ANALYSIS = REPO / "analysis_sid_recall" / "analysis.py"
OUT = HERE / "graphlet_instances_3000.npz"
MAX_INSTANCES = 3000
SEED = 42


def edges() -> tuple[int, list[set[int]]]:
    frame = pd.read_parquet(TRAIN, columns=["user", "seen_history", "target"])
    best = {}
    for user, history, target in zip(
        frame["user"].to_numpy(), frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)
    ):
        history = [] if history is None else [int(x) for x in history]
        user = int(user)
        if user not in best or len(history) > best[user][0]:
            best[user] = (len(history), [*history, int(target)])
    pairs = set()
    for _, session in best.values():
        for left, right in zip(session, session[1:]):
            pairs.add((left, right) if left < right else (right, left))
    nodes = int(max(max(left, right) for left, right in pairs)) + 1
    adjacency = [set() for _ in range(nodes)]
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return nodes, adjacency


def main() -> int:
    spec = importlib.util.spec_from_file_location("sid_analysis", ANALYSIS)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    nodes, adjacency = edges()
    cliques4 = analysis.enumerate_cliques(adjacency, 4)
    cliques5 = analysis.enumerate_cliques(adjacency, 5)
    print(f"cliques 4={len(cliques4)} 5={len(cliques5)}", flush=True)
    rng = np.random.default_rng(SEED)
    patterns = analysis.build_patterns(adjacency, rng, cliques4, cliques5, MAX_INSTANCES)
    arrays = {
        name: np.array([instance["nodes"] for instance in instances], dtype=np.int64)
        for name, instances in patterns.items()
    }
    np.savez_compressed(OUT, **arrays)
    for name, value in arrays.items():
        print(name, value.shape, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
