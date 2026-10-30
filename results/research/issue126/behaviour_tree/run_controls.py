"""The rewiring control, ten seeds in parallel.

The rewiring itself costs about a second; the distance computation is the
expensive part, so the ten seeds run as ten processes. The graph is built once
in the parent and shared with the workers.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from behaviour_tree import (  # noqa: E402
    four_point,
    rewire_preserving_degrees,
    sequential_interaction_graph,
)

REPO = HERE.parents[3]
PARQUET = REPO / "results" / "stage0_build_parquet"
N_ITEMS = 24587
SEED = 42
GLOBAL_QUADRUPLES = 50000
CONTROL_QUADRUPLES = 20000
REWIRING_SEEDS = 10
REWIRING_SWAPS = 40000

GRAPH = None
QUADRUPLES = None


def load():
    global GRAPH, QUADRUPLES
    if GRAPH is not None:
        return
    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    sequences = []
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        sequence = [int(item) for item in history]
        if target is not None:
            sequence.append(int(target))
        if len(sequence) > 1:
            sequences.append(sequence)
    GRAPH = sequential_interaction_graph(sequences, N_ITEMS)
    rng = np.random.default_rng(SEED)
    items = np.flatnonzero(np.asarray(GRAPH.indptr[1:]) > 0)
    labels, components = connected_components(GRAPH, directed=False)
    sizes = np.bincount(components)
    items = np.flatnonzero(components == int(np.argmax(sizes)))
    picks = np.empty((GLOBAL_QUADRUPLES, 4), dtype=np.int64)
    for index in range(GLOBAL_QUADRUPLES):
        picks[index] = rng.choice(items, size=4, replace=False)
    QUADRUPLES = picks[:CONTROL_QUADRUPLES]


def work(seed: int):
    load()
    stats = {}
    rewired = rewire_preserving_degrees(GRAPH, seed, REWIRING_SWAPS, stats)
    labels, components = connected_components(rewired, directed=False)
    sizes = np.bincount(components)
    result = four_point(rewired, QUADRUPLES)
    values = result.pop("values")
    result["seed"] = seed
    result["swaps_requested"] = stats["swaps_requested"]
    result["swaps_successful"] = stats["swaps_successful"]
    result["components"] = int(len(sizes))
    result["largest_component_items"] = int(sizes.max())
    result["largest_component_share"] = round(float(sizes.max() / N_ITEMS), 6)
    result["_values"] = values
    return result


def main() -> int:
    load()
    print("the graph has", GRAPH.nnz, "edges", flush=True)
    start = time.time()
    with ProcessPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(work, range(1000, 1000 + REWIRING_SEEDS)))
    print("the controls took", round(time.time() - start, 1), "s", flush=True)

    real = four_point(GRAPH, QUADRUPLES)
    real_values = real["values"]
    control_values = [row.pop("_values") for row in rows]
    for row in rows:
        row.pop("values", None)

    rng = np.random.default_rng(SEED + 1)
    boot = []
    for _ in range(2000):
        index = rng.integers(0, len(real_values), len(real_values))
        control_medians = [float(np.median(values[index])) for values in control_values]
        boot.append(float(np.median(real_values[index])) - float(np.mean(control_medians)))
    boot = np.asarray(boot)
    control_medians = [float(np.median(values)) for values in control_values]
    difference = float(real["median_delta"] - np.mean(control_medians))
    summary = {
        "real": {key: value for key, value in real.items() if key != "values"},
        "control_median_delta_mean": float(np.mean(control_medians)),
        "control_median_delta_std": float(np.std(control_medians)),
        "control_medians": control_medians,
        "effect_size_real_minus_control": difference,
        "paired_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                  float(np.percentile(boot, 97.5))],
        "paired_bootstrap_p_real_smaller": float(np.mean(boot < 0)),
        "verdict": ("the real graph is MORE tree-like than its degree-preserving "
                    "rewirings" if difference < 0 else
                    "the real graph is NOT more tree-like than its rewirings"),
    }
    pd.DataFrame(rows).to_csv(HERE / "rewiring_controls.csv", index=False)
    (HERE / "stage1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
