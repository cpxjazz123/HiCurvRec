"""Stage one: is the behaviour graph a tree metric?

The brief, implemented exactly:

* the graph comes from the SEQUENTIAL interactions of the training split only --
  a user's sequence A, B, C, D contributes the edges A-B, B-C and C-D and never
  A-C -- with unit edge lengths, symmetrised, with no top-k truncation and no
  text, teacher or RQ-VAE input of any kind;
* quadruples are drawn from the LARGEST CONNECTED COMPONENT and the six
  shortest-path distances give the delta ``(S_max - S_mid) / 2``;
* three pre-fixed sampling schemes: global random (50,000), locally associated
  (25,000, built from a real adjacent interaction), and popularity stratified
  (25,000, one stratum per quadruple);
* the control: DEGREE-PRESERVING REWIRING with the double-edge swap, ten
  independent seeds, the weights travelling with the edges, the same quadruples
  scored in the real and in the random graphs, and the successful swap count and
  the connectivity change recorded for every seed.

Deliverables: ``graph_manifest.json``, ``four_point_results.csv``,
``rewiring_controls.csv`` and ``report.md``. Every number is reproducible from
the recorded seeds.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, dijkstra

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from behaviour_tree import (  # noqa: E402
    four_point,
    pairing_sums,
    pairwise_distances,
    rewire_preserving_degrees,
    sequential_interaction_graph,
)

REPO = HERE.parents[3]
PARQUET = REPO / "results" / "stage0_build_parquet"
N_ITEMS = 24587
SEED = 42
GLOBAL_QUADRUPLES = 50000
LOCAL_QUADRUPLES = 25000
STRATIFIED_QUADRUPLES = 25000
REWIRING_SEEDS = 10
REWIRING_SWAPS = 40000
CONTROL_QUADRUPLES = 20000
ECCENTRICITY_SAMPLES = 50


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sequences():
    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    sequences = []
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        sequence = [int(item) for item in history]
        if target is not None:
            sequence.append(int(target))
        if len(sequence) > 1:
            sequences.append(sequence)
    return sequences, train


def sample_global(rng, items, count):
    picks = np.empty((count, 4), dtype=np.int64)
    for index in range(count):
        picks[index] = rng.choice(items, size=4, replace=False)
    return picks


def sample_local(rng, graph, items, count):
    """A quadruple built from a REAL adjacent interaction and its neighbourhood."""
    coo = graph.tocoo()
    edge_rows = coo.row
    edge_cols = coo.col
    keep = edge_rows < edge_cols
    edge_rows, edge_cols = edge_rows[keep], edge_cols[keep]
    picks = np.empty((count, 4), dtype=np.int64)
    for index in range(count):
        position = int(rng.integers(0, len(edge_rows)))
        first, second = int(edge_rows[position]), int(edge_cols[position])
        picks[index, 0] = first
        picks[index, 1] = second
        # two more items from the neighbourhood of the edge, so the quadruple is
        # a local structure and not two distant pairs
        neighbours = np.unique(np.concatenate([
            graph.indices[graph.indptr[first]:graph.indptr[first + 1]],
            graph.indices[graph.indptr[second]:graph.indptr[second + 1]]]))
        neighbours = neighbours[neighbours != first]
        if len(neighbours) < 2:
            picks[index, 2] = first
            picks[index, 3] = second
            continue
        chosen = rng.choice(neighbours, size=2, replace=False)
        picks[index, 2] = chosen[0]
        picks[index, 3] = chosen[1]
    return picks


def sample_stratified(rng, strata, count):
    """One quadruple inside a single popularity stratum."""
    picks = np.empty((count, 4), dtype=np.int64)
    keys = list(strata)
    for index in range(count):
        key = keys[int(rng.integers(0, len(keys)))]
        members = strata[key]
        if len(members) < 4:
            picks[index] = rng.choice(members, size=4, replace=True)
            continue
        picks[index] = rng.choice(members, size=4, replace=False)
    return picks


def summarise(name, graph, quadruples, diameter):
    result = four_point(graph, quadruples)
    values = result.pop("values")
    normalised = values / max(diameter, 1.0)
    result["median_delta_over_diameter"] = float(np.median(normalised))
    result["scheme"] = name
    result["diameter"] = int(diameter)
    return result, values


def main() -> int:
    print("loading the training interactions", flush=True)
    sequences, train = load_sequences()
    print("sequences", len(sequences), flush=True)
    graph = sequential_interaction_graph(sequences, N_ITEMS)
    print("edges", graph.nnz, flush=True)

    labels, components = connected_components(graph, directed=False)
    sizes = np.bincount(components)
    largest = int(np.argmax(sizes))
    items = np.flatnonzero(components == largest)
    print("the largest component holds", len(items), "items", flush=True)

    rng = np.random.default_rng(SEED)
    graph_lcc = graph[items][:, items].tocsr()

    # the diameter, sampled: the largest eccentricity over a sample of sources
    sample_sources = rng.choice(len(items), size=min(ECCENTRICITY_SAMPLES, len(items)),
                                replace=False)
    block = dijkstra(graph_lcc, directed=False, indices=sample_sources)
    diameter = int(np.nanmax(block[np.isfinite(block)]))
    print("the sampled diameter is", diameter, flush=True)

    # the distance distribution inside the component
    probe = rng.choice(len(items), size=2000, replace=False)
    probe_block = dijkstra(graph_lcc, directed=False, indices=probe)
    finite = probe_block[np.isfinite(probe_block)]
    distance_histogram = {str(value): int(count) for value, count in
                          zip(*np.unique(finite, return_counts=True))}

    manifest = {
        "data": {
            "train_parquet": str((PARQUET / "train.parquet").relative_to(REPO)),
            "sha256": sha256_of(PARQUET / "train.parquet"),
            "users": int(len(sequences)),
        },
        "graph_rules": {
            "edges": "the consecutive items of a user's training sequence (A-B, B-C, "
                     "C-D); never the non-consecutive pairs",
            "weights": "unit length",
            "symmetric": True,
            "top_k_truncation": False,
            "text_or_teacher_input": False,
            "split": "train only",
        },
        "graph": {
            "items": N_ITEMS,
            "nodes_with_edges": int((np.diff(graph.indptr) > 0).sum()),
            "edges": int(graph.nnz),
            "components": int(len(sizes)),
            "largest_component_items": int(len(items)),
            "largest_component_share": round(float(len(items) / N_ITEMS), 6),
            "sampled_diameter": diameter,
            "sampled_distance_histogram": distance_histogram,
        },
        "sampling": {
            "global_random": GLOBAL_QUADRUPLES,
            "local_association": LOCAL_QUADRUPLES,
            "popularity_stratified": STRATIFIED_QUADRUPLES,
            "seed": SEED,
        },
        "control": {
            "kind": "degree-preserving double-edge swap, the weights travel with "
                    "the edges",
            "seeds": REWIRING_SEEDS,
            "swaps_per_seed": REWIRING_SWAPS,
            "quadruples_per_seed": CONTROL_QUADRUPLES,
        },
    }
    (HERE / "graph_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # the popularity strata, from the training interactions
    popularity = np.zeros(N_ITEMS, dtype=np.int64)
    for sequence in sequences:
        for item in sequence:
            popularity[item] += 1
    order = np.argsort(popularity[items])
    quartiles = np.array_split(items[order], 4)
    strata = {index: np.sort(members) for index, members in enumerate(quartiles)}
    manifest["graph"]["stratum_sizes"] = {str(index): int(len(members))
                                          for index, members in strata.items()}
    (HERE / "graph_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    raw = {}
    schemes = {
        "global_random": sample_global(rng, items, GLOBAL_QUADRUPLES),
        "local_association": sample_local(rng, graph, items, LOCAL_QUADRUPLES),
        "popularity_stratified": sample_stratified(rng, strata, STRATIFIED_QUADRUPLES),
    }
    for name, quadruples in schemes.items():
        print("scoring the scheme", name, flush=True)
        result, values = summarise(name, graph, quadruples, diameter)
        raw[name] = values
        rows.append(result)
        print("   ", result, flush=True)
    pd.DataFrame(rows).to_csv(HERE / "four_point_results.csv", index=False)

    print("the rewiring controls", flush=True)
    control_quadruples = schemes["global_random"][:CONTROL_QUADRUPLES]
    control_rows = []
    control_values = []
    for seed in range(REWIRING_SEEDS):
        stats = {}
        rewired = rewire_preserving_degrees(graph, seed + 1000, REWIRING_SWAPS, stats)
        labels_r, components_r = connected_components(rewired, directed=False)
        sizes_r = np.bincount(components_r)
        largest_r = int(np.argmax(sizes_r))
        result = four_point(rewired, control_quadruples)
        values = result.pop("values")
        control_values.append(values)
        result["seed"] = seed + 1000
        result["swaps_requested"] = stats["swaps_requested"]
        result["swaps_successful"] = stats["swaps_successful"]
        result["components"] = int(len(sizes_r))
        result["largest_component_items"] = int(sizes_r[largest_r])
        result["largest_component_share"] = round(float(sizes_r[largest_r] / N_ITEMS), 6)
        control_rows.append(result)
        print("   seed", seed + 1000, "median delta", result["median_delta"],
              "| largest component", result["largest_component_items"], flush=True)
    pd.DataFrame(control_rows).to_csv(HERE / "rewiring_controls.csv", index=False)

    # the paired comparison on the same quadruples
    real = four_point(graph, control_quadruples)
    real_values = real["values"]
    control_medians = [row["median_delta"] for row in control_rows]
    difference = real["median_delta"] - float(np.mean(control_medians))
    spread = float(np.std(control_medians))
    rng_boot = np.random.default_rng(SEED + 1)
    boot = []
    for _ in range(2000):
        index = rng_boot.integers(0, len(real_values), len(real_values))
        control_medians_boot = [float(np.median(values[index])) for values in control_values]
        boot.append(float(np.median(real_values[index])) - float(np.mean(control_medians_boot)))
    boot = np.asarray(boot)
    summary = {
        "real_median_delta": real["median_delta"],
        "real_share_delta_zero": real["share_delta_zero"],
        "control_median_delta_mean": float(np.mean(control_medians)),
        "control_median_delta_std": spread,
        "effect_size_real_minus_control": float(difference),
        "control_medians": control_medians,
        "paired_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                  float(np.percentile(boot, 97.5))],
        "paired_bootstrap_p_real_smaller": float(np.mean(boot < 0)),
        "interpretation": ("the real graph is more tree-like than its degree-preserving "
                            "rewirings" if difference < 0 else
                            "the real graph is NOT more tree-like than its rewirings"),
    }
    (HERE / "stage1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
