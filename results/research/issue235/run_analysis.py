"""Measure the GraphMoRE sectional-curvature proxy on the frozen behaviour graph.

Reads the single canonical graph (CLAUDE.md 89b) and never rebuilds it.

Three facts about that file and the formula drive the implementation:

* The CSR in ``graph.npz`` stores **AF3 values in ``data``** (negative, down to -2511)
  and marks adjacency purely by ``indices``/``indptr``.  Reading ``data`` as weights
  yields negative degrees; the adjacency is rebuilt from the pattern with unit entries.
* The graph is the undirected simple projection of the training sequences, so a
  curvature sign describes a local neighbourhood and says nothing about the direction
  of a user transition (issue #235 C6).
* Eq. A.1 averages over reference nodes ``a``.  A full 24587x24587 distance table is
  604M entries, so the analysis fixes a seeded reference set and reports its size; the
  number is an estimate over that set, never presented as an exact full mean.

Two triple definitions are reported apart (issue #235 C9/C10):

* ``geodesic``  -- ``d(b,c) == 2`` and the centre on a shortest b-c path;
* ``neighbour`` -- any two distinct neighbours, adjacent ones included.

Usage (no arguments, per CLAUDE.md 9-10):
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/issue235/run_analysis.py
"""
from __future__ import annotations

import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue235"
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
METADATA = REPO / "results" / "research" / "behavior_graph" / "metadata.json"
SEED = 42
REFERENCE_SAMPLE = 512
#: Neighbour pairs kept per centre when a hub owns more (degree 2220 -> 2.4M pairs).
TRIPLES_PER_CENTRE = 10**9
#: |kappa| at or below this counts as near zero. Reported with every table.
NEAR_ZERO = 1e-9

ASSUMPTIONS = {
    "reference_set": ("Eq. A.1 averages over reference nodes a. We fix a seeded set of "
                      "%d nodes and average over those that reach all three endpoints, "
                      "so the mean is an estimate on that set." % REFERENCE_SAMPLE),
    "midpoint": ("A geodesic triple requires d(b,c) == 2 with the centre on a shortest "
                 "b-c path, which holds exactly when b and c are not adjacent."),
    "distance": "d is the unweighted hop count on the undirected projection.",
    "normalisation": "kappa is raw (unit: squared hops); no rescaling is applied.",
    "components": ("A triple is dropped when no reference node reaches all three "
                   "endpoints; unreachable distances are never treated as zero."),
    "direction": ("The graph is an undirected projection; no statement about edge "
                  "direction can be derived from these numbers."),
    "triple_cap": ("Centres with more than %d neighbour pairs are subsampled uniformly "
                   "with the global seed; the achieved coverage is reported per mode."
                   % TRIPLES_PER_CENTRE),
}


def load_graph() -> tuple[sp.csr_matrix, dict]:
    raw = np.load(GRAPH, allow_pickle=True)
    n = len(raw["indptr"]) - 1
    adjacency = sp.csr_matrix(
        (np.ones(len(raw["indices"]), dtype=np.int8), raw["indices"], raw["indptr"]),
        shape=(n, n))
    return adjacency, json.loads(METADATA.read_text(encoding="utf-8"))


def components(adjacency: sp.csr_matrix) -> tuple[np.ndarray, list[int]]:
    n = adjacency.shape[0]
    label = np.full(n, -1, dtype=np.int32)
    sizes: list[int] = []
    indptr, indices = adjacency.indptr, adjacency.indices
    for start in range(n):
        if label[start] != -1:
            continue
        current = len(sizes)
        queue = deque([start])
        label[start] = current
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for nxt in indices[indptr[node]:indptr[node + 1]]:
                if label[nxt] == -1:
                    label[nxt] = current
                    queue.append(nxt)
        sizes.append(size)
    return label, sizes


def bfs_from(adjacency: sp.csr_matrix, source: int) -> np.ndarray:
    """Distance from ``source`` to every node; -1 where unreachable."""
    n = adjacency.shape[0]
    distances = np.full(n, -1, dtype=np.int32)
    distances[source] = 0
    queue = deque([source])
    indptr, indices = adjacency.indptr, adjacency.indices
    while queue:
        node = queue.popleft()
        step = distances[node] + 1
        for nxt in indices[indptr[node]:indptr[node + 1]]:
            if distances[nxt] == -1:
                distances[nxt] = step
                queue.append(nxt)
    return distances


def reference_tables(adjacency: sp.csr_matrix, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(distances, reachable)`` shaped ``(R, N)`` for the seeded reference set."""
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(adjacency.shape[0]), REFERENCE_SAMPLE))
    distances = np.stack([bfs_from(adjacency, ref) for ref in picked])
    return distances, distances >= 0


def enumerate_triples(adjacency: sp.csr_matrix, cap: int,
                      seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """All ``(centre, left, right)`` neighbour pairs, capped per centre.

    The cap only binds on hubs, so the returned coverage record keeps the exact and
    sampled pair counts apart: presenting a capped count as the exact one is what
    issue #235 C13/C17 forbids.
    """
    rng = random.Random(seed)
    indptr, indices = adjacency.indptr, adjacency.indices
    centres: list[np.ndarray] = []
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    exact_pairs = 0
    capped_centres = 0
    for centre in range(adjacency.shape[0]):
        arms = indices[indptr[centre]:indptr[centre + 1]]
        if len(arms) < 2:
            continue
        # pairs (i, j) with i < j over the neighbour list, built without Python tuples
        first, second = np.triu_indices(len(arms), k=1)
        pair_count = first.size
        exact_pairs += pair_count
        if pair_count > cap:
            keep = sorted(rng.sample(range(pair_count), cap))
            first, second = first[keep], second[keep]
            capped_centres += 1
        centres.append(np.full(first.size, centre, dtype=np.int64))
        lefts.append(arms[first].astype(np.int64))
        rights.append(arms[second].astype(np.int64))
    used = int(sum(chunk.size for chunk in centres))
    coverage = {
        "exact_neighbour_pairs": int(exact_pairs),
        "sampled_neighbour_pairs": used,
        "coverage": used / exact_pairs if exact_pairs else None,
        "centres_capped": int(capped_centres),
        "cap_per_centre": int(cap),
    }
    return (np.concatenate(centres), np.concatenate(lefts), np.concatenate(rights), coverage)


def curvature(centres: np.ndarray, lefts: np.ndarray, rights: np.ndarray,
              distances: np.ndarray, reachable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eq. A.1 per triple, vectorised over both triples and references.

    ``square[a, v]`` is ``d(a, v)^2`` with unreachable entries masked to zero, so the
    masked sum is the numerator and the mask sum the denominator.
    """
    square = np.where(reachable, distances.astype(np.float32) ** 2, np.float32(0.0))
    usable = reachable[:, centres] & reachable[:, lefts] & reachable[:, rights]
    counts = usable.sum(axis=0)
    sum_m = (square[:, centres] * usable).sum(axis=0)
    sum_b = (square[:, lefts] * usable).sum(axis=0)
    sum_c = (square[:, rights] * usable).sum(axis=0)
    pair_square = np.where(centres >= 0, 4.0, 4.0).astype(np.float32)  # geodesic triples
    with np.errstate(invalid="ignore", divide="ignore"):
        values = (sum_m / counts + pair_square / 4.0
                  - (sum_b / counts + sum_c / counts) / 2.0)
    keep = counts > 0
    return values[keep], counts[keep]


def geodesic_mask(adjacency: sp.csr_matrix, lefts: np.ndarray,
                  rights: np.ndarray) -> np.ndarray:
    """``d(b,c) == 2``: both are neighbours of the centre and not adjacent to each other."""
    adjacent = np.asarray(adjacency[lefts, rights]).ravel().astype(bool)
    return ~adjacent


def summarise(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": None, "note": "no qualifying triple"}
    ordered = np.sort(values)
    counts, edges = np.histogram(values, bins=40)
    return {
        "triples": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(ordered[0]),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(ordered[-1]),
        "positive_fraction": float((values > NEAR_ZERO).mean()),
        "negative_fraction": float((values < -NEAR_ZERO).mean()),
        "near_zero_fraction": float((np.abs(values) <= NEAR_ZERO).mean()),
        "near_zero_tolerance": NEAR_ZERO,
        "histogram_counts": counts.tolist(),
        "histogram_edges": edges.tolist(),
    }


def main() -> int:
    started = time.time()
    adjacency, metadata = load_graph()
    n = adjacency.shape[0]
    degree = np.diff(adjacency.indptr).astype(int)
    _, sizes = components(adjacency)

    results: dict = {"graph": {
        "path": str(GRAPH.relative_to(REPO)),
        "sha256": metadata["outputs"]["graph.npz"]["sha256"],
        "nodes": int(n),
        "edges": int(adjacency.nnz // 2),
        "isolated": int((degree == 0).sum()),
        "degree_min": int(degree.min()),
        "degree_median": float(np.median(degree)),
        "degree_p90": float(np.percentile(degree, 90)),
        "degree_max": int(degree.max()),
        "components": len(sizes),
        "largest_component": int(max(sizes)),
        "largest_fraction": float(max(sizes) / n),
        "components_with_edges": int(sum(1 for s in sizes if s > 1)),
        "directed_source": metadata["counts"],
    }}
    print("graph stats done", flush=True)

    distances, reachable = reference_tables(adjacency, SEED)
    print(f"reference tables {distances.shape} built in {time.time() - started:.0f}s", flush=True)

    centres, lefts, rights, coverage = enumerate_triples(
        adjacency, TRIPLES_PER_CENTRE, SEED)
    print(f"triples enumerated {centres.size}; exact pairs {coverage['exact_neighbour_pairs']} "
          f"coverage {coverage['coverage']:.4%}; capped centres {coverage['centres_capped']}",
          flush=True)

    for mode in ("geodesic", "neighbour"):
        if mode == "geodesic":
            keep = geodesic_mask(adjacency, lefts, rights)
            c, l, r = centres[keep], lefts[keep], rights[keep]
        else:
            c, l, r = centres, lefts, rights
        values, counts = curvature(c, l, r, distances, reachable)
        entry = summarise(values)
        entry["mode"] = mode
        entry["definition"] = ("d(b,c)==2 with the centre on a shortest path" if mode == "geodesic"
                               else "any two distinct neighbours")
        entry["centres_covered"] = int(np.unique(c).size)
        entry["references_used_median"] = float(np.median(counts)) if counts.size else None
        results[mode] = entry
        print(f"{mode}: triples={entry['triples']} mean={entry['mean']:.6f} "
              f"pos={entry['positive_fraction']:.3f} neg={entry['negative_fraction']:.3f} "
              f"near0={entry['near_zero_fraction']:.3f}", flush=True)

    results["assumptions"] = ASSUMPTIONS
    results["seed"] = SEED
    results["reference_sample"] = REFERENCE_SAMPLE
    results["triple_coverage"] = coverage
    results["elapsed_s"] = round(time.time() - started, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"summary -> {OUT / 'summary.json'} in {results['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
