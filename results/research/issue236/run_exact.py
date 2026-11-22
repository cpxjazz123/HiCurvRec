"""Exact sectional curvature over every reference node of the principal component.

Issue #235 averaged over 512 sampled references; #236 removes that approximation.
The identity that makes it tractable (verified against brute force in tests/):

    kappa(m;b,c) = S[m]/R + d(b,c)^2/4 - (S[b]/R + S[c]/R)/2
    S[v] = sum_{a in refs} d(a,v)^2,  R = number of refs reaching all three endpoints

so one pass of breadth-first searches accumulates S once, and the 40.6M triples become
a linear combination.  The dense distance matrix is never materialised (C12); sums are
exact Python ints (C13); unreachable references are excluded, never zeroed (C14).

Usage (no arguments, per CLAUDE.md 9-10):
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/issue236/run_exact.py
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sectional_exact as sx  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue236"
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
METADATA = REPO / "results" / "research" / "behavior_graph" / "metadata.json"
PREV = REPO / "results" / "research" / "issue235"
NEAR_ZERO = 1e-9
BFS_CHUNK = 512


def load_adjacency() -> tuple[sp.csr_matrix, dict]:
    """Unit adjacency from the CSR **pattern**; ``data`` holds AF3 values, not weights."""
    raw = np.load(GRAPH, allow_pickle=True)
    n = len(raw["indptr"]) - 1
    adjacency = sp.csr_matrix(
        (np.ones(len(raw["indices"]), dtype=np.int8), raw["indices"], raw["indptr"]),
        shape=(n, n))
    return adjacency, json.loads(METADATA.read_text(encoding="utf-8"))


def principal_component(adjacency: sp.csr_matrix) -> np.ndarray:
    _, labels = sp.csgraph.connected_components(adjacency, directed=False)
    sizes = np.bincount(labels)
    biggest = int(np.argmax(sizes))
    return np.flatnonzero(labels == biggest)


def batch_bfs_square_sums(adjacency: sp.csr_matrix, references: np.ndarray,
                          nodes: np.ndarray) -> tuple[dict, dict, dict]:
    """Accumulate ``S[v]`` and the reference count by multi-source frontier expansion.

    Each chunk carries ``B`` references at once: a frontier matrix ``(B, n)`` marks the
    nodes reached in the current layer, and one sparse product produces the next layer.
    Only the frontier (not a dense distance matrix) is held per chunk.
    """
    n = adjacency.shape[0]
    square_sum = {int(node): 0 for node in nodes}
    counts = {int(node): 0 for node in nodes}
    max_distance = 0
    for start in range(0, len(references), BFS_CHUNK):
        chunk = references[start:start + BFS_CHUNK]
        rows = np.arange(len(chunk))
        frontier = np.zeros((len(chunk), n), dtype=np.float32)
        frontier[rows, chunk] = 1.0
        distances = np.zeros((len(chunk), n), dtype=np.int16)
        reached = frontier.astype(bool)
        step = 0
        while True:
            step += 1
            nxt = (adjacency @ frontier.T).T
            np.clip(nxt, 0, 1, out=nxt)
            nxt[reached] = 0.0
            nxt = nxt.astype(np.float32)
            if not nxt.any():
                break
            hit = nxt > 0
            distances[hit] = step
            reached |= hit
            frontier = nxt
        max_distance = max(max_distance, step - 1)
        for node in nodes:
            column = distances[:, node]
            mask = reached[:, node]
            square_sum[int(node)] += int((column[mask].astype(np.int64) ** 2).sum())
            counts[int(node)] += int(mask.sum())
    return square_sum, counts, {"max_distance": max_distance,
                                "references": int(len(references))}


def enumerate_triples(adjacency: sp.csr_matrix, geodesic: bool):
    """All ``(centre, left, right)`` index arrays, vectorised per centre."""
    indptr, indices = adjacency.indptr, adjacency.indices
    centres, lefts, rights = [], [], []
    for centre in range(adjacency.shape[0]):
        arms = indices[indptr[centre]:indptr[centre + 1]]
        if len(arms) < 2:
            continue
        first, second = np.triu_indices(len(arms), k=1)
        left = arms[first].astype(np.int64)
        right = arms[second].astype(np.int64)
        if geodesic:
            keep = np.asarray(adjacency[left, right]).ravel() == 0
            left, right, first = left[keep], right[keep], first[keep]
            if left.size == 0:
                continue
        centres.append(np.full(left.size, centre, dtype=np.int64))
        lefts.append(left)
        rights.append(right)
    return (np.concatenate(centres), np.concatenate(lefts), np.concatenate(rights))


def curvature(square_sum: dict, counts: dict, centres: np.ndarray, lefts: np.ndarray,
              rights: np.ndarray, pair_square: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.array(sorted(square_sum))
    index = {int(node): position for position, node in enumerate(nodes)}
    sums = np.array([square_sum[int(node)] for node in nodes], dtype=np.float64)
    refs = np.array([counts[int(node)] for node in nodes], dtype=np.int64)
    c = np.searchsorted(nodes, centres)
    l = np.searchsorted(nodes, lefts)
    r = np.searchsorted(nodes, rights)
    denominator = np.minimum(np.minimum(refs[c], refs[l]), refs[r]).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = (sums[c] / denominator + pair_square / 4.0
                  - (sums[l] / denominator + sums[r] / denominator) / 2.0)
    keep = denominator > 0
    return values[keep], denominator[keep]


def summarise(values: np.ndarray) -> dict:
    ordered = np.sort(values)
    counts, edges = np.histogram(values, bins=40)
    cdf_counts, cdf_edges = np.histogram(values, bins=40)
    cumulative = np.cumsum(counts) / counts.sum()
    return {
        "triples": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(ordered[0]), "max": float(ordered[-1]),
        "p05": float(np.percentile(values, 5)), "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)), "p95": float(np.percentile(values, 95)),
        "positive_fraction": float((values > NEAR_ZERO).mean()),
        "negative_fraction": float((values < -NEAR_ZERO).mean()),
        "near_zero_fraction": float((np.abs(values) <= NEAR_ZERO).mean()),
        "near_zero_tolerance": NEAR_ZERO,
        "histogram_counts": counts.tolist(),
        "histogram_edges": edges.tolist(),
        "cdf": cumulative.tolist(),
        "cdf_edges": cdf_edges.tolist(),
    }


def main() -> int:
    started = time.time()
    adjacency, metadata = load_adjacency()
    n = adjacency.shape[0]
    degree = np.diff(adjacency.indptr).astype(int)
    principal = principal_component(adjacency)
    isolated = np.flatnonzero(degree == 0)

    square_sum, counts, bfs_info = batch_bfs_square_sums(adjacency, principal, principal)
    bfs_done = time.time()
    print(f"BFS over {len(principal)} references in {bfs_done - started:.0f}s "
          f"(max distance {bfs_info['max_distance']})", flush=True)

    results = {
        "graph": {
            "path": str(GRAPH.relative_to(REPO)),
            "sha256": metadata["outputs"]["graph.npz"]["sha256"],
            "nodes": int(n),
            "edges": int(adjacency.nnz // 2),
            "isolated": int(isolated.size),
            "components_with_edges": 1,
            "principal_component": int(principal.size),
            "principal_fraction": float(principal.size / n),
            "degree_median": float(np.median(degree)),
            "degree_p90": float(np.percentile(degree, 90)),
            "degree_max": int(degree.max()),
        },
        "references": {"count": int(len(principal)), "source": "principal component",
                       "max_distance": bfs_info["max_distance"]},
        "subsets": {"isolated_nodes": int(isolated.size),
                    "nodes_without_reference_coverage": int(n - len(principal))},
    }

    for mode, geodesic in (("geodesic", True), ("neighbour", False)):
        centres, lefts, rights = enumerate_triples(adjacency, geodesic)
        pair_square = np.where(np.asarray(adjacency[lefts, rights]).ravel() > 0, 1.0, 4.0)
        values, denominators = curvature(square_sum, counts, centres, lefts, rights,
                                         pair_square)
        entry = summarise(values)
        entry["mode"] = mode
        entry["centres_covered"] = int(np.unique(centres).size)
        entry["denominator_median"] = float(np.median(denominators))
        entry["denominator_min"] = float(denominators.min())
        entry["denominator_max"] = float(denominators.max())
        results[mode] = entry
        print(f"{mode}: triples={entry['triples']} mean={entry['mean']:.6f} "
              f"pos={entry['positive_fraction']:.5f} neg={entry['negative_fraction']:.5f}",
              flush=True)

    results["timing"] = {
        "total_s": round(time.time() - started, 1),
        "bfs_s": round(bfs_done - started, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    results["assumptions"] = {
        "reference_set": "every node of the principal component (24,556), no sampling",
        "denominator": "references reaching centre and both neighbours",
        "distance": "unweighted hops on the undirected projection",
        "normalisation": "none; kappa is in squared hops",
        "direction": "undirected projection: no statement about edge direction",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary_exact.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"summary -> {OUT / 'summary_exact.json'} ({results['timing']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
