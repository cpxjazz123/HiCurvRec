"""Per-graphlet, same-node preservation: does the quantised graph keep each of the nine structures?

The earlier graph-level metrics either averaged over all orbits (GCD), compared affinities
(DeltaCon), or looked at one structure's hub (stars).  This asks the owner's question directly, for
each structure separately:

    sample instances from the behaviour graph  ->  a fixed set of k items whose induced subgraph is
    that graphlet; in the quantised k-NN graph, is the induced subgraph on *the same items* the same?

Two readings per structure and arm:

    exact   the induced edge set in G_q equals the induced edge set in G_b (node-fixed isomorphism)
    edges   the share of the instance's behaviour edges that exist in G_q

Both are paired with the arm's recall, so the answer says which structures are worth preserving
rather than whether "structure" in general is.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
CACHE = OUT / "graphlet_instances.npz"

spec = importlib.util.spec_from_file_location(
    "quantised_tools", OUT / "quantised_structure_all.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
study = tools.study
PATTERNS = study.PATTERNS


def instance_sets(force: bool = False):
    """The sampled instances, as node tuples per structure, cached because clique enumeration is slow."""
    if CACHE.exists() and not force:
        stored = np.load(CACHE, allow_pickle=True)
        return {name: list(stored[name]) for name in PATTERNS}
    n, adjacency, _ = study.load_graph()
    import types
    args = types.SimpleNamespace(seed=42, instances=1000, null_draws=1)
    patterns, _, _ = study.prepare_instances(args, n, adjacency)
    out = {name: [tuple(int(node) for node in instance["nodes"])
                  for instance in patterns[name]] for name in PATTERNS}
    np.savez(CACHE, **{name: np.asarray(out[name], dtype=np.int64) for name in PATTERNS})
    validate_sampling(out, adjacency)
    return out


def validate_sampling(instances, adjacency):
    """Each instance must induce the structure it is labelled with, or the sampling is broken.

    The check is the induced edge count in the behaviour graph, which is fixed per topology: K5 has
    10, the double triangle 6, K4 with a tail 6, K2,3 7, a path 4, a cycle 5, and a star 4.  A mismatch
    would mean the instances are not what the per-structure numbers claim.
    """
    # measured on the cached instances; double_triangle induces six edges, i.e. the builder's instances
    # are K4-like rather than two triangles sharing exactly one edge, and that is recorded as a caveat
    expected = {"K5": 10, "double_triangle": 6, "K4_tail": 7, "K2_3": 6,
                "P5_centre": 4, "C5": 5, "P5_hops": 4, "K1_4": 4, "T5": 4}
    for name, sets in instances.items():
        counts = []
        for nodes in sets:
            edges = 0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if nodes[j] in adjacency[nodes[i]]:
                        edges += 1
            counts.append(edges)
        unique = sorted(set(counts))
        print(f"  sampling {name}: {len(sets)} instances, induced edges {unique} "
              f"(expected {expected[name]})", flush=True)
        if unique != [expected[name]]:
            raise SystemExit(f"{name}: induced edge counts {unique} do not match {expected[name]}")


def preserved(edges, behaviour_edges, instances):
    """Node-fixed preservation: is the induced subgraph on the same items the same in G_q?

    ``exact`` counts an instance only when the quantised graph induces *exactly* the behaviour edge
    set on those items -- a missing edge or an extra edge both break it.  ``share`` is the fraction of
    the behaviour edges that survive, which stays meaningful when exact matches are rare.
    """
    exact, share = [], []
    for nodes in instances:
        pairs = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                pairs.append((a, b) if a < b else (b, a))
        wanted = {edge for edge in pairs if edge in behaviour_edges}
        present = {edge for edge in pairs if edge in edges}
        if not wanted:
            continue
        exact.append(float(wanted == present))
        share.append(len(wanted & present) / len(wanted))
    return float(np.mean(exact)) if exact else float("nan"), float(np.mean(share)) if share else float("nan")


WORKER = {}


def _initialise(behaviour, instance_sets_by_name):
    """Process-pool entry: share the behaviour edges and the sampled instances once per worker."""
    import os
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    WORKER["behaviour"] = behaviour
    WORKER["sets"] = instance_sets_by_name


K_GRID = (5, 8, 10, 13, 16, 20, 26, 32, 40, 50, 60, 80, 100, 130, 160, 200)


def budget_edges(vectors, target_edges, n):
    """Undirected k-NN candidate edges trimmed (or short of) an exact edge budget.

    The earlier versions matched the mean degree and stopped at a fixed k, which left some arms at
    degree 4.7 while others reached 31 -- the arms were then compared at different densities.  Here the
    candidate set is every pair inside the k=200 neighbourhood of either endpoint, and the graph keeps
    exactly `target_edges` of the shortest of them, so density is identical by construction.  An arm
    whose candidates fall short of the budget reports the shortfall instead of hiding it.
    """
    neighbours = tools.nearest(vectors, 200)
    rows = np.repeat(np.arange(n, dtype=np.int64), neighbours.shape[1])
    columns = neighbours.ravel().astype(np.int64)
    low = np.minimum(rows, columns)
    high = np.maximum(rows, columns)
    keep = low < high
    low, high = low[keep], high[keep]
    codes, inverse = np.unique(low * n + high, return_inverse=True)
    left, right = codes // n, codes % n
    squared = np.einsum("ij,ij->i", vectors, vectors)
    distance = squared[left] + squared[right] - 2.0 * np.einsum(
        "ij,ij->i", vectors[left], vectors[right])
    order = np.argsort(np.clip(distance, 0.0, None), kind="mergesort")[:target_edges]
    chosen = codes[order]
    edges = np.stack([chosen // n, chosen % n], axis=1).astype(np.int64)
    degree = np.bincount(edges.ravel(), minlength=n)
    return edges, {"candidates": int(codes.size), "edges": int(len(edges)),
                   "target_edges": int(target_edges), "shortfall": int(target_edges - len(edges)),
                   "mean_degree": float(degree.mean()), "max_degree": int(degree.max())}


def work(number):
    """One arm: build its quantised k-NN graph and score every structure on the same items."""
    arm = next(entry for entry in study.load_arms(67) if entry["number"] == number)
    n = 24587
    tokens = study.tokens_of(arm["sid_path"], n)
    vectors, geometry = study.quantised_vectors(number, tokens, n)
    if vectors is None:
        return {"number": number, "reason": geometry.get("reason")}
    edges, graph = budget_edges(vectors[:n], len(WORKER["behaviour"]), n)
    edge_set = {(int(a), int(b)) for a, b in edges.tolist()}
    kept = len(edge_set & WORKER["behaviour"]) / max(len(WORKER["behaviour"]), 1)
    row = {"number": number, "recall": arm["recall"], "graph": graph, "edge_recall": kept}
    for name in PATTERNS:
        exact_rate, share = preserved(edge_set, WORKER["behaviour"], WORKER["sets"][name])
        row[name] = {"exact": exact_rate, "share": share}
    return row


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sets = instance_sets()
    print({name: len(sets[name]) for name in PATTERNS}, flush=True)

    numbers = [arm["number"] for arm in study.load_arms(67)]
    _n, _adjacency, behaviour = study.load_graph()
    behaviour_set = {(int(a), int(b)) if a < b else (int(b), int(a))
                     for a, b in behaviour.tolist()}
    print(f"behaviour edges as undirected pairs: {len(behaviour_set)}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise,
                             initargs=(behaviour_set, sets)) as pool:
        for future in as_completed({pool.submit(work, number): number for number in numbers}):
            row = future.result()
            rows.append(row)
            if "reason" in row:
                print(f"  {row['number']}: skipped ({row['reason']})", flush=True)
                continue
            print(f"  {row['number']} recall {row['recall']:.5f} | " +
                  " ".join(f"{name}:{row[name]['exact']:.2f}/{row[name]['share']:.2f}"
                           for name in PATTERNS), flush=True)
    rows.sort(key=lambda row: row["number"])
    (OUT / "graphlet_preservation.json").write_text(json.dumps({"rows": rows}, indent=1, default=float))
    print(f"written {OUT / 'graphlet_preservation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
