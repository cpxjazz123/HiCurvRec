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
    return out


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


def work(number):
    """One arm: build its quantised k-NN graph and score every structure on the same items."""
    arm = next(entry for entry in study.load_arms(67) if entry["number"] == number)
    n = 24587
    tokens = study.tokens_of(arm["sid_path"], n)
    vectors, geometry = study.quantised_vectors(number, tokens, n)
    if vectors is None:
        return {"number": number, "reason": geometry.get("reason")}
    edges, graph = tools.knn_edges(vectors[:n])
    edge_set = {(int(a), int(b)) for a, b in edges.tolist()}
    row = {"number": number, "recall": arm["recall"], "graph": graph}
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
