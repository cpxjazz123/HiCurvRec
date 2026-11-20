"""Build the canonical DIRECTED behaviour graph (Issue #221).

Layout (results/research/behavior_graph_directed/):

    edges.parquet        source, target, weight   one row per ordered pair, no folding
    graph.npz            directed COO + out/in CSR, degrees, nodes, hashes' inputs
    node_features.npy    item embeddings, aligned by node id
    metadata.json        schema, sources, counts, direction statistics, output sha256

Every ordered pair ``(i_t -> i_{t+1})`` of the TRAIN sequences is one row, with
``weight`` the number of occurrences.  ``A -> B = 30`` and ``B -> A = 2`` are two
rows and are never merged; self-loops are dropped and counted.  The undirected
graph of #218/#220 stays where it is as a historical reference -- this builder
writes a different directory and never rewrites the old file.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_behavior_graph import FEATURES, ITEMS, TRAIN, sequences  # noqa: E402  (same textbook)

REPO = HERE.parent
OUT = REPO / "results" / "research" / "behavior_graph_directed"
SCHEMA = "genrec.behavior_graph.directed.v1"
UNDIRECTED = REPO / "results" / "research" / "behavior_graph"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directed_counts() -> tuple[Counter, int]:
    """Ordered transition counts over the TRAIN sequences, and the self-loop count."""
    counts: Counter = Counter()
    self_loops = 0
    for sequence in sequences():
        for left, right in zip(sequence[:-1], sequence[1:]):
            if left == right:
                self_loops += 1
                continue
            counts[(int(left), int(right))] += 1
    return counts, self_loops


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    counts, self_loops = directed_counts()
    sources = np.array([pair[0] for pair in counts], dtype=np.int64)
    targets = np.array([pair[1] for pair in counts], dtype=np.int64)
    weights = np.array([counts[pair] for pair in counts], dtype=np.int64)
    n_items = int(np.load(FEATURES, mmap_mode="r").shape[0])
    if sources.max() >= n_items or targets.max() >= n_items:
        raise ValueError("a transition names an item id outside the embedding table")

    order = np.lexsort((targets, sources))
    sources, targets, weights = sources[order], targets[order], weights[order]

    frame = pd.DataFrame({"source": sources, "target": targets, "weight": weights})
    edges_path = OUT / "edges.parquet"
    frame.to_parquet(edges_path, index=False)

    # directed CSR (out) and CSC-as-CSR (in); neither is forced symmetric
    from scipy import sparse
    out_csr = sparse.csr_matrix((weights.astype(np.float64), (sources, targets)),
                                shape=(n_items, n_items))
    in_csr = out_csr.T.tocsr()
    # degree counts edges; the weighted sums are separate fields, because a
    # degree that silently carries the weight cannot be compared with the
    # undirected reference's degree
    out_degree = np.diff(out_csr.indptr).astype(np.int64)
    in_degree = np.diff(in_csr.indptr).astype(np.int64)
    out_weight = np.asarray(out_csr.sum(axis=1)).ravel().astype(np.float64)
    in_weight = np.asarray(out_csr.sum(axis=0)).ravel().astype(np.float64)
    keys = set(zip(sources.tolist(), targets.tolist()))
    reciprocal = np.array([(b, a) in keys for a, b in zip(sources, targets)], dtype=bool)

    graph_path = OUT / "graph.npz"
    np.savez(
        graph_path,
        nodes=np.int64(n_items),
        rows=sources, cols=targets, weight=weights.astype(np.float64),
        out_indptr=out_csr.indptr.astype(np.int32),
        out_indices=out_csr.indices.astype(np.int32),
        out_data=out_csr.data.astype(np.float32),
        in_indptr=in_csr.indptr.astype(np.int32),
        in_indices=in_csr.indices.astype(np.int32),
        in_data=in_csr.data.astype(np.float32),
        out_degree=out_degree, in_degree=in_degree,
        out_weight=out_weight, in_weight=in_weight,
        reciprocal=reciprocal,
        directed=np.array(True),
        format=np.array("csr"),
    )
    features_path = OUT / "node_features.npy"
    np.save(features_path, np.load(FEATURES))

    # what the migration changed, against the historical undirected graph
    old_edges = pd.read_parquet(UNDIRECTED / "edges.parquet")
    old_pairs = len(old_edges)

    weight_values = weights.astype(np.float64)
    metadata = {
        "schema": SCHEMA,
        "definition": "E = {(i_t -> i_{t+1})} over the TRAIN sequences, ordered pairs only",
        "directed": True,
        "policy": {
            "source": "results/stage0_build_parquet/train.parquet, training histories only",
            "sequence": "one per user: longest history plus its target (same rule as the undirected build)",
            "edge": "consecutive distinct items; self-loops dropped and counted",
            "weight": "count of that ordered transition; reverse counts are never folded in",
            "symmetry": "none: out CSR and in CSC are stored separately and are not symmetric",
            "validation_test": "not read",
        },
        "sources": {
            "train.parquet": {"sha256": sha256(TRAIN)},
            "items.parquet": {"sha256": sha256(ITEMS)},
            "sentence_t5.npy": {"sha256": sha256(FEATURES)},
        },
        "counts": {
            "nodes": n_items,
            "directed_edges": int(len(sources)),
            "self_loops_dropped": int(self_loops),
            "total_transitions": int(weights.sum()),
            "reciprocal_edges": int(reciprocal.sum()),
            "reciprocal_pairs": int(reciprocal.sum()) // 2,
            "reciprocity_rate": float(reciprocal.mean()) if len(reciprocal) else 0.0,
            "one_way_edges": int((~reciprocal).sum()),
            "undirected_reference_edges": int(old_pairs),
            "asymmetry_note": "a directed edge count is expected to exceed the undirected one by exactly the non-reciprocal pairs",
        },
        "degree": {
            "out_max": int(out_degree.max()), "out_mean": float(out_degree.mean()),
            "in_max": int(in_degree.max()), "in_mean": float(in_degree.mean()),
            "out_median": float(np.median(out_degree)), "in_median": float(np.median(in_degree)),
        },
        "weight": {
            "max": float(weight_values.max()), "mean": float(weight_values.mean()),
            "median": float(np.median(weight_values)),
            "p90": float(np.percentile(weight_values, 90)),
        },
        "outputs": {
            "edges.parquet": {"bytes": edges_path.stat().st_size, "sha256": sha256(edges_path)},
            "graph.npz": {"bytes": graph_path.stat().st_size, "sha256": sha256(graph_path)},
            "node_features.npy": {"bytes": features_path.stat().st_size,
                                  "sha256": sha256(features_path)},
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    print(json.dumps({key: metadata[key] for key in ("counts", "degree", "weight")}, indent=1))


if __name__ == "__main__":
    main()
