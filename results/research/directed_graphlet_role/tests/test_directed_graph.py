"""Issue #221 CGV 1: the directed graph is directed, TRAIN-only and reproducible.

Each assertion fails on the failure the issue names: folding (A,B)/(B,A) into one
row, keeping a symmetric CSR, leaking VALID/TEST, or drifting from the TRAIN
sequences the graph claims to describe.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
GRAPH_DIR = REPO / "results" / "research" / "behavior_graph_directed"
sys.path.insert(0, str(REPO / "analysis_sid_recall"))

from build_behavior_graph import sequences  # noqa: E402


def load():
    metadata = json.loads((GRAPH_DIR / "metadata.json").read_text())
    payload = np.load(GRAPH_DIR / "graph.npz", allow_pickle=False)
    edges = pd.read_parquet(GRAPH_DIR / "edges.parquet")
    return metadata, payload, edges


def test_no_ordered_pair_is_stored_twice_and_weights_are_recounted():
    _metadata, _payload, edges = load()
    pairs = list(zip(edges["source"].tolist(), edges["target"].tolist()))
    assert len(pairs) == len(set(pairs)), "an ordered pair is stored twice"
    wanted = set(pairs[::97][:400])
    recount = Counter()
    for sequence in sequences():
        for left, right in zip(sequence[:-1], sequence[1:]):
            key = (int(left), int(right))
            if key in wanted:
                recount[key] += 1
    recorded = {(int(a), int(b)): int(w) for a, b, w in
                zip(edges["source"], edges["target"], edges["weight"])}
    for key, count in recount.items():
        assert recorded[key] == count, (key, recorded[key], count)


def test_reverse_edges_are_kept_apart():
    _metadata, _payload, edges = load()
    keys = set(zip(edges["source"].tolist(), edges["target"].tolist()))
    reciprocal = [(a, b) for (a, b) in keys if (b, a) in keys]
    assert reciprocal, "no reciprocal pair exists at all"
    weights = {(int(a), int(b)): int(w) for a, b, w in
               zip(edges["source"], edges["target"], edges["weight"])}
    assert any(weights[pair] != weights[(pair[1], pair[0])] for pair in reciprocal), \
        "every reciprocal pair has equal weights -- folding is suspected"


def test_no_self_loops_and_all_ids_inside_the_item_range():
    metadata, _payload, edges = load()
    assert (edges["source"] != edges["target"]).all()
    assert int(edges["source"].max()) < metadata["counts"]["nodes"]
    assert int(edges["target"].max()) < metadata["counts"]["nodes"]


def test_csr_is_not_symmetric():
    from scipy import sparse
    _metadata, payload, _edges = load()
    out = sparse.csr_matrix((payload["out_data"], payload["out_indices"], payload["out_indptr"]),
                            shape=(int(payload["nodes"]), int(payload["nodes"])))
    assert (out != out.T).nnz > 0, "the stored adjacency came back symmetric"


def test_degrees_match_the_edge_list():
    _metadata, payload, edges = load()
    n = int(payload["nodes"])
    out = np.bincount(edges["source"].to_numpy(), minlength=n)
    inn = np.bincount(edges["target"].to_numpy(), minlength=n)
    assert np.array_equal(out, payload["out_degree"])
    assert np.array_equal(inn, payload["in_degree"])
    assert out.sum() == inn.sum() == len(edges)


def test_undirected_reference_accounts_for_every_directed_edge():
    metadata, _payload, _edges = load()
    counts = metadata["counts"]
    # a reciprocal pair is two directed edges and one undirected edge
    assert counts["directed_edges"] == counts["one_way_edges"] + counts["reciprocal_edges"]
    assert counts["reciprocal_edges"] == 2 * counts["reciprocal_pairs"]
    assert counts["undirected_reference_edges"] == (counts["one_way_edges"]
                                                    + counts["reciprocal_pairs"])


def test_outputs_match_the_recorded_sha():
    metadata, _payload, _edges = load()
    for name, record in metadata["outputs"].items():
        digest = hashlib.sha256((GRAPH_DIR / name).read_bytes()).hexdigest()
        assert digest == record["sha256"], name
        assert (GRAPH_DIR / name).stat().st_size == record["bytes"]
