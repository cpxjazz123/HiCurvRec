"""Issue #226: the leak rules of the graph Teacher, as executable assertions.

Each test pins one invariant that the previous `graph_teacher.py` violated or never
checked: pair-disjoint splits, a message graph that cannot contain a validation or
test pair (in either direction), training positives that are actually masked out of
that graph, negatives that are not true edges, a direction task with both classes,
a shuffled control that keeps the edge count and drops self-loops, and a checkpoint
that reloads bitwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import graph_teacher2 as gt  # noqa: E402


def _synthetic(seed: int = 0, n_items: int = 90, edges: int = 400):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n_items, edges)
    cols = rng.integers(0, n_items, edges)
    keep = rows != cols
    return rows[keep].astype(np.int64), cols[keep].astype(np.int64)


def _pair_set(pairs):
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs}


def test_split_is_pair_disjoint_and_covers_the_graph():
    rows, cols = _synthetic()
    payload = gt.split_pairs(rows, cols)
    train = _pair_set(payload["train"])
    valid = _pair_set(payload["valid"])
    test = _pair_set(payload["test"])
    assert not (train & valid) and not (train & test) and not (valid & test)
    everything = _pair_set(zip(rows.tolist(), cols.tolist()))
    assert train | valid | test == everything
    assert payload["split_sha"] == gt.split_pairs(rows, cols)["split_sha"], "split is not reproducible"


def test_valid_and_test_pairs_never_enter_the_message_graph():
    rows, cols = _synthetic()
    payload = gt.split_pairs(rows, cols)
    materialised = gt.materialise(payload, rows, cols)
    observed = materialised["observed_pairs"]
    for a, b in payload["valid"].tolist() + payload["test"].tolist():
        assert (min(a, b), max(a, b)) not in observed
        assert (a, b) not in materialised["observed_directed"]
        assert (b, a) not in materialised["observed_directed"]


def test_training_positives_are_masked_out_of_the_message_graph():
    rows, cols = _synthetic()
    payload = gt.split_pairs(rows, cols)
    materialised = gt.materialise(payload, rows, cols)
    assert len(payload["masked"]) > 0
    for a, b in payload["masked"].tolist():
        assert (a, b) not in materialised["observed_directed"]
        assert (b, a) not in materialised["observed_directed"]


def test_negatives_are_not_true_edges_in_either_direction():
    rows, cols = _synthetic()
    payload = gt.split_pairs(rows, cols)
    materialised = gt.materialise(payload, rows, cols)
    full = materialised["full_directed"]
    degrees = np.zeros(90)
    for a, b in zip(rows.tolist(), cols.tolist()):
        degrees[a] += 1
        degrees[b] += 1
    rng = np.random.default_rng(7)
    sampled = 0
    for a, b in payload["masked"][:200].tolist():
        candidate = gt.degree_matched_negative(int(b), degrees, materialised["forbidden"], rng)
        if candidate is None:
            continue
        sampled += 1
        assert (int(b), candidate) not in full and (candidate, int(b)) not in full
        assert candidate != int(b)
    assert sampled > 0, "no negative could be sampled; the test would be vacuous"


def test_direction_examples_contain_both_classes_and_are_balanced():
    rows, cols = _synthetic()
    payload = gt.split_pairs(rows, cols)
    materialised = gt.materialise(payload, rows, cols)
    examples = gt.direction_examples(payload["masked"], materialised["full_directed"],
                                     np.random.default_rng(11))
    counts = np.bincount(examples["labels"], minlength=2)
    assert counts.min() > 0, f"one class is missing: {counts.tolist()}"
    assert abs(int(counts[0]) - int(counts[1])) / counts.sum() < 0.15


def test_direction_labels_come_from_the_original_graph():
    rows, cols = _synthetic()
    payload = gt.split_pairs(rows, cols)
    materialised = gt.materialise(payload, rows, cols)
    examples = gt.direction_examples(payload["masked"], materialised["full_directed"],
                                     np.random.default_rng(13))
    forward = materialised["full_directed"]
    for left, right, label in zip(examples["rows"][:100].tolist(),
                                  examples["cols"][:100].tolist(),
                                  examples["labels"][:100].tolist()):
        presented = (left, right) in forward
        assert int(presented) == (1 - int(label)), \
            "the label does not match the direction present in the original graph"


def test_shuffled_control_keeps_the_edge_count_and_drops_self_loops():
    index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    adjacency = torch.sparse_coo_tensor(index, torch.ones(5), size=(6, 6)).coalesce()
    degrees = np.array([1, 2, 2, 2, 2, 1], dtype=np.float64)
    shuffled = gt.shuffled_adjacency(adjacency, degrees, seed=3)
    assert shuffled._nnz() == adjacency._nnz()
    new_index = shuffled.indices()
    assert bool((new_index[0] != new_index[1]).all()), "a self-loop appeared"


def test_checkpoint_reload_reproduces_the_outputs():
    torch.manual_seed(5)
    encoder = gt.GraphEncoder(in_dim=16, hidden=32, out_dim=8)
    heads = gt.PairHeads(dim=8, hidden=16)
    index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    adjacency = torch.sparse_coo_tensor(index, torch.ones(3), size=(5, 5)).coalesce()
    features = torch.randn(5, 16)
    with torch.no_grad():
        state = encoder(features, adjacency)
        before = heads.edge(heads.pair(state, torch.tensor([0, 2]), torch.tensor([1, 4])))
    path = HERE / "_scratch_graph_teacher2.pth"
    torch.save({"encoder": encoder.state_dict(), "heads": heads.state_dict()}, path)
    payload = torch.load(path, weights_only=False)
    encoder2 = gt.GraphEncoder(in_dim=16, hidden=32, out_dim=8)
    heads2 = gt.PairHeads(dim=8, hidden=16)
    encoder2.load_state_dict(payload["encoder"], strict=True)
    heads2.load_state_dict(payload["heads"], strict=True)
    with torch.no_grad():
        after = heads2.edge(heads2.pair(encoder2(features, adjacency),
                                        torch.tensor([0, 2]), torch.tensor([1, 4])))
    assert torch.equal(before, after)
    path.unlink()
