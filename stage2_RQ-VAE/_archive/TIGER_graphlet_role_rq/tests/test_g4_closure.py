"""Issue #216: the four-arm machinery over the frozen closure/branch pool."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(TREE))

import g4_closure as g4  # noqa: E402


def test_pool_is_hash_checked(tmp_path, monkeypatch):
    pool = g4.load_pool("heldout")
    assert pool["triples"].shape[1] == 4
    assert set(np.unique(pool["labels"]).tolist()) == {0, 1}
    monkeypatch.setattr(g4, "POOL_DIR", tmp_path)
    with pytest.raises(RuntimeError):
        g4.load_pool("heldout")


def test_heldout_never_shares_both_classes_on_one_pair():
    pool = g4.load_pool("heldout")
    pairs = {(row[0], row[1], row[2]) for row in pool["triples"].tolist()}
    assert len(pairs) == len(pool["triples"]), "a triple appears twice"


def test_edge_sampler_returns_edges_and_matched_non_edges():
    graph = g4.load_graph()
    sampler = g4.EdgeSampler(graph)
    positives, negatives = sampler.sample(torch.Generator().manual_seed(0), count=200)
    assert len(positives) and len(negatives)
    for a, b in positives:
        assert (int(a), int(b)) in graph["edges"]
    for a, b in negatives:
        assert (int(a), int(b)) not in graph["edges"]
        assert g4._degree_bucket(int(graph["degree"][a])) == g4._degree_bucket(int(graph["degree"][b]))


def test_pair_sampler_is_balanced_and_labelled():
    pool = g4.load_pool("train")
    sampler = g4.PairSampler(pool)
    positive, negative = sampler.sample(torch.Generator().manual_seed(1), count=64)
    assert len(positive) == len(negative)
    labels = {(row[0], row[1], row[2]): label for row, label in zip(pool["triples"], pool["labels"])}
    for row in positive[:20]:
        assert labels[(int(row[0]), int(row[1]), int(row[2]))] == 1
    for row in negative[:20]:
        assert labels[(int(row[0]), int(row[1]), int(row[2]))] == 0


def test_infonce_and_bce_prefer_the_closed_pair():
    high = torch.tensor([0.8, 0.9])
    low = torch.tensor([0.1, 0.2])
    assert float(g4.pairwise_infonce(high, low, 0.5)) < float(g4.pairwise_infonce(low, high, 0.5))
    assert float(g4.triplet_bce(high, low)) < float(g4.triplet_bce(low, high))


def test_arm_c_and_d_heads_differ_in_the_interaction_branch():
    torch.manual_seed(0)
    add = g4.TripletPredictor(dim=8, hidden=16, interaction=False).eval()
    inter = g4.TripletPredictor(dim=8, hidden=16, interaction=True).eval()
    a, b, c = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    with torch.no_grad():
        assert add(a, b, c).shape == inter(a, b, c).shape == (4,)
    assert add.pair is None and inter.pair is not None


def test_gradient_reaches_the_encoder():
    encoder = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3))
    head = g4.TripletPredictor(dim=3, hidden=8, interaction=True)
    positive = encoder(torch.randn(8, 8))
    negative = encoder(torch.randn(8, 8))
    loss = g4.triplet_bce(head(positive, positive, positive), head(negative, negative, negative))
    loss.backward()
    norms = [float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None]
    assert norms and min(norms) > 0


def test_weight_calibration_hits_the_gradient_share():
    weight = g4.calibrate_weight(norm_rq=2.0, norm_aux=40.0, share=0.1)
    assert weight * 40.0 == pytest.approx(0.1 * 2.0)
