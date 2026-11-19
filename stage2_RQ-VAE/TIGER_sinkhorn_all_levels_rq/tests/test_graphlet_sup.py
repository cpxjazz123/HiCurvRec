"""Graphlet supervisor: a triplet classifier whose error trains the encoder."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(TREE))

from graphlet_sup import (TripletPredictor, TripletSampler, accuracy, calibrate_share,  # noqa: E402
                          graphlet_loss, load_graph)


def _graph_dir(tmp_path: Path, corrupt: bool = False) -> Path:
    folder = tmp_path / "behavior_graph"
    folder.mkdir()
    rows = np.array([0, 1, 0, 3, 4, 3, 5])
    cols = np.array([1, 2, 2, 4, 5, 6, 6])
    np.savez_compressed(folder / "graph.npz", nodes=np.array(7, dtype=np.int64),
                        rows=rows, cols=cols, af3=np.zeros(len(rows)),
                        triangles=np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int64))
    metadata = {"schema": "genrec.behavior_graph.v1",
                "policy": {"single_undirected_graph": True, "directed_graph_stored": False},
                "outputs": {"graph.npz": {
                    "bytes": (folder / "graph.npz").stat().st_size,
                    "sha256": hashlib.sha256((folder / "graph.npz").read_bytes()).hexdigest()}}}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    if corrupt:
        np.savez_compressed(folder / "graph.npz", nodes=np.array(7, dtype=np.int64),
                            rows=np.array([0]), cols=np.array([1]), af3=np.zeros(1),
                            triangles=np.zeros(1, dtype=np.int64))
    return folder


def test_predictor_is_permutation_invariant():
    torch.manual_seed(0)
    head = TripletPredictor(dim=8, hidden=16).eval()
    a, b, c = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    with torch.no_grad():
        direct = head(a, b, c)
        shuffled = head(c, a, b)
    assert torch.allclose(direct, shuffled, atol=1e-6)


def test_the_interaction_branch_changes_the_score():
    torch.manual_seed(1)
    with_interaction = TripletPredictor(dim=8, hidden=16, interaction=True).eval()
    without = TripletPredictor(dim=8, hidden=16, interaction=False).eval()
    a, b, c = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    with torch.no_grad():
        assert not torch.allclose(with_interaction(a, b, c), without(a, b, c))
    # scaling the pairwise products independently changes an additive-only head not at all
    swapped = (a * b + a * c + 2 * b * c) / 3.0
    node_term = with_interaction.node(torch.stack([a, b, c], 1)).mean(1)
    baseline = with_interaction.head(torch.cat([node_term, with_interaction.pair((a*b+a*c+b*c)/3)], 1))
    modified = with_interaction.head(torch.cat([node_term, with_interaction.pair(swapped)], 1))
    assert not torch.allclose(baseline, modified, atol=1e-6)


def test_gradient_reaches_the_encoder():
    encoder = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3))
    head = TripletPredictor(dim=3, hidden=8)
    triangle = encoder(torch.randn(8, 8))
    wedge = encoder(torch.randn(8, 8))
    loss = graphlet_loss(head(triangle, triangle, triangle), head(wedge, wedge, wedge))
    loss.backward()
    norms = [float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None]
    assert norms and min(norms) > 0


def test_loss_falls_when_triangles_score_higher():
    high = torch.full((8,), 2.0)
    low = torch.full((8,), -2.0)
    assert float(graphlet_loss(high, low)) < float(graphlet_loss(low, high))
    assert accuracy(high, low) == pytest.approx(1.0)
    assert accuracy(low, high) == pytest.approx(0.0)


def test_share_calibration_targets_the_total():
    weight = calibrate_share(recon=1e-4, term=2.0, share=0.1)
    assert (weight * 2.0) / (1e-4 + weight * 2.0) == pytest.approx(0.1, abs=1e-6)


def test_sampler_draws_closed_triangles_and_matched_wedges(tmp_path):
    graph = load_graph(_graph_dir(tmp_path))
    sampler = TripletSampler(graph)
    triangles, wedges = sampler.sample(torch.Generator().manual_seed(1), count=8)
    edges = {(int(a), int(b)) for a, b in zip(graph["rows"], graph["cols"])}
    edges |= {(b, a) for a, b in edges}
    assert len(triangles) and len(triangles) == len(wedges)
    for centre, left, right in triangles:
        assert (int(centre), int(left)) in edges and (int(centre), int(right)) in edges
        assert (int(left), int(right)) in edges
    for centre, left, right in wedges:
        assert (int(centre), int(left)) in edges and (int(centre), int(right)) in edges
        assert (int(left), int(right)) not in edges


def test_corrupted_graph_is_refused(tmp_path):
    with pytest.raises(RuntimeError) as error:
        load_graph(_graph_dir(tmp_path, corrupt=True))
    assert "sha256" in str(error.value) or "metadata" in str(error.value)
