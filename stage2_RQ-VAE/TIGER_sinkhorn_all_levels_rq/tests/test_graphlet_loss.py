"""Graphlet supervision: S1 on the L1 soft assignment, triangles against wedges."""
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

from graphlet_loss import (TripletSampler, calibrate_share, calibrate_temperature,  # noqa: E402
                           graphlet_infonce, load_graph, shared_probability, soft_assignment)


def _graph_dir(tmp_path: Path, corrupt: bool = False) -> Path:
    folder = tmp_path / "behavior_graph"
    folder.mkdir()
    # 0-1-2 is a triangle; 3-4-5 is an open wedge at 4 whose leaves (3 and 5) have
    # the same degree as the triangle's leaves, with 6 closing their degrees.
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
                            rows=np.array([0, 1]), cols=np.array([1, 2]), af3=np.zeros(2),
                            triangles=np.zeros(2, dtype=np.int64))
    return folder


def test_shared_probability_is_one_for_a_shared_codeword():
    one_hot = torch.tensor([[1.0, 0.0, 0.0]])
    assert float(shared_probability(one_hot, one_hot, one_hot)) == pytest.approx(1.0)


def test_shared_probability_is_zero_for_disjoint_supports():
    first = torch.tensor([[1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0]])
    assert float(shared_probability(first, second, first)) == pytest.approx(0.0)


def test_soft_assignment_is_a_distribution():
    distances = torch.rand(5, 7)
    soft = soft_assignment(distances, epsilon=0.003)
    assert torch.allclose(soft.sum(dim=1), torch.ones(5), atol=1e-5)


def test_infonce_prefers_triangles_with_higher_shared_probability():
    triangle = torch.tensor([0.9, 0.8])
    wedge = torch.tensor([0.2, 0.1])
    good = float(graphlet_infonce(triangle, wedge, temperature=0.5))
    bad = float(graphlet_infonce(wedge, triangle, temperature=0.5))
    assert good < bad


def test_gradient_reaches_the_encoder():
    encoder = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3))
    codebook = torch.randn(6, 3, requires_grad=False)
    closed = encoder(torch.randn(8, 8))
    open_ = encoder(torch.randn(8, 8))
    third_closed = encoder(torch.randn(8, 8))
    third_open = encoder(torch.randn(8, 8))

    def s1(first, second, third):
        soft = [soft_assignment(((z[:, None, :] - codebook[None]) ** 2).sum(2), 0.003)
                for z in (first, second, third)]
        return shared_probability(*soft)

    loss = graphlet_infonce(s1(closed, third_closed, closed), s1(open_, third_open, open_), 1.0)
    loss.backward()
    norms = [float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None]
    assert norms and min(norms) > 0


def test_calibrations():
    assert calibrate_temperature(0.8, 0.2) == pytest.approx(0.6)
    weight = calibrate_share(recon=1e-4, term=2.0, share=0.1)
    term = weight * 2.0
    assert term / (1e-4 + term) == pytest.approx(0.1, abs=1e-6)


def test_triangles_are_closed_and_wedges_are_open_and_matched(tmp_path):
    graph = load_graph(_graph_dir(tmp_path))
    sampler = TripletSampler(graph)
    generator = torch.Generator().manual_seed(1)
    triangles, wedges = sampler.sample(generator, count=32)
    assert len(triangles) and len(triangles) == len(wedges)
    edges = {(int(a), int(b)) for a, b in zip(graph["rows"], graph["cols"])}
    edges |= {(b, a) for a, b in edges}
    for centre, left, right in triangles:
        assert (int(centre), int(left)) in edges
        assert (int(centre), int(right)) in edges
        assert (int(left), int(right)) in edges
    for centre, left, right in wedges:
        assert (int(centre), int(left)) in edges
        assert (int(centre), int(right)) in edges
        assert (int(left), int(right)) not in edges


def test_corrupted_graph_is_refused(tmp_path):
    with pytest.raises(RuntimeError) as error:
        load_graph(_graph_dir(tmp_path, corrupt=True))
    assert "sha256" in str(error.value) or "metadata" in str(error.value)
