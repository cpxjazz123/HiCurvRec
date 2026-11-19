"""The level-1 behaviour structure term: closer neighbours, verified graph.

The mechanism is ``L = L_RQVAE + lambda * L_graph^(1)`` with
``L_graph^(1) = mean ||z_i - z_j||^2 / mean ||z||^2`` over behaviour edges and
``lambda`` calibrated once so that the term is a fixed share (10%) of the total
loss at the first step -- a measured quantity, not a tuned constant.

These tests fix the four properties the implementation must have:

* identical neighbours contribute nothing;
* pulling neighbours together lowers the term;
* the gradient reaches the encoder parameters (a term that is computed but not
  connected is the failure the encoder-gradient probe exists for);
* the edges come from the canonical graph and a graph whose sha256 no longer
  matches its metadata is refused (rule 89b).
"""
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

from structure_loss import (calibrate_share, graph_level1_loss, load_edges,  # noqa: E402
                            sample_edges)


def _graph_dir(tmp_path: Path, corrupt: bool = False) -> Path:
    folder = tmp_path / "behavior_graph"
    folder.mkdir()
    np.savez_compressed(folder / "graph.npz", nodes=np.array(4, dtype=np.int64),
                        rows=np.array([0, 1]), cols=np.array([1, 2]),
                        af3=np.zeros(2))
    metadata = {"schema": "genrec.behavior_graph.v1",
                "policy": {"single_undirected_graph": True, "directed_graph_stored": False},
                "outputs": {"graph.npz": {
                    "bytes": (folder / "graph.npz").stat().st_size,
                    "sha256": hashlib.sha256((folder / "graph.npz").read_bytes()).hexdigest()}}}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    if corrupt:
        np.savez_compressed(folder / "graph.npz", nodes=np.array(4, dtype=np.int64),
                            rows=np.array([0, 3]), cols=np.array([1, 2]), af3=np.zeros(2))
    return folder


def test_identical_neighbours_contribute_nothing():
    left = torch.ones(8, 4)
    right = left.clone()
    assert float(graph_level1_loss(left, right)) == pytest.approx(0.0, abs=1e-9)


def test_pulling_neighbours_together_lowers_the_term():
    torch.manual_seed(0)
    left = torch.randn(64, 8)
    right = torch.randn(64, 8)
    far = float(graph_level1_loss(left, right))
    near = float(graph_level1_loss(left, left + 0.01 * torch.randn(64, 8)))
    assert near < far


def test_gradient_reaches_the_encoder():
    encoder = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3))
    left = torch.randn(16, 8)
    right = torch.randn(16, 8)
    loss = graph_level1_loss(encoder(left), encoder(right))
    loss.backward()
    norms = [float(parameter.grad.abs().sum()) for parameter in encoder.parameters()
             if parameter.grad is not None]
    assert norms and min(norms) > 0


def test_calibration_sets_the_share():
    weight = calibrate_share(recon=1e-4, graph=2.0, share=0.1)
    assert weight == pytest.approx(0.1 * 1e-4 / (0.9 * 2.0))
    term = weight * 2.0
    assert term / (1e-4 + term) == pytest.approx(0.1, abs=1e-6)


def test_edges_come_from_the_verified_graph(tmp_path):
    edges = load_edges(_graph_dir(tmp_path))
    assert edges.shape == (2, 2)
    assert edges.tolist() == [[0, 1], [1, 2]]


def test_corrupted_graph_is_refused(tmp_path):
    with pytest.raises(RuntimeError) as error:
        load_edges(_graph_dir(tmp_path, corrupt=True))
    assert "sha256" in str(error.value) or "metadata" in str(error.value)


def test_sampling_is_deterministic_and_bounded():
    edges = np.arange(24).reshape(12, 2)
    generator = torch.Generator().manual_seed(7)
    first = sample_edges(edges, 5, generator)
    second = sample_edges(edges, 5, torch.Generator().manual_seed(7))
    assert first.shape == (5, 2)
    assert first.tolist() == second.tolist()
    assert len(sample_edges(edges, 50, torch.Generator().manual_seed(1))) == 12
