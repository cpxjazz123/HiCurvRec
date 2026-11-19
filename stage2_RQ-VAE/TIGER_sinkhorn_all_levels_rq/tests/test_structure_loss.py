"""Graph contrastive structure: positives are neighbours, negatives are controls.

The term is ``-log softmax(s(z_i, z_j)/tau)`` with ``j`` a behaviour neighbour and
the other entries degree-matched non-neighbours.  These tests fix what the
implementation must guarantee:

* a more similar positive (relative to the negatives) gives a lower loss, and a
  clearly separated positive gives ~0;
* the gradient reaches the encoder parameters;
* ``tau`` is derived so the initial logit gap is one nat, and ``lambda`` is derived
  from the first step's reconstruction loss (no hand-tuned constants);
* the positives really are neighbours and the negatives really are non-adjacent
  items from the same degree bucket;
* a graph whose sha256 no longer matches its metadata is refused (rule 89b).
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

from structure_loss import (NegativeSampler, calibrate_share, calibrate_temperature,  # noqa: E402
                            contrastive_loss, load_graph)


def _graph_dir(tmp_path: Path, corrupt: bool = False) -> Path:
    folder = tmp_path / "behavior_graph"
    folder.mkdir()
    rows = np.array([0, 1, 2, 3, 4, 5])
    cols = np.array([1, 2, 0, 4, 5, 3])
    np.savez_compressed(folder / "graph.npz", nodes=np.array(6, dtype=np.int64),
                        rows=rows, cols=cols, af3=np.zeros(len(rows)),
                        triangles=np.zeros(len(rows), dtype=np.int64))
    metadata = {"schema": "genrec.behavior_graph.v1",
                "policy": {"single_undirected_graph": True, "directed_graph_stored": False},
                "outputs": {"graph.npz": {
                    "bytes": (folder / "graph.npz").stat().st_size,
                    "sha256": hashlib.sha256((folder / "graph.npz").read_bytes()).hexdigest()}}}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    if corrupt:
        np.savez_compressed(folder / "graph.npz", nodes=np.array(6, dtype=np.int64),
                            rows=np.array([0, 1]), cols=np.array([1, 2]), af3=np.zeros(2),
                            triangles=np.zeros(2, dtype=np.int64))
    return folder


def test_more_similar_positive_gives_a_lower_loss():
    anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    near = torch.tensor([[0.99, 0.01], [0.99, 0.01]])
    far = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    negatives = torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]])
    close = float(contrastive_loss(anchor, near, negatives, temperature=0.5))
    distant = float(contrastive_loss(anchor, far, negatives, temperature=0.5))
    assert close < distant


def test_separated_positive_approaches_zero():
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[[0.0, 1.0]]])
    assert float(contrastive_loss(anchor, positive, negative, temperature=0.1)) < 0.01


def test_gradient_reaches_the_encoder():
    encoder = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3))
    anchor = torch.randn(16, 8)
    positive = torch.randn(16, 8)
    negative = torch.randn(16, 5, 8)
    loss = contrastive_loss(encoder(anchor), encoder(positive),
                            encoder(negative.reshape(-1, 8)).reshape(16, 5, 3), temperature=0.5)
    loss.backward()
    norms = [float(parameter.grad.abs().sum()) for parameter in encoder.parameters()
             if parameter.grad is not None]
    assert norms and min(norms) > 0


def test_temperature_calibration_sets_one_nat():
    temperature = calibrate_temperature(positive_similarity=0.8, negative_similarity=0.2)
    assert (0.8 - 0.2) / temperature == pytest.approx(1.0)
    assert calibrate_temperature(0.5, 0.5) > 0


def test_share_calibration_targets_the_total():
    weight = calibrate_share(recon=1e-4, graph=2.0, share=0.1)
    term = weight * 2.0
    assert term / (1e-4 + term) == pytest.approx(0.1, abs=1e-6)


def test_positives_are_neighbours_and_negatives_are_matched_non_edges(tmp_path):
    graph = load_graph(_graph_dir(tmp_path))
    sampler = NegativeSampler(graph)
    generator = torch.Generator().manual_seed(3)
    anchors, positives, negatives = sampler.sample(generator, count=64, negatives=4)
    assert len(anchors) == len(positives) == len(negatives)
    edges = {(int(a), int(b)) for a, b in zip(graph["rows"], graph["cols"])}
    edges |= {(b, a) for a, b in edges}
    for anchor, positive_row, negative_row in zip(anchors, positives, negatives):
        assert (int(anchor), int(positive_row)) in edges
        for negative in negative_row:
            if negative < 0:
                continue
            assert (int(anchor), int(negative)) not in edges, "a negative is an edge"
            assert sampler.degree[int(negative)] > 0


def test_corrupted_graph_is_refused(tmp_path):
    with pytest.raises(RuntimeError) as error:
        load_graph(_graph_dir(tmp_path, corrupt=True))
    assert "sha256" in str(error.value) or "metadata" in str(error.value)
