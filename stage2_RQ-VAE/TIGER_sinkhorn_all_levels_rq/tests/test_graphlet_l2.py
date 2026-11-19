"""E2: L2 graphlet supervision with hard negatives inside the L1 group."""
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

from graphlet_l2 import (GroupPairSampler, calibrate_share, calibrate_temperature,  # noqa: E402
                         graphlet2_infonce, l2_soft, load_graph, role_slices,
                         shared_probability, soft_assignment)


def _graph_dir(tmp_path: Path, corrupt: bool = False) -> Path:
    folder = tmp_path / "behavior_graph"
    folder.mkdir()
    rows = np.array([0, 1, 2, 3])
    cols = np.array([1, 2, 3, 4])
    np.savez_compressed(folder / "graph.npz", nodes=np.array(5, dtype=np.int64),
                        rows=rows, cols=cols, af3=np.zeros(len(rows)))
    metadata = {"schema": "genrec.behavior_graph.v1",
                "policy": {"single_undirected_graph": True, "directed_graph_stored": False},
                "outputs": {"graph.npz": {
                    "bytes": (folder / "graph.npz").stat().st_size,
                    "sha256": hashlib.sha256((folder / "graph.npz").read_bytes()).hexdigest()}}}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    if corrupt:
        np.savez_compressed(folder / "graph.npz", nodes=np.array(5, dtype=np.int64),
                            rows=np.array([0]), cols=np.array([1]), af3=np.zeros(1))
    return folder


def test_shared_probability_bounds():
    one_hot = torch.tensor([[1.0, 0.0]])
    disjoint = torch.tensor([[0.0, 1.0]])
    assert float(shared_probability(one_hot, one_hot)) == pytest.approx(1.0)
    assert float(shared_probability(one_hot, disjoint)) == pytest.approx(0.0)


def test_soft_assignment_is_a_distribution():
    soft = soft_assignment(torch.rand(4, 6), epsilon=0.003)
    assert torch.allclose(soft.sum(dim=1), torch.ones(4), atol=1e-5)


def test_infonce_prefers_the_structurally_closer_pair():
    positive = torch.tensor([0.8, 0.9])
    negative = torch.tensor([0.1, 0.2])
    assert float(graphlet2_infonce(positive, negative, 0.5)) < float(
        graphlet2_infonce(negative, positive, 0.5))


def test_gradient_reaches_the_encoder():
    encoder = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3))
    anchors = encoder(torch.randn(8, 8))
    positives = encoder(torch.randn(8, 8))
    negatives = encoder(torch.randn(8, 8))
    codebook = torch.randn(6, 3)

    def s2(first, second):
        soft = [soft_assignment(((z[:, None, :] - codebook[None]) ** 2).sum(2), 0.003)
                for z in (first, second)]
        return shared_probability(*soft)

    loss = graphlet2_infonce(s2(anchors, positives), s2(anchors, negatives), 1.0)
    loss.backward()
    norms = [float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None]
    assert norms and min(norms) > 0


def test_pairs_stay_inside_the_group_and_negatives_are_not_neighbours(tmp_path):
    graph = load_graph(_graph_dir(tmp_path))
    sampler = GroupPairSampler(graph)
    # Every item shares one L1 code, so the groups are the whole graph.
    sampler.rebuild(np.zeros(graph["n_nodes"], dtype=np.int64))
    triples = sampler.sample(torch.Generator().manual_seed(2), count=64)
    edges = {(int(a), int(b)) for a, b in zip(graph["rows"], graph["cols"])}
    edges |= {(b, a) for a, b in edges}
    assert len(triples)
    for anchor, positive, negative in triples:
        assert (int(anchor), int(positive)) in edges
        assert (int(anchor), int(negative)) not in edges
        assert sampler.codes[int(anchor)] == sampler.codes[int(positive)] == sampler.codes[int(negative)]


def test_empty_group_pool_is_reported(tmp_path):
    graph = load_graph(_graph_dir(tmp_path))
    sampler = GroupPairSampler(graph)
    codes = np.full(graph["n_nodes"], -1, dtype=np.int64)
    sampler.rebuild(codes)
    with pytest.raises(RuntimeError):
        sampler.sample(torch.Generator().manual_seed(3), count=4)


def test_calibrations():
    positive = torch.tensor([0.7, 0.7])
    negative = torch.tensor([0.2, 0.2])
    assert calibrate_temperature(positive, negative) == pytest.approx(0.5)
    # While the gap is zero the spread carries the scale instead of a fixed floor.
    flat = calibrate_temperature(torch.full((8,), 0.3), torch.full((8,), 0.3))
    assert 0 < flat < 1e-3
    weight = calibrate_share(recon=1e-4, term=2.0, share=0.1)
    assert (weight * 2.0) / (1e-4 + weight * 2.0) == pytest.approx(0.1, abs=1e-6)


def test_corrupted_graph_is_refused(tmp_path):
    with pytest.raises(RuntimeError) as error:
        load_graph(_graph_dir(tmp_path, corrupt=True))
    assert "sha256" in str(error.value) or "metadata" in str(error.value)


def test_roles_are_split_by_stride_not_by_contiguous_thirds():
    soft = torch.arange(9, dtype=torch.float).reshape(3, 3)   # rows a1,p1,n1
    anchor, positive, negative = role_slices(soft, count=1)
    assert anchor.tolist() == [[0.0, 1.0, 2.0]]
    assert positive.tolist() == [[3.0, 4.0, 5.0]]
    assert negative.tolist() == [[6.0, 7.0, 8.0]]
    # three triples: rows a1 p1 n1 a2 p2 n2 a3 p3 n3
    soft = torch.arange(9 * 2, dtype=torch.float).reshape(9, 2)
    anchor, positive, negative = role_slices(soft, count=3)
    assert anchor.shape == (3, 2) and positive.shape == (3, 2)
    assert anchor[:, 0].tolist() == [0.0, 6.0, 12.0]
    with pytest.raises(ValueError):
        role_slices(soft, count=2)


class _Level:
    """A level with a fixed codebook, standing in for a VQLayer."""

    def __init__(self, codes):
        self.codes = torch.as_tensor(codes, dtype=torch.float)

    def get_code_embs(self):
        return self.codes

    def _distances(self, latent):
        return ((latent[:, None, :] - self.codes[None, :, :]) ** 2).sum(dim=2)


def test_level2_probabilities_use_the_level1_residual():
    level1 = _Level([[0.0], [10.0]])
    level2 = _Level([[0.0], [1.0]])
    latent = torch.tensor([[9.9], [10.1]])
    on_residual = l2_soft(level1, level2, latent, epsilon=0.5)
    quantised = torch.tensor([[10.0], [10.0]])
    residual = latent - quantised
    expected = soft_assignment(level2._distances(residual), 0.5)
    on_latent = soft_assignment(level2._distances(latent), 0.5)
    assert torch.allclose(on_residual, expected, atol=1e-6)
    assert not torch.allclose(on_residual, on_latent, atol=1e-3)


def test_negatives_must_share_the_anchor_degree_bucket(tmp_path):
    graph = load_graph(_graph_dir(tmp_path))
    sampler = GroupPairSampler(graph)
    one_group = np.zeros(graph["n_nodes"], dtype=np.int64)
    sampler.rebuild(one_group)
    triples = sampler.sample(torch.Generator().manual_seed(5), count=32)
    for anchor, _positive, negative in triples:
        assert sampler.degree[int(anchor)] > 0
        # the fixture's degrees span 1..2, which fall in different buckets
        from graphlet_l2 import _degree_bucket
        assert _degree_bucket(int(sampler.degree[int(negative)])) == _degree_bucket(
            int(sampler.degree[int(anchor)]))
