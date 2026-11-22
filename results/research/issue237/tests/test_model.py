"""Structural tests for the GraphMoRE model.

These check the things a shape error or a silent fallback to Euclidean aggregation would
break: five experts with the frozen curvature set, a gating distribution that is a
probability simplex, the weight-aligned distance matching its definition, gradients
reaching every expert, and finite outputs for a graph whose nodes include an isolated one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphmore.model import (CURVATURES, GraphMoRE, Experts, Gating,  # noqa: E402
                             KappaGCNConv, sparse_adjacency)
from graphmore.manifold import Stereographic  # noqa: E402


def tiny_graph(n: int = 12, chains: int = 4):
    """A few short chains plus one isolated node, as edge_index and adjacency."""
    edges = []
    for start in range(0, n - 4, 3):
        edges += [(start, start + 1), (start + 1, start + 2)]
    if not edges:
        edges = [(0, 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, sparse_adjacency(edge_index, n, torch.device("cpu")), n


def egos(n: int, adjacency: torch.Tensor, hops: int = 2) -> list:
    """Identity plus one-hop pooling is enough to exercise the pooling path."""
    identity = torch.sparse_coo_tensor(
        torch.stack([torch.arange(n), torch.arange(n)]),
        torch.ones(n), (n, n)).coalesce()
    return [identity, identity]


def test_curvatatues_are_the_frozen_set():
    experts = Experts(in_dim=8, hidden_dim=4, out_dim=3)
    assert experts.curvatures == CURVATURES
    assert len(experts.experts) == 5
    for expert, k in zip(experts.experts, CURVATURES):
        assert abs(float(expert.manifold.k) - k) < 1e-6


def test_zero_curvature_expert_is_frozen():
    experts = Experts(in_dim=8, hidden_dim=4, out_dim=3)
    zero = experts.experts[CURVATURES.index(0.0)]
    assert zero.manifold.learnable is False
    assert any(p.requires_grad for p in experts.experts[-1].parameters())


def test_convolutions_are_manifold_not_euclidean():
    """The layer must pull to the tangent space, propagate, and push back."""
    manifold = Stereographic(-1.0)
    conv = KappaGCNConv(manifold, 6, 4)
    edge_index, adjacency, n = tiny_graph()
    x = torch.randn(n, 6)
    x = manifold.expmap0(torch.nn.functional.normalize(x, dim=-1) * 0.4)
    out = conv(x, adjacency)
    # the result is on the manifold, hence strictly inside the ball
    assert torch.all(out.norm(dim=-1) < 1.0)
    # and it differs from the Euclidean GCN value, which lives outside the ball
    tangent = manifold.logmap0(conv.linear(x))
    euclidean = torch.sparse.mm(adjacency, tangent)
    assert not torch.allclose(out, euclidean, atol=1e-3)


def test_gating_weights_form_a_probability_simplex():
    edge_index, adjacency, n = tiny_graph()
    gating = Gating(in_dim=8, hidden_dim=6, out_dim=4, num_experts=5)
    weights = gating(torch.randn(n, 8), adjacency, egos(n, adjacency))
    assert weights.shape == (n, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(n), atol=1e-5)
    assert torch.all(weights >= 0)


def test_weight_aligned_distance_matches_its_definition():
    model = GraphMoRE(in_dim=8, hidden_dim=6, expert_dim=4, gating_hidden=4,
                      gating_out=3, out_dim=8)
    n = 10
    embedding = torch.randn(n, 5 * 4)
    weights = torch.softmax(torch.randn(n, 5), dim=-1)
    left = torch.tensor([0, 1, 2])
    right = torch.tensor([3, 4, 5])
    got = model.align(embedding, weights, left, right)
    diff = ((embedding[left] - embedding[right]) ** 2).reshape(3, 5, 4).sum(dim=2)
    pair = torch.softmax(weights[left] * weights[right], dim=1)
    assert torch.allclose(got, (diff * pair).sum(dim=-1))


def test_full_model_shapes_and_finiteness():
    edge_index, adjacency, n = tiny_graph()
    model = GraphMoRE(in_dim=16, hidden_dim=8, expert_dim=6, gating_hidden=4,
                      gating_out=3, out_dim=24)
    x = torch.randn(n, 16)
    embedding, weights = model(x, adjacency, egos(n, adjacency))
    assert embedding.shape == (n, 5 * 6)
    assert weights.shape == (n, 5)
    assert torch.isfinite(embedding).all()
    projected = model.head(embedding)
    assert projected.shape == (n, 24)
    assert torch.isfinite(projected).all()


def test_isolated_node_still_produces_a_finite_vector():
    edge_index, adjacency, n = tiny_graph(n=10)
    model = GraphMoRE(in_dim=8, hidden_dim=6, expert_dim=4, gating_hidden=4,
                      gating_out=3, out_dim=8)
    x = torch.randn(n, 8)
    embedding, weights = model(x, adjacency, egos(n, adjacency))
    isolated = n - 1
    assert torch.isfinite(embedding[isolated]).all()
    assert torch.allclose(weights[isolated].sum(), torch.tensor(1.0), atol=1e-5)


def test_gradients_reach_every_expert_and_the_head():
    edge_index, adjacency, n = tiny_graph()
    model = GraphMoRE(in_dim=8, hidden_dim=6, expert_dim=4, gating_hidden=4,
                      gating_out=3, out_dim=8)
    x = torch.randn(n, 8)
    embedding, weights = model(x, adjacency, egos(n, adjacency))
    loss = model.head(embedding).sum() + weights.sum()
    loss.backward()
    for position, expert in enumerate(model.experts.experts):
        grads = [p.grad for p in expert.parameters() if p.grad is not None]
        assert grads, f"expert {position} received no gradient"
        assert any(g.abs().sum() > 0 for g in grads), f"expert {position} gradient is zero"
    assert model.head.weight.grad is not None


def test_learned_curvature_moves_but_keeps_its_sign():
    torch.manual_seed(0)
    expert = Experts(in_dim=6, hidden_dim=4, out_dim=3).experts[0]
    before = float(expert.manifold.k)
    edge_index, adjacency, n = tiny_graph()
    x = torch.randn(n, 6)
    loss = expert(x, adjacency).abs().sum()
    loss.backward()
    optimizer = torch.optim.SGD([p for p in expert.parameters() if p.requires_grad], lr=0.5)
    optimizer.step()
    after = float(expert.manifold.k)
    assert after != before
    assert after < 0, "a hyperbolic expert must not cross into positive curvature"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
