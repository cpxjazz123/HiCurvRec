"""Tests for the curvature-guided soft community experiment.

They pin the properties the experiment's conclusion rests on: the soft
assignment is a distribution, the branch combination has no hidden scale, the
hyperbolic branch really lands inside the ball, the two losses behave as the
protocol claims, the metrics are the ones the AF3 pipeline already uses, and
the behaviour graph is the frozen one the recorded AF3 measurements used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.research.science.community_curvature import data, metrics, model  # noqa: E402


def _undirected(rows: np.ndarray, cols: np.ndarray, size: int) -> sp.csr_matrix:
    """An undirected simple graph, the only kind the AF3 convention sees."""
    values = np.ones(rows.size)
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(size, size)).tocsr()
    out = matrix.maximum(matrix.transpose().tocsr()).tocsr()
    out.setdiag(0)
    out.eliminate_zeros()
    return out


# ------------------------------------------------------------------------ model
def test_assignment_is_a_distribution():
    net = model.CommunityModel(input_dim=16, free_dim=8, hyper_dim=0, n_communities=5,
                               seed=1)
    x = torch.randn(7, 16)
    pi = net.assignment(x)
    assert pi.shape == (7, 5)
    assert torch.allclose(pi.sum(dim=1), torch.ones(7, dtype=pi.dtype), atol=1e-6)
    assert bool((pi >= 0).all())


def test_branch_scale_cannot_change_the_assignment():
    """The per-item standardisation makes the branch scale irrelevant."""
    net = model.CommunityModel(input_dim=16, free_dim=8, hyper_dim=0, n_communities=5,
                               seed=1)
    x = torch.randn(6, 16)
    before = net.assignment(x)
    with torch.no_grad():
        net.free_projection.weight.mul_(1000.0)
        net.free_centres.mul_(1000.0)
    after = net.assignment(x)
    assert torch.allclose(before, after, atol=1e-5)


def test_hyperbolic_branch_lands_inside_the_ball():
    net = model.CommunityModel(input_dim=16, free_dim=8, hyper_dim=8, n_communities=5,
                               curvature=1.0, seed=1)
    x = torch.randn(9, 16) * 5.0
    points = net.hyperbolic_branch(x)
    limit = (1.0 - net.contract.BALL_EPS) / np.sqrt(1.0)
    assert bool((points.norm(dim=1) <= limit + 1e-6).all())


def test_hyperbolic_branch_differs_from_the_free_branch():
    net = model.CommunityModel(input_dim=16, free_dim=8, hyper_dim=8, n_communities=5,
                               curvature=1.0, seed=1)
    x = torch.randn(5, 16)
    assert not torch.allclose(net.hyperbolic_branch(x), net.free_branch(x))


def test_combined_distance_averages_the_two_branches():
    net = model.CommunityModel(input_dim=16, free_dim=8, hyper_dim=8, n_communities=5,
                               curvature=1.0, seed=1)
    x = torch.randn(4, 16)
    free = model._standardise(torch.cdist(net.free_branch(x).double(),
                                           net.free_centres.double()))
    ball = net.hyperbolic_branch(x).double()
    centres = net.centres()[1].double()
    hyper = model._standardise(net.contract.distance(ball[:, None, :], centres[None, :, :],
                                                     1.0))
    expected = (free + hyper) / 2.0
    assert torch.allclose(net.distances(x), expected, atol=1e-6)


# ------------------------------------------------------------------------ losses
def test_ranking_loss_is_zero_when_the_margin_is_met():
    anchor = torch.tensor([[1.0, 0.0]])
    high = torch.tensor([[1.0, 0.0]])
    low = torch.tensor([[0.0, 1.0]])
    assert float(model.ranking_loss(anchor, high, low, margin=0.5)) == 0.0


def test_ranking_loss_is_positive_when_the_order_is_reversed():
    anchor = torch.tensor([[0.5, 0.5]])
    high = torch.tensor([[0.0, 1.0]])
    low = torch.tensor([[1.0, 0.0]])
    assert float(model.ranking_loss(anchor, high, low, margin=0.5)) > 0.0


def test_usage_loss_is_zero_for_a_uniform_usage_and_one_for_a_collapse():
    uniform = torch.full((64,), 1.0 / 64.0)
    collapsed = torch.zeros(64)
    collapsed[0] = 1.0
    assert float(model.usage_loss(uniform, 64)) == pytest.approx(0.0, abs=1e-6)
    assert float(model.usage_loss(collapsed, 64)) == pytest.approx(1.0, abs=1e-6)


def test_margin_is_the_inverse_of_the_community_count():
    assert model.margin_for(64) == pytest.approx(1.0 / 64.0)


def test_overlap_is_the_shared_mass():
    left = torch.tensor([[0.5, 0.5]])
    right = torch.tensor([[0.5, 0.5]])
    assert float(model.overlap(left, right)) == pytest.approx(0.5)


# ------------------------------------------------------------------------ data
def test_ranking_pairs_take_the_extreme_curvature_neighbours():
    """A hand-made graph: one anchor with a triangle edge and a leaf edge."""
    # 0-1-2 triangle with a leaf 3 hanging off node 2. The degrees are
    # [2, 2, 3, 1], so AF3(0,1) = 4 - 2 - 2 + 3 = 3, AF3(0,2) = 4 - 2 - 3 + 3 = 2
    # and AF3(2,3) = 4 - 3 - 1 = 0.
    rows = np.array([0, 0, 1, 2])
    cols = np.array([1, 2, 2, 3])
    values = np.ones(rows.size)
    adjacency = _undirected(rows, cols, 4)
    graph = data.Graph(split="test", nodes=[0, 1, 2, 3], index={i: i for i in range(4)},
                        adjacency=adjacency, edges=(rows, cols))
    curvature = data.af3_module().edge_curvatures(adjacency)
    pairs = data.ranking_pairs(graph, curvature)
    # nodes 0, 1 and 2 each have two different curvatures to compare; the leaf
    # has a single edge and carries no ranking
    assert len(pairs) == 3
    first = int(np.nonzero(pairs.anchor == 0)[0][0])
    assert pairs.high[first] == 1        # the edge with the higher AF3
    assert pairs.low[first] == 2         # the edge with the lower AF3
    assert pairs.af3_high[first] == 3.0
    assert pairs.af3_low[first] == 2.0
    leaf = int(np.nonzero(pairs.anchor == 2)[0][0])
    assert pairs.low[leaf] == 3          # for node 2 the leaf is the lowest


def test_ranking_pairs_drop_anchors_without_a_curvature_spread():
    # a single path 0-1-2: the middle anchor's two edges have the same AF3.
    rows = np.array([0, 1])
    cols = np.array([1, 2])
    adjacency = _undirected(rows, cols, 3)
    graph = data.Graph(split="test", nodes=[0, 1, 2], index={i: i for i in range(3)},
                        adjacency=adjacency, edges=(rows, cols))
    curvature = data.af3_module().edge_curvatures(adjacency)
    assert len(data.ranking_pairs(graph, curvature)) == 0


def test_the_training_graph_is_the_frozen_af3_authority():
    """The graph this experiment measures must be the one issue #136 measured."""
    from analysis_sid_recall.curvature_analysis import build_behaviour_graph

    nodes, index, adjacency, edges = build_behaviour_graph()
    graph = data.build_graph("train")
    assert list(graph.nodes) == list(nodes)
    # the authority lists each edge once, in one orientation; this graph lists
    # each undirected edge once, the way the AF3 pass reads them
    frozen = {(int(u), int(v)) for u, v in zip(edges[0].tolist(), edges[1].tolist())}
    mine = {(int(u), int(v)) for u, v in zip(graph.edges[0].tolist(), graph.edges[1].tolist())}
    assert mine == frozen


def test_the_behaviour_graph_is_the_binary_one_the_af3_convention_needs():
    """The convention is defined on an unweighted graph; a weighted one changes it."""
    graph = data.build_graph("train")
    values = graph.adjacency.data
    assert set(np.unique(values).tolist()) <= {1.0}
    degrees = np.asarray(graph.adjacency.sum(axis=1)).ravel()
    assert float(degrees.sum()) == 2.0 * graph.edges[0].size
    # the weighted builder stores one orientation per pair with a count, the
    # binary graph both orientations with a one: the same edges, the same count
    from analysis_sid_recall.curvature_analysis import build_behaviour_graph
    _, _, weighted, _ = build_behaviour_graph()
    assert weighted.nnz == graph.edges[0].size


# ------------------------------------------------------------------------ metrics
def test_transitivity_of_a_triangle_is_one_and_of_a_path_is_zero():
    triangle = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    value, triangles, wedges = metrics._transitivity(triangle)
    assert value == pytest.approx(1.0)
    assert triangles == 1 and wedges == 3
    path = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    value, triangles, wedges = metrics._transitivity(path)
    assert value == pytest.approx(0.0)


def test_separation_reads_the_anchor_anchored_difference():
    pi = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    pairs = data.RankingPairs(anchor=np.array([0]), high=np.array([1]), low=np.array([2]),
                              af3_high=np.array([3.0]), af3_low=np.array([1.0]))
    result = metrics.separation(pi, pairs, replicates=50)
    assert result["s_high"] == pytest.approx(0.82)
    assert result["s_low"] == pytest.approx(0.18)
    assert result["separation"] == pytest.approx(0.64)
    assert result["ci_low"] <= result["separation"] <= result["ci_high"]


def test_occupancy_counts_the_used_communities():
    pi = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = metrics.occupancy(pi)
    assert result["used_communities"] == 2
    assert result["largest_usage_share"] == pytest.approx(2.0 / 3.0)
    assert 1.0 < result["effective_communities"] <= 2.0


def test_curvature_gap_is_positive_when_the_communities_hold_the_triangles():
    """Two communities, the triangle inside one of them: the gap must be positive."""
    rows = np.array([0, 0, 1, 2, 3])
    cols = np.array([1, 2, 2, 3, 4])
    values = np.ones(rows.size)
    adjacency = _undirected(rows, cols, 5)
    graph = data.Graph(split="test", nodes=[0, 1, 2, 3, 4], index={i: i for i in range(5)},
                        adjacency=adjacency, edges=(rows, cols))
    curvature = data.af3_module().edge_curvatures(adjacency)
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    gap = metrics.curvature_gap(graph, curvature, labels)
    assert gap["status"] == "DEFINED"
    assert gap["within_mean"] > gap["cross_mean"]
    assert gap["gap"] > 0
