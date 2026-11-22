"""Tests for the GraphMoRE sectional-curvature proxy on small hand-checkable graphs.

The formula under test (GraphMoRE, AAAI 2025, Appendix A.1, Eq. A.1):

    kappa(m; b, c) = mean_a [ d(a,m)^2 + d(b,c)^2/4 - (d(a,b)^2 + d(a,c)^2)/2 ]

with ``d`` the unweighted shortest-path distance.  Every graph here is small enough
that the expected value is written out by hand, so a wrong sign, a wrong midpoint
test or an off-by-one in the distance is caught rather than absorbed by the data.

The issue is explicit that a tree or a ring must NOT be asserted to produce a
particular sign (C24): these tests check that the implementation equals the
formula, not that the formula produces the textbook sign.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sectional as sc  # noqa: E402


# ---------------------------------------------------------------- graph fixtures


def path_graph(n: int = 7) -> nx.Graph:
    return nx.path_graph(n)


def star_graph(leaves: int = 6) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from((0, leaf) for leaf in range(1, leaves + 1))
    return graph


def ring_graph(n: int = 8) -> nx.Graph:
    return nx.cycle_graph(n)


def grid_graph(rows: int = 4, cols: int = 4) -> nx.Graph:
    return nx.grid_2d_graph(rows, cols)


def k4() -> nx.Graph:
    return nx.complete_graph(4)


def two_components() -> nx.Graph:
    """A disjoint pair: one triangle plus an isolated node.  No cross distances."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    graph.add_node(9)
    return graph


# ---------------------------------------------------------------- distance layer


def test_distances_are_unweighted_shortest_paths():
    graph = path_graph(7)
    sqrt = sc.all_pairs_shortest(graph, sources=[0, 3])
    assert sqrt[0][0] == 0
    assert sqrt[0][6] == 6
    assert sqrt[3][3] == 0
    assert sqrt[3][0] == 3


def test_component_split_keeps_infinite_distances_out_of_the_sum():
    """C11: an unreachable reference must be excluded, never treated as 0."""
    graph = two_components()
    sqrt = sc.all_pairs_shortest(graph, sources=None)
    assert 9 not in sqrt[0], "the isolated node must not appear as a zero distance"
    reached = sc.reachable_distance_square(sqrt, 0, [1, 2, 9])
    assert reached == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------- the formula


def test_midpoint_requires_both_neighbours_and_distance_two():
    """C9: b and c are distinct neighbours of m with d(b,c) == 2."""
    graph = k4()
    triples = list(sc.central_triples(graph, center=0, geodesic_only=True))
    # In K4 every pair of distinct neighbours (b, c) is at distance 1, so no triple
    # of the geodesic kind exists -- the midpoint definition must reject all of them.
    assert triples == []
    loose = list(sc.central_triples(graph, center=0, geodesic_only=False))
    assert len(loose) == 3  # C(3,2) unordered neighbour pairs


def test_ring_has_one_geodesic_triple_per_centre():
    graph = ring_graph(8)
    triples = list(sc.central_triples(graph, center=0))
    # neighbours 1 and 7 are at distance 2 (the long way round is 6), and 0 is the
    # unique midpoint of that 2-path.
    assert triples == [(1, 7)]
    assert sc.is_geodesic_midpoint(graph, 0, 1, 7) is True


def test_path_graph_midpoint_is_geodesic():
    graph = path_graph(7)
    assert sc.is_geodesic_midpoint(graph, 3, 2, 4) is True
    # b and c at distance 2 but 1 is not the midpoint of 1-0-2? in a path 0-1-2 yes.
    assert sc.is_geodesic_midpoint(graph, 1, 0, 2) is True
    # neighbours of 3 are 2 and 4 only; 3 is their midpoint.
    assert list(sc.central_triples(graph, center=3)) == [(2, 4)]


def test_kappa_of_a_path_triple_matches_hand_computation():
    """A 7-node path, centre 3, b=2, c=4, all references reachable.

    d(b,c)^2/4 = 4/4 = 1.
    For a on the path the three squared distances are computed by hand below.
    """
    graph = path_graph(7)
    sqrt = sc.all_pairs_shortest(graph)
    kappa = sc.sectional_curvature(graph, sqrt, center=3, left=2, right=4)
    expected = []
    for a in range(7):
        da_m = abs(a - 3)
        da_b = abs(a - 2)
        da_c = abs(a - 4)
        expected.append(da_m ** 2 + 1.0 - (da_b ** 2 + da_c ** 2) / 2.0)
    assert kappa == pytest.approx(sum(expected) / len(expected))


def test_kappa_of_a_star_triple_matches_hand_computation():
    """Star with 6 leaves; centre 0 is the hub, b=1, c=2.

    d(b,c)=2 so d(b,c)^2/4 = 1.  a runs over the hub and the leaves.
    """
    graph = star_graph(6)
    sqrt = sc.all_pairs_shortest(graph)
    kappa = sc.sectional_curvature(graph, sqrt, center=0, left=1, right=2)
    expected = []
    for a in graph.nodes():
        da_m = 0 if a == 0 else 1
        da_b = 0 if a == 1 else (1 if a == 0 else 2)
        da_c = 0 if a == 2 else (1 if a == 0 else 2)
        expected.append(da_m ** 2 + 1.0 - (da_b ** 2 + da_c ** 2) / 2.0)
    assert kappa == pytest.approx(sum(expected) / len(expected))


def test_grid_centre_matches_brute_force_enumeration():
    """Euclidean lattice: compare against a literal transcription of Eq. A.1."""
    graph = grid_graph(4, 4)
    sqrt = sc.all_pairs_shortest(graph)
    centre = (1, 1)
    left, right = (0, 1), (1, 0)  # both neighbours of the centre, distance 2 apart
    assert sc.is_geodesic_midpoint(graph, centre, left, right)
    got = sc.sectional_curvature(graph, sqrt, centre, left, right)

    total, count = 0.0, 0
    for a in graph.nodes():
        da_m = abs(a[0] - centre[0]) + abs(a[1] - centre[1])
        da_b = abs(a[0] - left[0]) + abs(a[1] - left[1])
        da_c = abs(a[0] - right[0]) + abs(a[1] - right[1])
        dbc = 2
        total += da_m ** 2 + dbc ** 2 / 4.0 - (da_b ** 2 + da_c ** 2) / 2.0
        count += 1
    assert got == pytest.approx(total / count)


def test_kappa_is_normalised_over_the_common_reference_set():
    """C8: the mean runs over references reachable from b, c and m alike.

    The expected value is read from the graph's own distance table; hand-rolling the
    distances here is how the earlier version of this test went wrong (it mistook an
    adjacency for a zero distance).
    """
    graph = two_components()
    sqrt = sc.all_pairs_shortest(graph, sources=[0, 1, 2])
    kappa = sc.sectional_curvature(graph, sqrt, center=1, left=0, right=2)
    dbc = sqrt[0][2]
    expected = []
    for a in (0, 1, 2):
        expected.append(sqrt[1][a] ** 2 + dbc ** 2 / 4.0
                        - (sqrt[0][a] ** 2 + sqrt[2][a] ** 2) / 2.0)
    assert kappa == pytest.approx(sum(expected) / 3)
    # and the sum is positive here, which is a property of this triple, not of the code
    assert kappa > 0


def test_sampling_is_seeded_and_reports_its_coverage():
    """C12/C14: the sampler is deterministic per seed and states what it covered."""
    graph = grid_graph(6, 6)
    first = sc.sample_centres(graph, max_centres=10, seed=42)
    second = sc.sample_centres(graph, max_centres=10, seed=42)
    assert first == second
    assert len(first) == 10
    assert sc.sample_centres(graph, max_centres=None, seed=42) is None  # full coverage


def test_full_and_sampled_estimates_agree_on_a_regular_small_graph():
    """C16: on a small graph the sampled mean must reproduce the exact one closely.

    A cycle is used because its curvature is uniform, so a small sample cannot be
    fooled by node-to-node variation -- any difference then comes from the estimator.
    """
    graph = ring_graph(12)
    exact = sc.curvature_distribution(graph, max_centres=None, seed=42)
    coarse = sc.curvature_distribution(graph, max_centres=6, seed=42)
    assert coarse["coverage"] == pytest.approx(0.5)
    assert coarse["sampled"] is True
    assert exact["sampled"] is False
    assert coarse["raw_mean"] == pytest.approx(exact["raw_mean"])


def test_nested_samples_are_stable_on_a_random_graph():
    """C16: a larger sample of the same seed must stay near the smaller one."""
    graph = nx.random_regular_graph(3, 40, seed=7)
    small = sc.curvature_distribution(graph, max_centres=8, seed=42)
    large = sc.curvature_distribution(graph, max_centres=32, seed=42)
    exact = sc.curvature_distribution(graph, max_centres=None, seed=42)
    # Both estimates bracket the exact value within one sample's spread; the error of
    # the larger sample must not exceed that of the smaller one by more than its own
    # sampling noise allows, which is checked as a loose band rather than equality.
    for estimate in (small, large, exact):
        assert estimate["raw_mean"] is not None
    assert abs(large["raw_mean"] - exact["raw_mean"]) <= abs(small["raw_mean"] - exact["raw_mean"]) + 0.1


def test_curve_is_only_defined_where_the_triple_exists():
    """C19: a node with no qualifying triple is reported as such, never as 0.

    Node 0 has degree 3, so it owns at least one neighbour pair whose endpoints are
    two hops apart; nodes 3 and 4 are an isolated edge and can own none.
    """
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 5), (3, 4)])
    per_node = sc.per_node_curvature(graph, max_centres=None, seed=42)
    assert per_node[3]["n_triples"] == 0
    assert per_node[3]["mean"] is None
    assert per_node[4]["mean"] is None
    assert per_node[0]["n_triples"] >= 1
    assert per_node[0]["mean"] is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
