"""Tests for the streaming square-distance-sum trick used by issue #236.

Issue #235 averaged over 512 sampled reference nodes and recomputed the whole sum per
triple.  Issue #236 needs all 24,556 references, so the sum is accumulated once per
node and the curvature becomes a linear combination:

    kappa(m;b,c) = S[m]/R + d(b,c)^2/4 - (S[b]/R + S[c]/R)/2

where ``S[v] = sum_a d(a,v)^2`` over the **same** reference set and ``R`` is the
number of references that reach all three endpoints.

That equivalence is the thing under test: a wrong denominator, a dropped unreachable
reference or an int64 overflow would silently move every number in the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sectional_exact as sx  # noqa: E402


# ------------------------------------------------------------------ fixtures


def ring(n: int = 8) -> nx.Graph:
    return nx.cycle_graph(n)


def grid(rows: int = 4, cols: int = 4) -> nx.Graph:
    return nx.grid_2d_graph(rows, cols)


def star(leaves: int = 6) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from((0, leaf) for leaf in range(1, leaves + 1))
    return g


def path(n: int = 7) -> nx.Graph:
    return nx.path_graph(n)


def split_graph() -> nx.Graph:
    """A 5-node principal component plus an isolated edge: no cross distances.

    Node 0 has neighbours 1, 2 and 5 with 5 not adjacent to 1 or 2, so it owns
    geodesic triples -- the triangle alone would not (all its pairs are adjacent).
    """
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 5), (3, 4)])
    return g


def _positions(graph: nx.Graph) -> dict:
    return {node: index for index, node in enumerate(sorted(graph.nodes(), key=repr))}


def _dense_square_sums(graph: nx.Graph, references: list) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation: one dense BFS per reference, no tricks."""
    nodes = sorted(graph.nodes(), key=repr)
    index = {node: i for i, node in enumerate(nodes)}
    reach = np.zeros((len(references), len(nodes)), dtype=bool)
    squares = np.zeros((len(references), len(nodes)), dtype=np.int64)
    for row, ref in enumerate(references):
        distances = nx.single_source_shortest_path_length(graph, ref)
        for node, distance in distances.items():
            column = index[node]
            reach[row, column] = True
            squares[row, column] = int(distance) ** 2
    return reach, squares


def _brute_kappa(graph: nx.Graph, centre, left, right, references: list) -> float:
    """Literal Eq. A.1: loop over the references that reach all three endpoints."""
    usable = []
    for ref in references:
        d_ref = nx.single_source_shortest_path_length(graph, ref)
        if centre in d_ref and left in d_ref and right in d_ref:
            usable.append((d_ref[centre], d_ref[left], d_ref[right]))
    if not usable:
        return None
    dbc = nx.shortest_path_length(graph, left, right)
    total = 0.0
    for da_m, da_b, da_c in usable:
        total += da_m ** 2 + dbc ** 2 / 4.0 - (da_b ** 2 + da_c ** 2) / 2.0
    return total / len(usable)


# ------------------------------------------------------------------ equivalence


@pytest.mark.parametrize("maker", [ring, grid, star, path, split_graph])
def test_streaming_sums_match_a_dense_reference_implementation(maker):
    """The accumulated S[v] must equal the column sum of the dense square matrix."""
    graph = maker()
    references = sorted(graph.nodes(), key=repr)  # all references, no sampling
    reach, squares = _dense_square_sums(graph, references)
    stream = sx.accumulate_square_sums(graph, references)
    order = sorted(graph.nodes(), key=repr)
    for column, node in enumerate(order):
        assert stream.square_sum[node] == int(squares[:, column].sum())
        assert stream.reference_count(node) == int(reach[:, column].sum())


@pytest.mark.parametrize("maker", [ring, grid, star, path, split_graph])
def test_streaming_kappa_equals_brute_force(maker):
    """Every geodesic triple's curvature must match the literal loop over references."""
    graph = maker()
    references = sorted(graph.nodes(), key=repr)
    stream = sx.accumulate_square_sums(graph, references)
    checked = 0
    for centre in graph.nodes():
        for left, right in sx.geodesic_triples(graph, centre):
            got = sx.triple_curvature(stream, graph, centre, left, right)
            want = _brute_kappa(graph, centre, left, right, references)
            if want is None:
                assert got is None
                continue
            assert got == pytest.approx(want, rel=1e-12, abs=1e-12)
            checked += 1
    assert checked > 0, "the fixture produced no geodesic triple"


def test_denominator_is_the_shared_reference_count_not_the_total():
    """C9: a triple reached by fewer references must divide by that smaller number."""
    graph = split_graph()
    references = [0, 1, 2, 3, 4]  # 3 and 4 are in the other component
    stream = sx.accumulate_square_sums(graph, references)
    assert stream.reference_count(0) == 3
    assert stream.reference_count(3) == 2
    # a triple around node 0 can only use the three references in its own component
    value = sx.triple_curvature(stream, graph, 1, 0, 2)
    assert value == pytest.approx(_brute_kappa(graph, 1, 0, 2, references))


def test_unreachable_references_are_excluded_not_zeroed():
    """C14: treating an unreachable d(a,v) as 0 would drag every sum down."""
    graph = split_graph()
    stream = sx.accumulate_square_sums(graph, [0, 1, 2, 3, 4])
    # node 3 sees only references 3 and 4, both at squared distance 0 and 1
    assert stream.square_sum[3] == 0 + 1
    assert stream.reference_count(3) == 2


# ------------------------------------------------------------------ edge cases


def test_isolated_node_has_no_triple_and_no_zero_fill():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 5), (3, 4)])
    stream = sx.accumulate_square_sums(graph, sorted(graph.nodes(), key=repr))
    assert list(sx.geodesic_triples(graph, 3)) == []
    assert sx.per_node_curvature(graph, stream)[3]["n_triples"] == 0
    assert sx.per_node_curvature(graph, stream)[3]["mean"] is None


def test_non_geodesic_neighbour_pair_is_excluded_from_the_main_mode():
    """C5: adjacent neighbours are not a midpoint triple."""
    graph = nx.complete_graph(4)
    assert list(sx.geodesic_triples(graph, 0)) == []
    assert len(list(sx.neighbour_triples(graph, 0))) == 3


def test_integer_accumulator_cannot_overflow_at_the_reported_scale():
    """C13: sum of d^2 over 24,556 references, d <= 24,555, must fit in int64."""
    worst = 24556 * (24555 ** 2)
    assert worst < np.iinfo(np.int64).max
    assert worst > np.iinfo(np.int32).max  # int32 would overflow, so int64 is required


def test_accumulator_is_exact_integer_and_result_is_float64():
    """C13: the sum must be an exact integer (dict of Python ints, no numpy overflow)."""
    graph = ring(8)
    stream = sx.accumulate_square_sums(graph, sorted(graph.nodes(), key=repr))
    assert isinstance(stream.square_sum[0], int)
    assert stream.square_sum[0] == sum(
        nx.single_source_shortest_path_length(graph, a)[0] ** 2 for a in graph.nodes())
    value = sx.triple_curvature(stream, graph, 0, 1, 7)
    assert isinstance(value, float)


def test_ring_curvature_matches_hand_computation():
    """A ring with all references: the value is checked against an explicit sum."""
    graph = ring(8)
    references = sorted(graph.nodes(), key=repr)
    stream = sx.accumulate_square_sums(graph, references)
    assert sx.geodesic_triples(graph, 0) == [(1, 7)]
    got = sx.triple_curvature(stream, graph, 0, 1, 7)
    total = 0.0
    for a in references:
        d = nx.single_source_shortest_path_length(graph, a)
        total += d[0] ** 2 + 1.0 - (d[1] ** 2 + d[7] ** 2) / 2.0
    assert got == pytest.approx(total / len(references))


def test_reference_set_is_the_whole_component():
    """C8/C23: the reference list is every node of the principal component."""
    graph = split_graph()
    principal = sx.principal_component(graph)
    assert principal == [0, 1, 2, 5]
    stream = sx.accumulate_square_sums(graph, principal)
    assert stream.references == principal


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
