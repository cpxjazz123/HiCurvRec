"""The frozen local-structure descriptor, checked against hand-computed values.

Every number below is derived on a five-node graph whose degrees, triangles,
AF3 values and community split are written out in this file, so the descriptor
is compared with an oracle that does not reuse the implementation. The real
TRAIN graph is only used to check that the rebuild still reproduces the frozen
AF3 cache of issue #136 bit for bit; the descriptor arithmetic is measured on
the small graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import structure_table as table  # noqa: E402

# Graph: triangle 0-1-2, edge 2-3, edge 3-4. Nodes 3 and 4 are outside the
# triangle; the communities are {0,1,2} and {3,4}.
EDGES = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
LABELS = np.array([0, 0, 0, 1, 1], dtype=np.int64)


def _synthetic():
    adjacency = np.zeros((5, 5), dtype=np.int64)
    for first, second in EDGES:
        adjacency[first, second] = 1
        adjacency[second, first] = 1
    degree = adjacency.sum(1)
    # T_e = |N(u) & N(v)|, the number of triangles containing the edge.
    triangles_per_edge = {}
    for first, second in EDGES:
        common = np.intersect1d(np.nonzero(adjacency[first])[0],
                                np.nonzero(adjacency[second])[0])
        triangles_per_edge[(first, second)] = int(common.size)
        triangles_per_edge[(second, first)] = int(common.size)
    af3 = {(first, second): 4.0 - degree[first] - degree[second]
           + 3.0 * triangles_per_edge[(first, second)] for first, second in EDGES}
    for first, second in EDGES:
        af3[(second, first)] = af3[(first, second)]
    return adjacency, degree, triangles_per_edge, af3


def _descriptors():
    adjacency, degree, triangles_per_edge, af3 = _synthetic()
    return table.descriptor_matrix(
        adjacency=adjacency, degree=degree, triangles_per_edge=triangles_per_edge,
        af3=af3, community=LABELS, nodes_len=5)


def test_degree_and_triangle_counts_match_the_hand_computation():
    _, degree, triangles_per_edge, _ = _synthetic()
    raw, observed = _descriptors()
    # degrees: node 2 has three neighbours, nodes 0/1 two, nodes 3/4 one
    assert np.array_equal(degree, np.array([2, 2, 3, 2, 1]))
    assert np.isclose(raw[0, 0], np.log1p(2))
    assert np.isclose(raw[2, 0], np.log1p(3))
    # per-node triangle count is half the sum of T_e over incident edges:
    # node 0: (1 + 1) / 2 = 1, node 2: (1 + 1 + 0) / 2 = 1
    assert np.isclose(raw[0, 1], np.log1p(1.0))
    assert np.isclose(raw[2, 1], np.log1p(1.0))
    assert np.isclose(raw[3, 1], np.log1p(0.0))
    assert observed[0, 0] and observed[4, 0]


def test_af3_moments_are_the_incident_edge_moments():
    _, _, _, af3 = _synthetic()
    raw, _ = _descriptors()
    incident = np.array([af3[(0, 1)], af3[(0, 2)]], dtype=np.float64)
    assert np.isclose(raw[0, 2], incident.mean())
    assert np.isclose(raw[0, 3], np.quantile(incident, 0.9) - np.quantile(incident, 0.1))
    singleton = np.array([af3[(3, 4)]], dtype=np.float64)
    assert np.isclose(raw[4, 2], singleton.mean())
    # one edge gives a spread of exactly zero, which is measured, not missing
    assert np.isclose(raw[4, 3], 0.0)


def test_triangle_participation_shares_and_entropy():
    raw, observed = _descriptors()
    # node 2: edges to 0, 1 lie in a triangle, the edge to 3 does not -> 2/3
    assert np.isclose(raw[2, 4], 2.0 / 3.0)
    assert np.isclose(raw[2, 5], 1.0 / 3.0)
    # node 3: edges to 2 (no triangle) and 4 (no triangle) -> 0
    assert np.isclose(raw[3, 4], 0.0)
    # neighbour degrees of node 2 are 2, 2, 2: one bucket -> entropy 0
    assert np.isclose(raw[2, 6], 0.0)
    # neighbour degrees of node 1 are 2, 3 -> two buckets, normalised entropy 1
    assert np.isclose(raw[1, 6], 1.0)
    # entropy is undefined with a single neighbour, and recorded as such
    assert not observed[4, 6]
    assert np.isnan(raw[4, 6])


def test_cross_community_share_is_the_label_disagreement():
    raw, _ = _descriptors()
    # node 2 sits in community 0, its neighbours 0, 1 are community 0 and its
    # neighbour 3 is community 1 -> one of three edges crosses
    assert np.isclose(raw[2, 7], 1.0 / 3.0)
    # node 3 (label 1) has one neighbour in community 0 and one in its own
    assert np.isclose(raw[3, 7], 0.5)
    # node 4 (label 1) has a single neighbour, which shares its community
    assert np.isclose(raw[4, 7], 0.0)
    assert np.isclose(raw[0, 7], 0.0)


def test_absent_nodes_are_all_unavailable_and_never_zero_filled():
    raw, observed = _descriptors()
    # a node with no incident edge at all has no local structure
    raw_node = table._node_descriptor(0, np.zeros((5, 5), dtype=np.int64), {}, {}, LABELS)
    assert all(np.isnan(value) for value in raw_node)
    assert not observed[4, 6] and np.isnan(raw[4, 6])


def test_standardisation_is_the_train_mean_and_std_over_observed_entries():
    raw, observed = _descriptors()
    values, mean, std = table.standardise(raw, observed)
    for dim in range(raw.shape[1]):
        column = raw[observed[:, dim], dim]
        assert np.isclose(mean[dim], column.mean())
        assert np.isclose(std[dim], column.std())
        assert np.isclose(values[observed[:, dim], dim].mean(), 0.0, atol=1e-6)
        expected = (column - column.mean()) / column.std() if column.std() > 0 else 0.0
        assert np.allclose(values[observed[:, dim], dim],
                           np.broadcast_to(expected, column.shape), atol=1e-6)
    # an undefined entry is the population mean, and the mask keeps it apart
    assert np.all(values[~observed] == 0.0)


def test_descriptor_declares_eight_named_dimensions():
    assert table.DESCRIPTOR_DIMS == 8
    assert len(table.DESCRIPTOR_NAMES) == table.DESCRIPTOR_DIMS


def test_frozen_triangle_and_pair_sets_are_deterministic_and_typed():
    adjacency, degree, triangles_per_edge, af3 = _synthetic()
    first = table.frozen_sets(adjacency=adjacency, community=LABELS, seed=42,
                              triangle_cap=10, pair_cap=10)
    second = table.frozen_sets(adjacency=adjacency, community=LABELS, seed=42,
                               triangle_cap=10, pair_cap=10)
    assert np.array_equal(first["triangles"], second["triangles"])
    assert np.array_equal(first["same_pairs"], second["same_pairs"])
    assert np.array_equal(first["cross_pairs"], second["cross_pairs"])
    # exactly one triangle exists, (0, 1, 2), and one cross edge, (2, 3)
    assert np.array_equal(first["triangles"], np.array([[0, 1, 2]]))
    assert np.array_equal(first["cross_pairs"], np.array([[2, 3]]))
    # every same-group pair is inside one community and is not the same node
    for first_node, second_node in first["same_pairs"]:
        assert first_node != second_node
        assert LABELS[first_node] == LABELS[second_node]


def test_train_graph_rebuild_reproduces_the_frozen_af3_cache():
    root = TREE.parents[1]
    cache = root / "results" / "research" / "issue121" / "community_curvature" / "train_edge_af3.npz"
    if not cache.is_file():
        pytest.skip("the frozen AF3 cache of issue #136 is not present: NOT_RUN")
    graph = table.train_graph(root)
    frozen = np.load(cache)
    assert graph["nodes"].size == int(frozen["nodes"])
    assert graph["rows"].size == int(frozen["edges"])
    assert np.array_equal(graph["rows"], frozen["rows"].astype(np.int64))
    assert np.array_equal(graph["cols"], frozen["cols"].astype(np.int64))
    assert np.array_equal(graph["af3"], frozen["af3"].astype(np.float64))


def test_frozen_table_on_disk_is_pinned_by_its_manifest():
    root = TREE.parents[1]
    path = table.table_path(root)
    if not path.is_file():
        pytest.skip("the frozen structure table has not been built yet: NOT_RUN")
    loaded = table.load(path)
    manifest = table.verify(path)
    assert manifest["table_sha256"] == table.sha256_of(path)
    assert loaded["values"].shape[1] == table.DESCRIPTOR_DIMS
    assert loaded["values"].shape[0] == int(loaded["metadata"]["n_items"])
    assert loaded["triangles"].shape[1] == 3
    assert loaded["same_pairs"].shape[1] == 2
    assert loaded["cross_pairs"].shape[1] == 2
    # the gate reads the standardised matrix; the mask is what says which dims
    # were measured, and it must survive the round trip
    assert loaded["observed"].dtype == bool
    assert np.all(loaded["values"][~loaded["observed"]] == 0.0)
