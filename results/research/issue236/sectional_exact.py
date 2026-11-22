"""Exact GraphMoRE sectional curvature over all reference nodes.

Issue #235 averaged over 512 sampled references and recomputed the sum for every
triple.  With 24,556 references that is 40.6M x 24.5k distance lookups, so #236 uses
the linear identity that makes the formula separable:

    kappa(m;b,c) = S[m]/R + d(b,c)^2/4 - (S[b]/R + S[c]/R)/2

with ``S[v] = sum_a d(a,v)^2`` accumulated **once** per node over the shared reference
set and ``R`` the number of references reaching all three endpoints.  The dense
distance matrix is never materialised (C12), and the accumulators are int64 (C13)
because ``24556 * 24555^2`` overflows int32.

Two triple definitions are kept apart exactly as in #235 (C5/C6):

* ``geodesic``  -- ``d(b,c) == 2`` with the centre on a shortest path, which holds
  when b and c are neighbours of the centre and not adjacent to each other;
* ``neighbour`` -- any two distinct neighbours.

The import surface needs numpy and networkx only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class SquareSums:
    """``S[v] = sum_a d(a,v)^2`` over one shared reference set.

    ``square_sum`` is keyed by the graph's **own node labels** (which may be tuples,
    as in a lattice fixture), so a caller never has to reproduce the internal order.
    """

    references: list
    square_sum: dict = field(default_factory=dict)
    _counts: dict = field(default_factory=dict)

    def reference_count(self, node) -> int:
        return int(self._counts.get(node, 0))


def principal_component(graph: nx.Graph) -> list:
    """Nodes of the largest connected component, sorted by repr for stable order."""
    components = list(nx.connected_components(graph))
    if not components:
        return []
    biggest = max(components, key=len)
    return sorted(biggest, key=repr)


def accumulate_square_sums(graph: nx.Graph, references: list) -> SquareSums:
    """Stream one BFS per reference and accumulate squared distances.

    Only three arrays of length ``|V|`` are held: the int64 accumulator plus a small
    node->index map.  Unreachable nodes are skipped without being treated as 0.
    """
    # int64 per node: 24556 references x max 24555^2 exceeds int32 (tested).
    square_sum = {node: 0 for node in graph.nodes()}
    counts: dict = {}
    for reference in references:
        distances = nx.single_source_shortest_path_length(graph, reference)
        for node, distance in distances.items():
            square_sum[node] += int(distance) * int(distance)
            counts[node] = counts.get(node, 0) + 1
    return SquareSums(references=list(references), square_sum=square_sum, _counts=counts)


def _by_node(stream: SquareSums, graph: nx.Graph) -> dict:
    """``{node: (S[v], R_v)}`` keyed by the graph's own node labels."""
    return {node: (int(stream.square_sum[node]), stream.reference_count(node))
            for node in graph.nodes()}


def geodesic_triples(graph: nx.Graph, centre) -> list:
    """Neighbour pairs with ``d(b,c) == 2`` and the centre on the shortest path."""
    arms = list(graph.neighbors(centre))
    out = []
    for first in range(len(arms)):
        for second in range(first + 1, len(arms)):
            left, right = arms[first], arms[second]
            if not graph.has_edge(left, right):
                out.append((left, right))
    return out


def neighbour_triples(graph: nx.Graph, centre) -> list:
    """Every unordered pair of distinct neighbours, adjacent ones included."""
    arms = list(graph.neighbors(centre))
    return [(arms[first], arms[second])
            for first in range(len(arms)) for second in range(first + 1, len(arms))]


def triple_curvature(stream: SquareSums, graph: nx.Graph, centre, left, right):
    """Eq. A.1 via the separable form; ``None`` when no reference reaches all three."""
    table = _by_node(stream, graph)
    s_m, r_m = table[centre]
    s_b, r_b = table[left]
    s_c, r_c = table[right]
    # A reference that misses any endpoint cannot contribute to the mean.  When all
    # three see the same reference set (the usual case inside one component) the
    # shared denominator is just that count; otherwise the literal dense definition is
    # used for this triple so a partial overlap is never silently averaged over the
    # wrong denominator.
    if not (r_m == r_b == r_c):
        return _dense_triple(graph, centre, left, right, stream.references)
    shared = r_m
    if shared == 0:
        return None
    try:
        dbc = float(nx.shortest_path_length(graph, left, right))
    except nx.NetworkXNoPath:
        return None
    return (s_m / shared + dbc * dbc / 4.0 - (s_b / shared + s_c / shared) / 2.0)


def _dense_triple(graph: nx.Graph, centre, left, right, references: list):
    """Literal definition for triples whose endpoints see different reference sets."""
    usable = []
    for reference in references:
        distances = nx.single_source_shortest_path_length(graph, reference)
        if centre in distances and left in distances and right in distances:
            usable.append((distances[centre], distances[left], distances[right]))
    if not usable:
        return None
    dbc = nx.shortest_path_length(graph, left, right)
    total = 0.0
    for da_m, da_b, da_c in usable:
        total += da_m ** 2 + dbc ** 2 / 4.0 - (da_b ** 2 + da_c ** 2) / 2.0
    return total / len(usable)


def per_node_curvature(graph: nx.Graph, stream: SquareSums, mode: str = "geodesic") -> dict:
    """Node-level statistics; a node without a qualifying triple keeps ``mean: None``."""
    triples = geodesic_triples if mode == "geodesic" else neighbour_triples
    out = {}
    for centre in graph.nodes():
        values = []
        for left, right in triples(graph, centre):
            value = triple_curvature(stream, graph, centre, left, right)
            if value is not None:
                values.append(value)
        ordered = sorted(values)
        out[centre] = {
            "n_triples": len(values),
            "mean": (sum(values) / len(values)) if values else None,
            "median": ordered[len(ordered) // 2] if values else None,
        }
    return out
