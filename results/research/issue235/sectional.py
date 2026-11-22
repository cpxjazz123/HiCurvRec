"""GraphMoRE sectional-curvature proxy on an unweighted graph.

Formula (GraphMoRE, AAAI 2025, Appendix A.1, Eq. A.1):

    kappa(m; b, c) = mean_a [ d(a,m)^2 + d(b,c)^2/4 - (d(a,b)^2 + d(a,c)^2)/2 ]

``d`` is the unweighted shortest-path distance, ``b`` and ``c`` are distinct
neighbours of ``m``, and ``a`` runs over the reference set the three endpoints can
all reach.  The ``1/4`` is what makes the expression zero for a Euclidean lattice at
a lattice point, which is the sanity check this module is tested against.

Design notes that matter for reading the numbers later:

* **The mean is over references reachable from m, b and c alike.**  A node in another
  component contributes no term; treating its distance as 0 would silently bias every
  curvature toward ``d(b,c)^2/4``.
* **Two sampling modes.**  ``geodesic_only`` keeps only triples where ``d(b,c) == 2``
  and ``m`` sits on a shortest b-c path (the midpoint the paper analyses).  The looser
  mode keeps every neighbour pair, which is what a plain "sample two neighbours"
  reading of the paper produces; the two are always reported apart.
* **Nothing here infers direction.**  The graph is an undirected projection, so a
  curvature sign says something about the local neighbourhood, never about the
  direction of a user transition.

The import surface needs numpy and networkx only, so it runs in the Stage 2
interpreter without touching torch.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Iterable, Iterator, Sequence

import networkx as nx


def neighbours(graph: nx.Graph, node) -> list:
    return list(graph.neighbors(node))


def bfs_distances(graph: nx.Graph, source) -> dict:
    """Unweighted distances from ``source`` to every node in its component."""
    distances = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for nxt in graph.neighbors(current):
            if nxt not in distances:
                distances[nxt] = distances[current] + 1
                queue.append(nxt)
    return distances


def all_pairs_shortest(graph: nx.Graph, sources: Iterable | None = None) -> dict:
    """``{source: {target: distance}}`` restricted to each source's own component.

    Unreachable nodes are absent rather than infinite, which is what lets the caller
    drop them from the average instead of weighting them as zero distance.
    """
    chosen = list(graph.nodes()) if sources is None else list(sources)
    return {source: bfs_distances(graph, source) for source in chosen}


def reachable_distance_square(distances: dict, source, references: Sequence) -> list:
    """Squared distances to the references ``source`` can actually reach."""
    table = distances[source]
    return [float(table[ref]) ** 2 for ref in references if ref in table]


def is_geodesic_midpoint(graph: nx.Graph, centre, left, right) -> bool:
    """``centre`` lies on a shortest left-right path and ``d(left,right) == 2``."""
    if left == right or not graph.has_edge(centre, left) or not graph.has_edge(centre, right):
        return False
    if graph.has_edge(left, right):
        return False  # distance 1, so the centre cannot be a midpoint of length 2
    return True


def central_triples(graph: nx.Graph, center, geodesic_only: bool = True) -> Iterator[tuple]:
    """Unordered neighbour pairs ``(b, c)`` for one centre.

    ``geodesic_only`` keeps the paper's midpoint triples (``d(b,c) == 2`` with the
    centre on a shortest path).  The looser mode keeps **every** distinct neighbour
    pair, adjacent ones included -- that is the difference being reported, so
    filtering here would erase it.
    """
    arms = neighbours(graph, center)
    for first in range(len(arms)):
        for second in range(first + 1, len(arms)):
            left, right = arms[first], arms[second]
            if geodesic_only and not is_geodesic_midpoint(graph, center, left, right):
                continue
            yield (left, right)


def sectional_curvature(graph: nx.Graph, distances: dict, center, left, right) -> float:
    """Eq. A.1 for one triple, averaged over the commonly reachable references."""
    reference = [node for node in distances[center]
                 if node in distances[left] and node in distances[right]]
    if not reference:
        raise ValueError("the triple shares no reachable reference node")
    mid = distances[center]
    left_d = distances[left]
    right_d = distances[right]
    # d(b,c)^2: b and c are both one hop from the centre and not adjacent, so it is 2
    # for a geodesic triple, but read it from the table so non-geodesic triples are
    # not silently forced into the same number.
    pair_square = float(left_d[right]) ** 2 if right in left_d else 4.0
    total = 0.0
    for node in reference:
        da_m = mid[node]
        da_b = left_d[node]
        da_c = right_d[node]
        total += da_m * da_m + pair_square / 4.0 - (da_b * da_b + da_c * da_c) / 2.0
    return total / len(reference)


def sample_centres(graph: nx.Graph, max_centres: int | None, seed: int = 42):
    """Balanced centre sample, or ``None`` meaning "use every node"."""
    if max_centres is None:
        return None
    nodes = sorted(graph.nodes(), key=repr)
    if max_centres >= len(nodes):
        return None
    rng = random.Random(seed)
    picked = set(rng.sample(range(len(nodes)), max_centres))
    return [nodes[index] for index in sorted(picked)]


def curvature_distribution(graph: nx.Graph, max_centres: int | None = None,
                           seed: int = 42, geodesic_only: bool = True) -> dict:
    """Every qualifying triple in the sample, plus the coverage actually achieved."""
    sampled = sample_centres(graph, max_centres, seed)
    centres = list(graph.nodes()) if sampled is None else sampled
    distances = all_pairs_shortest(graph, sources=None)
    values: list[float] = []
    weighted: list[float] = []
    skipped_disconnected = 0
    for center in centres:
        per_centre = []
        for left, right in central_triples(graph, center, geodesic_only=geodesic_only):
            try:
                value = sectional_curvature(graph, distances, center, left, right)
            except ValueError:
                skipped_disconnected += 1
                continue
            values.append(value)
            per_centre.append(value)
        # triple-weighted view: a centre with many triples contributes each of them
        weighted.extend(per_centre)
    return {
        "values": values,
        "weighted_values": weighted,
        "centres": len(centres),
        "total_centres": graph.number_of_nodes(),
        "coverage": len(centres) / graph.number_of_nodes(),
        "skipped_disconnected": skipped_disconnected,
        "raw_mean": (sum(values) / len(values)) if values else None,
        "geodesic_only": geodesic_only,
        "sampled": sampled is not None,
    }


def per_node_curvature(graph: nx.Graph, max_centres: int | None = None,
                       seed: int = 42, geodesic_only: bool = True) -> dict:
    """Node-level statistics; a node with no qualifying triple gets ``mean: None``."""
    sampled = sample_centres(graph, max_centres, seed)
    centres = list(graph.nodes()) if sampled is None else sampled
    distances = all_pairs_shortest(graph, sources=None)
    out: dict = {}
    for center in centres:
        values = []
        for left, right in central_triples(graph, center, geodesic_only=geodesic_only):
            try:
                values.append(sectional_curvature(graph, distances, center, left, right))
            except ValueError:
                continue
        if not values:
            out[center] = {"n_triples": 0, "mean": None, "median": None,
                           "positive": 0, "negative": 0, "near_zero": 0}
            continue
        ordered = sorted(values)
        out[center] = {
            "n_triples": len(values),
            "mean": sum(values) / len(values),
            "median": ordered[len(ordered) // 2],
            "positive": sum(1 for v in values if v > 0),
            "negative": sum(1 for v in values if v < 0),
            "near_zero": sum(1 for v in values if v == 0),
        }
    return out
