"""Independent implementation of GraphMoRE A.1 for issue #248.

Different mathematical path from the production code: the graph is built with networkx, distances
come from networkx BFS, and Eq. A.1 is summed term by term over the reference set. No import of
``tools.graph_infra.sectional_full``, no closed form, no precomputed ``S2``.

    from reference import evaluate
    values = evaluate(n, edges, triples, references)
"""
from fractions import Fraction
import networkx as nx
import numpy as np


def evaluate(n, edges, triples, references):
    """Mean of Eq. A.1 over the given references; NaN where no reference is reachable."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    values = []
    for m, b, c in triples:
        contributions = []
        for a in references:
            if m not in distances[a] or b not in distances[a] or c not in distances[a]:
                continue
            am = distances[a][m]; ab = distances[a][b]; ac = distances[a][c]
            bc = distances[b][c]
            contributions.append(Fraction(4 * am * am + bc * bc - 2 * ab * ab - 2 * ac * ac, 4))
        values.append(float(sum(contributions) / len(contributions)) if contributions else np.nan)
    return np.asarray(values)


def evaluate_float(n, edges, triples, references):
    """Same quantity in float64, for a second independent numeric path."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    values = []
    for m, b, c in triples:
        contributions = []
        for a in references:
            if m not in distances[a] or b not in distances[a] or c not in distances[a]:
                continue
            am = distances[a][m]; ab = distances[a][b]; ac = distances[a][c]
            bc = distances[b][c]
            contributions.append(am * am + bc * bc / 4.0 - (ab * ab + ac * ac) / 2.0)
        values.append(float(np.mean(contributions)) if contributions else np.nan)
    return np.asarray(values)
