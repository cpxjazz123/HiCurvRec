"""Independent paper A.1 evaluator using networkx BFS and rational arithmetic."""
from fractions import Fraction
import networkx as nx
import numpy as np


def evaluate(n, edges, triples, references):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    result = []
    for m, b, c in triples:
        values = []
        for a in references:
            if m not in distances[a] or b not in distances[a] or c not in distances[a]:
                continue
            am, ab, ac, bc = distances[a][m], distances[a][b], distances[a][c], distances[b][c]
            values.append(Fraction(4 * am * am + bc * bc - 2 * ab * ab - 2 * ac * ac, 4))
        result.append(float(sum(values) / len(values)) if values else np.nan)
    return np.asarray(result)
