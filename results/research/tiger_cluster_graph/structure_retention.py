"""Structure retention of TIGER pre- vs post-quantization cluster graphs.

For the same sampled behaviour-graph structures (K5, double triangle, K4+tail, K2,3, P5, C5,
star, T), measure the induced-edge share and exact induced-subgraph match in:

    G_z  KMeans(z) + within-cluster kNN
    G_q  KMeans(q) + within-cluster kNN

The two graphs use the same clustering rule and the same edge budget.  The behaviour graph is the
answer key, never an input to the graph construction.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/structure_retention.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPHS = HERE / "cluster_graphs.npz"
OUT = HERE / "structure_retention.json"
MD = HERE / "structure_retention.md"
ANALYSIS = REPO / "analysis_sid_recall" / "analysis.py"

MAX_INSTANCES = 500
SEED = 42


def edges_of(array: np.ndarray) -> list[tuple[int, int]]:
    return [(int(left), int(right)) for left, right in array.tolist()]


def graph_from_edges(nodes: int, edges: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(edges_of(edges))
    return graph


def induced(graph: nx.Graph, nodes: list[int]) -> set[tuple[int, int]]:
    found = set()
    for first in range(len(nodes)):
        for second in range(first + 1, len(nodes)):
            left, right = nodes[first], nodes[second]
            if graph.has_edge(left, right):
                found.add((min(left, right), max(left, right)))
    return found


def main() -> int:
    payload = np.load(GRAPHS)
    behaviour_edges = payload["behaviour"]
    z_edges = payload["z_edges"]
    q_edges = payload["q_edges"]
    nodes = int(max(behaviour_edges.max(), z_edges.max(), q_edges.max())) + 1
    behaviour = graph_from_edges(nodes, behaviour_edges)
    graph_z = graph_from_edges(nodes, z_edges)
    graph_q = graph_from_edges(nodes, q_edges)

    spec = importlib.util.spec_from_file_location("sid_analysis", ANALYSIS)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    adjacency = [set(behaviour.neighbors(node)) for node in range(nodes)]
    print("enumerating cliques", flush=True)
    cliques4 = analysis.enumerate_cliques(adjacency, 4)
    cliques5 = analysis.enumerate_cliques(adjacency, 5)
    print(f"cliques: 4={len(cliques4)} 5={len(cliques5)}", flush=True)
    rng = np.random.default_rng(SEED)
    patterns = analysis.build_patterns(adjacency, rng, cliques4, cliques5, MAX_INSTANCES)

    retention = {}
    for name, instances in patterns.items():
        if not instances:
            continue
        exact_z = exact_q = 0
        share_z = share_q = 0.0
        count = 0
        for instance in instances:
            members = [int(node) for node in instance["nodes"]]
            truth = induced(behaviour, members)
            if not truth:
                continue
            z_found = induced(graph_z, members)
            q_found = induced(graph_q, members)
            share_z += len(truth & z_found) / len(truth)
            share_q += len(truth & q_found) / len(truth)
            exact_z += int(z_found == truth)
            exact_q += int(q_found == truth)
            count += 1
        if count:
            retention[name] = {
                "instances": count,
                "z_edge_share": share_z / count,
                "q_edge_share": share_q / count,
                "z_exact": exact_z / count,
                "q_exact": exact_q / count,
            }
        print(name, retention.get(name), flush=True)

    def clustering(graph: nx.Graph) -> np.ndarray:
        values = nx.clustering(graph)
        return np.array([values[node] for node in range(nodes)], dtype=np.float64)

    def core(graph: nx.Graph) -> np.ndarray:
        values = nx.core_number(graph)
        return np.array([values[node] for node in range(nodes)], dtype=np.float64)

    degree_b = np.array([behaviour.degree(node) for node in range(nodes)], dtype=np.float64)
    degree_z = np.array([graph_z.degree(node) for node in range(nodes)], dtype=np.float64)
    degree_q = np.array([graph_q.degree(node) for node in range(nodes)], dtype=np.float64)
    clustering_b, clustering_z, clustering_q = clustering(behaviour), clustering(graph_z), clustering(graph_q)
    core_b, core_z, core_q = core(behaviour), core(graph_z), core(graph_q)

    result = {
        "instances_requested": MAX_INSTANCES,
        "seed": SEED,
        "patterns": retention,
        "graph_rank": {
            "z_degree": float(spearmanr(degree_b, degree_z).statistic),
            "q_degree": float(spearmanr(degree_b, degree_q).statistic),
            "z_clustering": float(spearmanr(clustering_b, clustering_z).statistic),
            "q_clustering": float(spearmanr(clustering_b, clustering_q).statistic),
            "z_core": float(spearmanr(core_b, core_z).statistic),
            "q_core": float(spearmanr(core_b, core_q).statistic),
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# TIGER cluster-graph structure retention\n"]
    lines.append("| structure | instances | z edge share | q edge share | z exact | q exact |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for name, row in retention.items():
        lines.append(
            f"| {name} | {row['instances']} | {row['z_edge_share']:.4f} | {row['q_edge_share']:.4f} | "
            f"{row['z_exact']:.4f} | {row['q_exact']:.4f} |\n"
        )
    lines.append("\n## Graph-level rank correlation with behaviour\n\n")
    lines.append("| graph | degree rho | clustering rho | core rho |\n")
    lines.append("|---|---:|---:|---:|\n")
    rank = result["graph_rank"]
    lines.append(f"| z | {rank['z_degree']:.4f} | {rank['z_clustering']:.4f} | {rank['z_core']:.4f} |\n")
    lines.append(f"| q | {rank['q_degree']:.4f} | {rank['q_clustering']:.4f} | {rank['q_core']:.4f} |\n")
    MD.write_text("".join(lines) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
