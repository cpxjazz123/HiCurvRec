"""Does the quantised graph keep the behaviour graph's *topology*, and does the same product keep its role?

Three questions, each measured separately because they are not the same claim:

    distribution   do the two graphs contain the same local motifs at all?  (orbit totals, cosine / L1)
    organisation   are the same products in dense communities, the same core, the same hub set?
                   (Louvain agreement, k-core and clustering rank correlations, top-1% hub overlap)
    role of an item is a product that was a star centre still a centre -- in the top decile of degree,
                   and still sparse around itself -- rather than merely "a star exists somewhere"

Every metric is paired with the arm's recall in the summary step, and the summary also controls for the
arm's overall behaviour-edge retention, because that confound dominated the per-structure table.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"

spec = importlib.util.spec_from_file_location(
    "graphlet_tools", OUT / "graphlet_preservation.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
study = tools.study

ORCA = REPO / "results" / "research" / "issue213" / "orca" / "orca"
TOPO = OUT / "topology_graphs"
STAR_MIN_DEGREE = 10
STAR_MAX_CLUSTERING = 0.05


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    change = np.flatnonzero(np.diff(sorted_values) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(values)]]) - 1
    ranks = np.empty(len(values))
    ranks[order] = np.repeat((starts + ends) / 2.0, ends - starts + 1)
    return ranks


def spearman(first, second):
    a = average_ranks(first) - (len(first) - 1) / 2
    b = average_ranks(second) - (len(second) - 1) / 2
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def associated_rand(first, second):
    pairs = {}
    for a, b in zip(first, second):
        pairs[(int(a), int(b))] = pairs.get((int(a), int(b)), 0) + 1
    index = sum(value * (value - 1) / 2 for value in pairs.values())
    rows = sum(count * (count - 1) / 2 for count in np.bincount(np.asarray(first, dtype=int)))
    columns = sum(count * (count - 1) / 2 for count in np.bincount(np.asarray(second, dtype=int)))
    total = len(first) * (len(first) - 1) / 2
    expected = rows * columns / total if total else 0.0
    maximum = 0.5 * (rows + columns)
    return float((index - expected) / (maximum - expected)) if maximum != expected else 1.0


def normalised_mutual(first, second):
    pairs = {}
    for a, b in zip(first, second):
        pairs[(int(a), int(b))] = pairs.get((int(a), int(b)), 0) + 1
    n = len(first)
    a_counts = np.bincount(np.asarray(first, dtype=int))
    b_counts = np.bincount(np.asarray(second, dtype=int))
    value = 0.0
    for (a, b), count in pairs.items():
        p = count / n
        value += p * np.log(p / ((a_counts[a] / n) * (b_counts[b] / n)))
    pa = a_counts / n
    pb = b_counts / n
    ha = -float((pa * np.log(pa + 1e-12)).sum())
    hb = -float((pb * np.log(pb + 1e-12)).sum())
    return float(2 * value / (ha + hb)) if (ha + hb) else 1.0


def louvain(n, edges, seed=42):
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges.tolist())
    parts = nx.algorithms.community.louvain_communities(graph, seed=seed)
    labels = np.full(n, -1, dtype=np.int64)
    for index, block in enumerate(parts):
        for node in block:
            labels[int(node)] = index
    return labels


WORKER = {}


def _initialise(n, behaviour_edges, behaviour_set, behaviour_labels, orbit_b5, orbit_b4,
                degrees_b, clustering_b, core_b, star_centres):
    import os
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    WORKER.update(n=n, edges=behaviour_edges, edge_set=behaviour_set, labels=behaviour_labels,
                  orbit5=orbit_b5, orbit4=orbit_b4, degrees=degrees_b, clustering=clustering_b,
                  core=core_b, stars=star_centres)


def work(number):
    import networkx as nx
    n = WORKER["n"]
    arm = next(entry for entry in study.load_arms(67) if entry["number"] == number)
    tokens = study.tokens_of(arm["sid_path"], n)
    vectors, geometry = study.quantised_vectors(number, tokens, n)
    if vectors is None:
        return {"number": number, "reason": geometry.get("reason")}
    edges, graph = tools.budget_edges(vectors[:n], len(WORKER["edge_set"]), n)
    quantised = nx.Graph()
    quantised.add_nodes_from(range(n))
    quantised.add_edges_from(edges.tolist())
    degrees_q = np.array([quantised.degree(node) for node in range(n)], dtype=float)
    clustering_q = np.array([nx.clustering(quantised, node) for node in range(n)])
    core_map_q = nx.core_number(quantised)
    core_q = np.array([core_map_q.get(node, 0) for node in range(n)], dtype=float)
    labels_q = louvain(n, edges)

    # the orbit totals must come from *this* edge set, not from the earlier density-matched run:
    # mixing graphs would make the motif comparison describe one graph and the degree statistics another
    topo = TOPO / f"{number}.npz"
    if topo.exists():
        orbit_q5 = np.load(topo)["orbit5"]
    else:
        in_path = TOPO / f"{number}.in"
        out_path = TOPO / f"{number}.out"
        tools.tools.write_graph(in_path, n, edges)
        import subprocess
        subprocess.run([str(ORCA), "node", "5", str(in_path), str(out_path)], check=True,
                       capture_output=True)
        orbit_q5 = np.loadtxt(out_path, dtype=np.float64)
        np.savez(topo, orbit5=orbit_q5, edges=edges)
    total_b = WORKER["orbit5"].sum(axis=0)
    total_q = orbit_q5.sum(axis=0)
    cos = float(total_b @ total_q / (np.linalg.norm(total_b) * np.linalg.norm(total_q) + 1e-12))
    l1 = float(np.abs(total_b / total_b.sum() - total_q / total_q.sum()).sum())

    degree_rho = spearman(WORKER["degrees"], degrees_q)
    clustering_rho = spearman(WORKER["clustering"], clustering_q)
    core_rho = spearman(WORKER["core"], core_q)
    top = max(int(0.01 * n), 1)
    hubs_b = set(np.argsort(-WORKER["degrees"])[:top].tolist())
    hubs_q = set(np.argsort(-degrees_q)[:top].tolist())
    hub_jaccard = len(hubs_b & hubs_q) / len(hubs_b | hubs_q)

    centres = WORKER["stars"]
    centre_top = float(np.mean([degrees_q[node] >= np.quantile(degrees_q, 0.9) for node in centres]))
    centre_sparse = float(np.mean([clustering_q[node] < STAR_MAX_CLUSTERING for node in centres]))

    row = {"number": number, "recall": arm["recall"], "graph": graph,
           "orbit_total_cos": cos, "orbit_total_l1": l1,
           "community_ari": associated_rand(WORKER["labels"], labels_q),
           "community_nmi": normalised_mutual(WORKER["labels"], labels_q),
           "degree_rho": degree_rho, "clustering_rho": clustering_rho, "core_rho": core_rho,
           "hub_jaccard_1pct": float(hub_jaccard),
           "star_centre_top10": centre_top, "star_centre_sparse": centre_sparse,
           "star_centres": int(len(centres))}
    return row


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import networkx as nx
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    TOPO.mkdir(parents=True, exist_ok=True)
    n, adjacency, behaviour_edges = study.load_graph()
    behaviour_set = {(int(a), int(b)) if a < b else (int(b), int(a))
                     for a, b in behaviour_edges.tolist()}
    behaviour_graph = nx.Graph()
    behaviour_graph.add_nodes_from(range(n))
    behaviour_graph.add_edges_from(behaviour_edges.tolist())
    degrees_b = np.array([behaviour_graph.degree(node) for node in range(n)], dtype=float)
    clustering_b = np.array([nx.clustering(behaviour_graph, node) for node in range(n)])
    core_map = nx.core_number(behaviour_graph)
    core_b = np.array([core_map.get(node, 0) for node in range(n)], dtype=float)
    centres = [node for node in range(n) if degrees_b[node] >= STAR_MIN_DEGREE
               and clustering_b[node] < STAR_MAX_CLUSTERING]
    print(f"behaviour: {len(behaviour_set)} edges, {len(centres)} sparse hubs "
          f"(degree >= {STAR_MIN_DEGREE}, clustering < {STAR_MAX_CLUSTERING})", flush=True)
    labels_b = louvain(n, behaviour_edges)
    orbit_b5 = np.loadtxt(REPO / "results" / "research" / "issue213" / "orca" / "node5.out")
    orbit_b4 = np.loadtxt(REPO / "results" / "research" / "issue213" / "orca" / "node4.out")

    numbers = [arm["number"] for arm in study.load_arms(67)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialise,
                             initargs=(n, behaviour_edges, behaviour_set, labels_b, orbit_b5,
                                       orbit_b4, degrees_b, clustering_b, core_b, centres)) as pool:
        for future in as_completed({pool.submit(work, number): number for number in numbers}):
            row = future.result()
            rows.append(row)
            if "reason" in row:
                print(f"  {row['number']}: skipped ({row['reason']})", flush=True)
                continue
            print(f"  {row['number']} recall {row['recall']:.5f} | cos {row['orbit_total_cos']:.4f} "
                  f"| ARI {row['community_ari']:+.3f} | deg {row['degree_rho']:+.3f} "
                  f"| clust {row['clustering_rho']:+.3f} | core {row['core_rho']:+.3f} "
                  f"| hubJ {row['hub_jaccard_1pct']:.4f} | centre top10 {row['star_centre_top10']:.3f}",
                  flush=True)
    rows.sort(key=lambda row: row["number"])
    (OUT / "topology_preservation.json").write_text(json.dumps({"rows": rows}, indent=1, default=float))
    print(f"written {OUT / 'topology_preservation.json'} ({len(rows)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
