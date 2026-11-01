#!/usr/bin/env python3
"""Does the curvature of a SID's behaviour quotient graph predict its recall?

The question, and the three traps that come with it, are the ones the research
plan set out:

* the curvature of the quotient graph of the SID's prefixes, per level,
  standardised against a size-preserving random relabelling (a deeper level has
  more super-nodes, so the raw numbers are not comparable across levels);
* the basic SID-quality measures computed alongside it, because a SID whose
  prefixes already agree with the behaviour communities has a "nicer" curvature
  for reasons that have nothing to do with geometry;
* the arms are not independent (several come from one mechanism family), so the
  leave-one-family-out check is the honest one.

Everything here is train-only: the behaviour graph is built from the training
split's transitions, and no validation or test interaction enters a feature.

    python3 analysis_sid_recall/curvature_analysis.py

The exact Ollivier-Ricci curvature needs the transport problem, so it needs
``GraphRicciCurvature`` and ``POT``:

    python3 -m pip install --user POT GraphRicciCurvature

Without them the script still runs and reports the fast Lin-Lu-Yau curvature,
which is a lower bound of the Ollivier-Ricci one, and says so in the output.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis_sid_recall" / "curvature"
PAIRS = REPO / "analysis_sid_recall" / "sid_recall.csv"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"


# --------------------------------------------------------------- behaviour graph
def build_behaviour_graph() -> tuple:
    """The co-occurrence graph of the TRAINING split only.

    The construction is the one the hierarchy audit uses, so the two analyses
    speak about the same graph: a row's history plus its target is one training
    sequence, and consecutive items in it are adjacent.
    """
    import pandas as pd

    frame = pd.read_parquet(TRAIN, columns=["history", "target"])
    weights: dict[tuple[int, int], float] = {}
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first == second:
                continue
            key = (first, second) if first < second else (second, first)
            weights[key] = weights.get(key, 0.0) + 0.5
    nodes = sorted({item for edge in weights for item in edge})
    index = {item: position for position, item in enumerate(nodes)}
    rows, cols, data = [], [], []
    for (first, second), weight in weights.items():
        rows.append(index[first])
        cols.append(index[second])
        data.append(weight)
    adjacency = sp.coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    return nodes, index, adjacency


# ------------------------------------------------------------------- quotient
def quotient(adjacency: sp.csr_matrix, assignment: np.ndarray, n_groups: int) -> sp.csr_matrix:
    """The quotient graph: the super-nodes are the prefix groups."""
    incidence = sp.csr_matrix(
        (np.ones(assignment.shape[0]), (assignment, np.arange(assignment.shape[0]))),
        shape=(n_groups, assignment.shape[0]))
    merged = incidence @ adjacency @ incidence.T
    merged = merged.tocsr()
    merged.data = np.ones_like(merged.data)
    merged.setdiag(0)
    merged.eliminate_zeros()
    return merged


# ------------------------------------------------------------------ curvature
def _lazy_measures(graph: sp.csr_matrix, u: int, v: int, support: np.ndarray) -> tuple:
    """The lazy random walk measures of the two ends of an edge.

    ``P = 0.5 I + 0.5 A / deg``, so the measure from ``u`` puts half its mass on
    ``u`` itself and half on its neighbours.
    """
    degree = np.asarray(graph.sum(axis=1)).ravel()
    out = []
    for source, other in ((u, v), (v, u)):
        neighbours = graph.indices[graph.indptr[source]:graph.indptr[source + 1]]
        weight = np.zeros(support.shape[0])
        for position, node in enumerate(support):
            if node == source:
                weight[position] += 0.5
            elif node in neighbours:
                weight[position] += 0.5 / max(degree[source], 1e-12)
        out.append(weight)
    return out[0], out[1]


def ollivier_curvature(graph: sp.csr_matrix, sample: int, seed: int,
                       exact_distance: bool = False) -> float:
    """The mean Ollivier-Ricci curvature of the lazy random walk.

    For an edge ``(u, v)`` the curvature is ``1 - W1(m_u, m_v) / d(u, v)`` with
    ``d = 1`` for an edge, ``m_u`` the lazy walk measure of ``u`` and ``W1`` the
    1-Wasserstein distance on the graph's metric. The transport problem is
    solved exactly on the support of the two measures.

    ``exact_distance`` computes the support's distances with a breadth-first
    search instead of the 1/2-hop bound; it is affordable for the small
    quotients of the first level and serves as the cross-check of the bound.
    """
    import ot

    graph = graph.tocsr()
    if graph.nnz == 0:
        return float("nan")
    rows, cols = graph.nonzero()
    edges = list(zip(rows.tolist(), cols.tolist()))
    if len(edges) > sample:
        random.Random(seed).shuffle(edges)
        edges = edges[:sample]
    degree = np.asarray(graph.sum(axis=1)).ravel()
    values = []
    for u, v in edges:
        neighbours_u = graph.indices[graph.indptr[u]:graph.indptr[u + 1]]
        neighbours_v = graph.indices[graph.indptr[v]:graph.indptr[v + 1]]
        support = np.unique(np.concatenate(([u, v], neighbours_u, neighbours_v)))
        position = {int(node): index for index, node in enumerate(support)}
        size = support.shape[0]
        weight_u = np.zeros(size)
        weight_v = np.zeros(size)
        weight_u[position[u]] = 0.5
        weight_v[position[v]] = 0.5
        if neighbours_u.size and degree[u] > 0:
            weight_u[np.isin(support, neighbours_u)] += 0.5 / degree[u]
        if neighbours_v.size and degree[v] > 0:
            weight_v[np.isin(support, neighbours_v)] += 0.5 / degree[v]
        if weight_u.sum() <= 0 or weight_v.sum() <= 0:
            continue
        weight_u /= weight_u.sum()
        weight_v /= weight_v.sum()
        cost = np.full((size, size), 2.0)
        np.fill_diagonal(cost, 0.0)
        for node in support.tolist():
            index = position[node]
            neighbours = graph.indices[graph.indptr[node]:graph.indptr[node + 1]]
            for neighbour in neighbours.tolist():
                other = position.get(int(neighbour))
                if other is not None:
                    cost[index, other] = 1.0
        if exact_distance:
            cost = _exact_support_distances(graph, support, position, cost)
        try:
            transport = ot.emd2(weight_u, weight_v, cost)
        except Exception:
            continue
        if transport != transport:
            continue
        values.append(1.0 - float(transport))
    if not values:
        return float("nan")
    return float(np.mean(values))


def _exact_support_distances(graph: sp.csr_matrix, support: np.ndarray,
                             position: dict, bound: np.ndarray) -> np.ndarray:
    """The support's shortest-path distances, by a bounded breadth-first search.

    The 1/2-hop bound is a local approximation; this replaces it with the real
    distance between the support's nodes whenever the graph is small enough to
    search (the first level's quotient).
    """
    import collections

    adjacency = [set(graph.indices[graph.indptr[node]:graph.indptr[node + 1]].tolist())
                 for node in range(graph.shape[0])]
    out = bound.copy()
    for index, node in enumerate(support.tolist()):
        distance = {node: 0}
        frontier = collections.deque([node])
        depth = 0
        while frontier and depth < 4:
            depth += 1
            for _ in range(len(frontier)):
                current = frontier.popleft()
                for neighbour in adjacency[current]:
                    if neighbour not in distance:
                        distance[neighbour] = depth
                        frontier.append(neighbour)
        for other_index, other in enumerate(support.tolist()):
            value = distance.get(other)
            if value is not None:
                out[index, other_index] = float(value)
    return out


# ------------------------------------------------------------------- features
def sid_features(table: dict, adjacency: sp.csr_matrix, index: dict, nodes: list,
                 nulls: int, curvature_edges: int, seed: int) -> dict:
    """The curvature and the control features of one SID table."""
    width = len(next(iter(table.values())))
    levels = min(3, width)
    assignments = {}
    for level in range(1, levels + 1):
        groups: dict[tuple, list[int]] = {}
        for item, row in table.items():
            position = index.get(int(item))
            if position is None:
                continue
            groups.setdefault(tuple(row[:level]), []).append(position)
        keys = list(groups)
        assignment = np.full(len(nodes), -1, dtype=np.int64)
        for number, key in enumerate(keys):
            for position in groups[key]:
                assignment[position] = number
        assignments[level] = (keys, assignment)

    out: dict = {"levels": {}}
    for level in range(1, levels + 1):
        keys, assignment = assignments[level]
        n_groups = len(keys)
        sizes = np.array([int((assignment == number).sum()) for number in range(n_groups)])
        entry = {"prefix_groups": n_groups,
                 "group_size_max": int(sizes.max()) if sizes.size else 0,
                 "group_size_min": int(sizes.min()) if sizes.size else 0}
        graph = quotient(adjacency, assignment, n_groups)
        entry["quotient_edges"] = int(graph.nnz // 2)
        entry["curvature"] = ollivier_curvature(graph, curvature_edges, seed + level)
        if level == 1 and graph.shape[0] <= 1000:
            # the exact shortest-path variant is affordable only for the small
            # quotients; it is the cross-check of the 1/2-hop bound
            entry["curvature_exact_distance"] = ollivier_curvature(
                graph, min(curvature_edges, 200), seed + level, exact_distance=True)

        nulls_values: dict[str, list] = {"curvature": []}
        rng = np.random.default_rng(seed + 1000 * level)
        for _ in range(nulls):
            order = rng.permutation(len(nodes))
            shuffled = np.empty_like(assignment)
            cursor = 0
            for number in range(n_groups):
                size = int(sizes[number])
                shuffled[order[cursor:cursor + size]] = number
                cursor += size
            null_graph = quotient(adjacency, shuffled, n_groups)
            nulls_values["curvature"].append(
                ollivier_curvature(null_graph, curvature_edges, seed + 31 * level))
        for key in ("curvature",):
            values = [value for value in nulls_values[key] if value == value]
            real = entry[key]
            if not values or real != real:
                entry[key + "_z"] = None
                entry[key + "_null_mean"] = None
                continue
            mean = float(np.mean(values))
            std = float(np.std(values))
            entry[key + "_null_mean"] = mean
            entry[key + "_null_std"] = std
            entry[key + "_z"] = ((real - mean) / std) if std > 1e-12 else None
        out["levels"][str(level)] = entry

    try:
        import networkx as nx
        network = nx.from_scipy_sparse_array(adjacency)
        for level in range(1, levels + 1):
            keys, assignment = assignments[level]
            communities = []
            for number in range(len(keys)):
                members = np.nonzero(assignment == number)[0]
                if members.size:
                    communities.append({nodes[position] for position in members})
            out["levels"][str(level)]["modularity"] = (
                nx.community.modularity(network, communities) if communities else None)
    except Exception as error:                        # pragma: no cover - diagnostics
        out["modularity_error"] = str(error)

    rows = [tuple(row) for row in table.values()]
    out["codes_used"] = [len({row[level] for row in rows}) for level in range(levels)]
    triples = {row[:levels] for row in rows}
    out["collision"] = 1.0 - len(triples) / len(rows)
    out["items"] = len(rows)
    return out


# ----------------------------------------------------------------- statistics
def family_of(row: dict) -> str:
    """The mechanism family of a pair: the arm directory of its SID table.

    The collected copy under ``analysis_sid_recall/sids/`` carries the original
    path in its file name, so the arm is read from there and not from the
    directory the copy happens to sit in.
    """
    name = row["sid_file"].split("/")[-1]
    parts = name.replace(".json", "").split("__")
    for position, part in enumerate(parts):
        if part == "stage2_RQ-VAE" and position + 1 < len(parts):
            return parts[position + 1]
    return Path(row["sid_paths"].split(" | ")[0]).parts[-2]


def rank(values: list) -> list:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in order[position:end + 1]:
            result[index] = average
        position = end + 1
    return result


def spearman(x: list, y: list) -> float:
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else float("nan")


def residual(values: list, controls: list) -> list:
    """The part of ``values`` that the controls do not explain."""
    n = len(values)
    design = [[1.0] + [control[i] for control in controls] for i in range(n)]
    k = len(design[0])
    ata = [[sum(design[i][a] * design[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    aty = [sum(design[i][a] * values[i] for i in range(n)) for a in range(k)]
    matrix = [row[:] + [aty[index]] for index, row in enumerate(ata)]
    for column in range(k):
        pivot = max(range(column, k), key=lambda r: abs(matrix[r][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        if abs(matrix[column][column]) < 1e-12:
            continue
        for row in range(k):
            if row == column:
                continue
            factor = matrix[row][column] / matrix[column][column]
            for c in range(column, k + 1):
                matrix[row][c] -= factor * matrix[column][c]
    beta = [matrix[i][k] / matrix[i][i] if abs(matrix[i][i]) > 1e-12 else 0.0 for i in range(k)]
    return [values[i] - sum(beta[a] * design[i][a] for a in range(k)) for i in range(n)]


def partial_spearman(x: list, y: list, controls: list) -> float:
    """The rank correlation of x and y with the controls taken out."""
    rx, ry = rank(x), rank(y)
    if not controls:
        return spearman(rx, ry)
    rc = [rank([control[i] for control in controls]) for control in controls]
    return spearman(residual(rx, rc), residual(ry, rc))


def permutation_p(x: list, y: list, observed: float, trials: int = 2000, seed: int = 0) -> float:
    rng = random.Random(seed)
    shuffled = list(y)
    extreme = 0
    for _ in range(trials):
        rng.shuffle(shuffled)
        value = spearman(x, shuffled)
        if value == value and abs(value) >= abs(observed) - 1e-12:
            extreme += 1
    return (extreme + 1) / (trials + 1)


def bootstrap_ci(x: list, y: list, trials: int = 2000, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    n = len(x)
    values = []
    for _ in range(trials):
        indices = [rng.randrange(n) for _ in range(n)]
        value = spearman([x[i] for i in indices], [y[i] for i in indices])
        if value == value:
            values.append(value)
    if not values:
        return (None, None)
    values.sort()
    return (values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1])


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=str(PAIRS))
    parser.add_argument("--nulls", type=int, default=5)
    parser.add_argument("--curvature-edges", type=int, default=1500,
                        help="edges sampled per quotient graph")
    parser.add_argument("--limit", type=int, default=0, help="0 = every pair")
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = REPO / pairs_path
    rows = list(csv.DictReader(pairs_path.open()))
    if args.limit:
        rows = rows[:args.limit]

    nodes, index, adjacency = build_behaviour_graph()
    print(f"pairs: {len(rows)} | behaviour graph: {len(nodes)} items, "
          f"{int(adjacency.nnz // 2)} edges", flush=True)

    records = []
    for number, row in enumerate(rows):
        sid_path = REPO / row["sid_paths"].split(" | ")[0]
        if not sid_path.is_file():
            sid_path = REPO / "analysis_sid_recall" / "sids" / row["sid_file"].split("/")[-1]
        if not sid_path.is_file():
            print(f"  missing {row['variant']}", flush=True)
            continue
        table = json.loads(sid_path.read_text())
        features = sid_features(table, adjacency, index, nodes, nulls=args.nulls,
                                curvature_edges=args.curvature_edges,
                                seed=args.seed + 97 * number)
        family = family_of(row)
        records.append({"variant": row["variant"], "family": family,
                        "recall": float(row["test_recall@10"]),
                        "n_eval": int(row["n_eval"]) if row["n_eval"] else None,
                        "sid_path": str(sid_path.relative_to(REPO)),
                        "sid_sha256": row["sid_sha256"], **features})
        if (number + 1) % 5 == 0:
            print(f"  {number + 1}/{len(rows)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_arm_features.json").write_text(json.dumps(records, indent=2) + "\n",
                                                encoding="utf-8")

    recalls = [record["recall"] for record in records]

    def column(getter):
        values = []
        for record in records:
            value = getter(record)
            values.append(value if value is not None and value == value else None)
        return values

    curvature_features = {
        "z_curv_L1": lambda r: r["levels"]["1"]["curvature_z"],
        "z_curv_L2": lambda r: r["levels"]["2"]["curvature_z"],
        "z_curv_L3": lambda r: r["levels"]["3"]["curvature_z"],
        "z_curv_L1_minus_L3": lambda r: (r["levels"]["1"]["curvature_z"] - r["levels"]["3"]["curvature_z"]
                                         if r["levels"]["1"]["curvature_z"] is not None
                                         and r["levels"]["3"]["curvature_z"] is not None else None),
        "curvature_L1_exact_distance": lambda r: r["levels"]["1"].get("curvature_exact_distance"),
        "sign_change_L1_L3": lambda r: (1.0 if (r["levels"]["1"]["curvature_z"] or 0) > 0
                                        and (r["levels"]["3"]["curvature_z"] or 0) < 0 else 0.0),
    }
    controls = {
        "modularity_L1": lambda r: r["levels"]["1"].get("modularity"),
        "modularity_L2": lambda r: r["levels"]["2"].get("modularity"),
        "codes_used_L1": lambda r: r["codes_used"][0],
        "codes_used_L2": lambda r: r["codes_used"][1],
        "codes_used_L3": lambda r: r["codes_used"][2],
        "collision": lambda r: r["collision"],
    }

    def correlation(values):
        pairs = [(value, recall) for value, recall in zip(values, recalls) if value is not None]
        if len(pairs) < 5:
            return None
        x = [value for value, _ in pairs]
        y = [recall for _, recall in pairs]
        rho = spearman(x, y)
        return {"n": len(pairs), "spearman": rho,
                "permutation_p": permutation_p(x, y, rho, seed=args.seed),
                "bootstrap_ci95": bootstrap_ci(x, y, seed=args.seed)}

    summary = {"pairs": len(records), "nulls": args.nulls,
               "curvature_edges": args.curvature_edges, "seed": args.seed,
               "features": {}, "controls": {}, "partial": {}}
    for name, getter in curvature_features.items():
        summary["features"][name] = correlation(column(getter))
    for name, getter in controls.items():
        summary["controls"][name] = correlation(column(getter))

    for name, getter in curvature_features.items():
        values = column(getter)
        for control_name, control_getter in controls.items():
            control_values = column(control_getter)
            usable = [(v, c, recall) for v, c, recall in zip(values, control_values, recalls)
                      if v is not None and c is not None]
            if len(usable) < 8:
                continue
            summary["partial"].setdefault(name, {})[control_name] = {
                "n": len(usable),
                "partial_spearman": partial_spearman([u[0] for u in usable],
                                                    [u[2] for u in usable],
                                                    [[u[1]] for u in usable])}

    # the 182 tables are not 182 independent observations: they come from 115
    # mechanism families, and the biggest family alone contributes 20 tables.
    # The aggregated view keeps the best-recalling table of each family.
    best_per_family: dict[str, dict] = {}
    for record in records:
        current = best_per_family.get(record["family"])
        if current is None or record["recall"] > current["recall"]:
            best_per_family[record["family"]] = record
    aggregated = list(best_per_family.values())
    summary["families"] = len(aggregated)
    summary["aggregated_features"] = {}
    for name, getter in curvature_features.items():
        usable = [(getter(record), record["recall"]) for record in aggregated
                  if getter(record) is not None]
        if len(usable) < 5:
            continue
        x = [value for value, _ in usable]
        y = [recall for _, recall in usable]
        rho = spearman(x, y)
        summary["aggregated_features"][name] = {
            "n": len(usable), "spearman": rho,
            "permutation_p": permutation_p(x, y, rho, seed=args.seed),
            "bootstrap_ci95": bootstrap_ci(x, y, seed=args.seed)}

    families = sorted({record["family"] for record in records})
    values = column(curvature_features["z_curv_L1"])
    leave_one_out = {}
    for family in families:
        keep = [(value, recall) for record, value, recall
                in zip(records, values, recalls)
                if value is not None and record["family"] != family]
        if len(keep) < 8:
            continue
        leave_one_out[family] = {
            "n": len(keep),
            "spearman_z_curv_L1": spearman([v for v, _ in keep], [r for _, r in keep])}
    summary["leave_one_family_out"] = leave_one_out

    (OUT / "curvature_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(summary["features"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
