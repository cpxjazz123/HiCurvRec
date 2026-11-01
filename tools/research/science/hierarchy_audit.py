"""Is there a trustworthy product hierarchy in the training behaviour?

The HGS mechanism needs a tree to supervise against. The repository has no Amazon
category metadata (``dataset/`` is empty), and the behaviour graph's four-point
tree-ness was already found to be close to a random reconnect graph, so a
candidate hierarchy built from behaviour must EARN its use:

* it is built from the TRAINING split only (transitions, symmetrised, weighted by
  the transition count);
* it is checked against a DEGREE-PRESERVING random reconnect of the same graph,
  built the same way (same algorithm, same resolutions, same seed);
* the comparison uses the INDEPENDENT validation pairs, which took no part in
  building either structure.

Two measurements per level:

``co_membership``
    the fraction of validation behaviour pairs whose two items fall in the same
    cell of that level. A hierarchy that carries real structure puts more of them
    together than a random graph with the same degrees does.

``tree_ness``
    the project's four-point statistic on the hierarchy-induced tree distance:
    for quadruples of items, the largest and the middle of the three pair sums of
    the tree distances. A tree gives 0 (the two largest sums are equal); a space
    that is not a tree gives more.

The verdict is written to ``results/research/hierarchy_audit/report.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
VALID_PAIRS = REPO / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_valid_s42.npz"
OUT = REPO / "results" / "research" / "hierarchy_audit"

#: One resolution per level, coarse to fine. The levels are NESTED: each level is
#: a refinement of the level above it, which is what makes the result a tree.
LEVELS = (1.0, 4.0, 8.0)
SEED = 42
#: How many degree-preserving reconnects the control averages over.
CONTROLS = 3
#: Quadruples used for the four-point statistic.
QUADS = 20000


def build_graph() -> nx.Graph:
    frame = pd.read_parquet(TRAIN, columns=["history", "target"])
    weight: dict[tuple[int, int], float] = {}
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first == second:
                continue
            key = (first, second) if first < second else (second, first)
            weight[key] = weight.get(key, 0.0) + 0.5
    graph = nx.Graph()
    for (first, second), value in weight.items():
        graph.add_edge(first, second, weight=float(value))
    return graph


def nested_levels(graph: nx.Graph) -> list[dict[int, int]]:
    """A nested hierarchy: each level refines the one above it."""
    levels: list[dict[int, int]] = []
    assignment: dict[int, int] = {node: 0 for node in graph.nodes()}
    levels.append(dict(assignment))
    for resolution in LEVELS:
        refined: dict[int, int] = {}
        groups: dict[int, list[int]] = {}
        for node, cell in assignment.items():
            groups.setdefault(cell, []).append(node)
        offset = 0
        for cell in sorted(groups):
            members = groups[cell]
            if len(members) < 4:
                for node in members:
                    refined[node] = offset
                offset += 1
                continue
            subgraph = graph.subgraph(members)
            communities = nx.algorithms.community.louvain_communities(
                subgraph, resolution=resolution, seed=SEED)
            for community in communities:
                for node in community:
                    refined[node] = offset
                offset += 1
        assignment = refined
        levels.append(dict(assignment))
    return levels


def co_membership(levels: list[dict[int, int]], anchors, positives) -> list[float]:
    out = []
    for level in levels:
        same = sum(1 for a, b in zip(anchors, positives) if level.get(a) == level.get(b))
        out.append(same / max(len(anchors), 1))
    return out


def tree_ness(levels: list[dict[int, int]], nodes, rng) -> float:
    """The four-point statistic of the hierarchy-induced tree distance."""
    import itertools
    depth = len(levels) - 1
    index = {node: position for position, node in enumerate(nodes)}

    def tree_distance(first: int, second: int) -> int:
        # the number of levels at which the two items still share a cell
        shared = 0
        for level in reversed(levels):
            if level.get(first) == level.get(second):
                shared += 1
            else:
                break
        return depth - shared

    values = []
    sampled = rng.choice(len(nodes), size=(QUADS, 4), replace=True)
    for row in sampled:
        a, b, c, p = (nodes[int(value)] for value in row)
        sums = sorted([tree_distance(a, b) + tree_distance(c, p),
                       tree_distance(a, c) + tree_distance(b, p),
                       tree_distance(a, p) + tree_distance(b, c)])
        values.append(0.5 * (sums[2] - sums[1]))
    del index
    return float(np.mean(values))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    graph = build_graph()
    pairs = np.load(VALID_PAIRS, allow_pickle=True)
    anchors = pairs["anchors"].tolist()
    rng = np.random.default_rng(SEED)

    real_levels = nested_levels(graph)
    real_cells = [len(set(level.values())) for level in real_levels]
    real_co = co_membership(real_levels, anchors, pairs["positives"].tolist())
    nodes = sorted(graph.nodes())
    sample_nodes = [nodes[int(i)] for i in rng.choice(len(nodes), size=min(4000, len(nodes)),
                                                      replace=False)]
    real_tree = tree_ness(real_levels, sample_nodes, np.random.default_rng(SEED))

    control_co = []
    control_cells = []
    control_tree = []
    for trial in range(CONTROLS):
        shuffled = graph.copy()
        swaps = min(3 * shuffled.number_of_edges(), 300_000)
        nx.double_edge_swap(shuffled, nswap=swaps, max_tries=swaps * 10,
                            seed=SEED + trial)
        levels = nested_levels(shuffled)
        control_cells.append([len(set(level.values())) for level in levels])
        control_co.append(co_membership(levels, anchors, pairs["positives"].tolist()))
        control_tree.append(tree_ness(levels, sample_nodes, np.random.default_rng(SEED)))

    control_co = np.asarray(control_co)
    report = {
        "graph": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
                  "source": "train.parquet transitions, symmetrised"},
        "levels": [{"resolution": LEVELS[i], "cells": real_cells[i + 1],
                    "co_membership": real_co[i + 1],
                    "control_co_membership": {"mean": float(control_co[:, i + 1].mean()),
                                              "std": float(control_co[:, i + 1].std())},
                    "z": float((real_co[i + 1] - control_co[:, i + 1].mean())
                               / max(control_co[:, i + 1].std(), 1e-9))}
                   for i in range(len(LEVELS))],
        "root": {"cells": real_cells[0], "co_membership": real_co[0]},
        "tree_ness": {"real": real_tree,
                      "control_mean": float(np.mean(control_tree)),
                      "control_std": float(np.std(control_tree))},
        "validation_pairs": len(anchors),
        "controls": CONTROLS,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # The frozen hierarchy itself, in the item-row order the trainer uses: one
    # integer cell per level, -1 for an item the training graph never mentions.
    items = sorted(int(key) for key in json.loads(
        (REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json").read_text()))
    table = np.full((len(items), len(real_levels)), -1, dtype=np.int32)
    for row, item in enumerate(items):
        for level_index, level in enumerate(real_levels):
            table[row, level_index] = int(level.get(item, -1))
    np.savez_compressed(OUT / "hierarchy.npz", items=np.asarray(items, dtype=np.int64),
                        cells=table,
                        levels=np.asarray([len(set(level.values())) for level in real_levels]))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
