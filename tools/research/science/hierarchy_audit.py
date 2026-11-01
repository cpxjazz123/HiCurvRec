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

import hashlib
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


def nesting_breaks(levels: list[dict[int, int]]) -> int:
    """Pairs of levels where a finer cell is split across two coarser parents.

    The trainer treats the table as a tree (top-down longest common prefix), so a
    cell at level l+1 that appears under two different cells at level l would
    silently make ``T`` disagree with the behaviour graph. Counted, not assumed.
    """
    breaks = 0
    for coarse, fine in zip(levels, levels[1:]):
        seen: dict[int, int] = {}
        for item, cell in fine.items():
            parent = coarse.get(item)
            if parent is None:
                continue
            previous = seen.setdefault(cell, parent)
            if previous != parent:
                breaks += 1
    return breaks


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    graph = build_graph()
    pairs = np.load(VALID_PAIRS, allow_pickle=True)
    anchors = pairs["anchors"].tolist()
    rng = np.random.default_rng(SEED)

    real_levels = nested_levels(graph)
    real_cells = [len(set(level.values())) for level in real_levels]
    real_co = co_membership(real_levels, anchors, pairs["positives"].tolist())
    del rng

    control_co = []
    control_cells = []
    for trial in range(CONTROLS):
        shuffled = graph.copy()
        swaps = min(3 * shuffled.number_of_edges(), 300_000)
        nx.double_edge_swap(shuffled, nswap=swaps, max_tries=swaps * 10,
                            seed=SEED + trial)
        levels = nested_levels(shuffled)
        control_cells.append([len(set(level.values())) for level in levels])
        control_co.append(co_membership(levels, anchors, pairs["positives"].tolist()))

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
        "nesting_breaks": nesting_breaks(real_levels),
        "note": ("the four-point statistic of a nested partition is identically zero, "
                 "so it cannot separate a real hierarchy from a rewire; the "
                 "co-membership z-scores against degree-preserving controls are the "
                 "evidence, and a table-level nesting check is reported instead"),
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
    manifest = {
        "artifact": "results/research/hierarchy_audit/hierarchy.npz",
        "sha256": hashlib.sha256((OUT / "hierarchy.npz").read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256((OUT / "report.json").read_bytes()).hexdigest(),
        "builder": "tools/research/science/hierarchy_audit.py",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "regenerate": "python3 tools/research/science/hierarchy_audit.py",
        "inputs": {
            "train_parquet": str(TRAIN.relative_to(REPO)),
            "train_parquet_sha256": (hashlib.sha256(TRAIN.read_bytes()).hexdigest()
                                     if TRAIN.exists() else None),
            "validation_pairs": str(VALID_PAIRS.relative_to(REPO)),
            "validation_pairs_sha256": (hashlib.sha256(VALID_PAIRS.read_bytes()).hexdigest()
                                        if VALID_PAIRS.exists() else None),
        },
        "table": {
            "rows": int(table.shape[0]),
            "levels": int(table.shape[1]),
            "cells_per_level": [int(len(set(level.values()))) for level in real_levels],
            "unknown_items_per_level": [int((table[:, level] < 0).sum())
                                        for level in range(table.shape[1])],
            "nesting_breaks": nesting_breaks(real_levels),
            "row_order": "ascending item id of the frozen baseline item_sids.json",
        },
        "train_only": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
