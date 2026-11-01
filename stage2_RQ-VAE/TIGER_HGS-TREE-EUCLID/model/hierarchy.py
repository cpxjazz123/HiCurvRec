"""The frozen behaviour hierarchy the HGS supervision reads.

Built by ``tools/research/science/hierarchy_audit.py`` from the TRAINING split
only, and audited against a degree-preserving random reconnect on the independent
validation pairs (z = 181 / 76 / 14 for the three levels, so the hierarchy puts
held-out behaviour pairs together 10-20x more often than the control does).

The table is (items, levels): column 0 is the root (one cell), then 17, 948 and
13416 cells at increasing depth. ``-1`` marks an item the training graph never
mentions; those items take part in no supervision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
TABLE = REPO / "results" / "research" / "hierarchy_audit" / "hierarchy.npz"
SIDS = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"


def load() -> dict:
    payload = np.load(TABLE)
    items = payload["items"]
    cells = payload["cells"]
    row_of = {int(item): row for row, item in enumerate(items)}
    # the trainer indexes the batch by the embedding row, and the embedding order
    # is the frozen baseline SID key order
    order = list(json.loads(SIDS.read_text()))
    mapped = np.full((len(order), cells.shape[1]), -1, dtype=np.int64)
    for row, item in enumerate(order):
        source = row_of.get(int(item))
        if source is not None:
            mapped[row] = cells[source]
    return {"cells": torch.from_numpy(mapped), "levels": int(cells.shape[1]),
            "items": len(order)}


def tree_distance(cells: torch.Tensor) -> torch.Tensor:
    """``T(i,j) = levels - shared(i,j)``: identical items have distance 0.

    A level counts only if every level above it also agrees, so the agreement of
    the levels is accumulated as a running product, not summed.
    """
    rows = cells.shape[0]
    running = torch.ones(rows, rows, dtype=torch.bool)
    shared = torch.zeros(rows, rows)
    for level in range(cells.shape[1]):
        same = (cells[:, None, level] == cells[None, :, level]) & (cells[:, None, level] >= 0)
        running = running & same
        shared = shared + running.float()
    return float(cells.shape[1]) - shared


def sample_triples(cells: torch.Tensor, rows: int, generator: torch.Generator):
    """(anchor, near, far): near shares the deepest cell, far is in another branch."""
    anchors = torch.randint(0, cells.shape[0], (rows,), generator=generator)
    deepest = cells[:, -1]
    second = cells[:, 1] if cells.shape[1] > 1 else cells[:, -1]
    near = anchors.clone()
    far = anchors.clone()
    order = torch.randperm(cells.shape[0], generator=generator)
    buckets: dict[int, torch.Tensor] = {}
    for row in range(rows):
        anchor = int(anchors[row])
        cell = int(deepest[anchor])
        if cell < 0:
            continue
        members = buckets.get(cell)
        if members is None:
            members = (deepest == cell).nonzero(as_tuple=False).flatten()
            buckets[cell] = members
        near[row] = members[torch.randint(0, len(members), (1,), generator=generator)]
        branch = int(second[anchor])
        for candidate in order:
            if int(second[candidate]) != branch and int(second[candidate]) >= 0:
                far[row] = candidate
                break
    keep = (deepest[anchors] >= 0) & (second[anchors] >= 0) & (far != anchors)
    return anchors[keep], near[keep], far[keep]
