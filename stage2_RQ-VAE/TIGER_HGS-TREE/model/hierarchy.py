"""The frozen behaviour hierarchy the HGS supervision reads, and its sampler.

Built by ``tools/research/science/hierarchy_audit.py`` from the TRAINING split
only, and audited against a degree-preserving random reconnect on the independent
validation pairs (z = 181 / 76 / 14 for the three levels, so the hierarchy puts
held-out behaviour pairs together 10-20x more often than the control does).

The table is (items, levels): column 0 is the root (one cell), then 17, 948 and
13416 cells at increasing depth. ``-1`` marks an item the training graph never
mentions; those items take part in no supervision.

``Sampler`` exists because the first HGS run supervised a random 128-row slice
of the batch and could therefore not see the finest level at all: with 13,416
leaf cells the expected number of same-cell pairs in such a slice is 0.62, so
half of the steps supervised no fine-level pair while the 17-cell coarse level
supplied 978. The sampler reads cached cell -> member tables over the whole
catalogue and builds positives that genuinely exist, at every level.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
TABLE = REPO / "results" / "research" / "hierarchy_audit" / "hierarchy.npz"
SIDS = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"

#: Rows one step may put through the encoder for supervision: a compute bound,
#: not a tuned parameter. It caps ``len(out["rows"])``.
SUPERVISION_ROWS = 512


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
    """``T(i,j) = levels - LCP(i,j)``, read top down.

    A level counts only if every level above it also agrees, so the agreement of
    the levels is accumulated as a running product, not summed.
    """
    rows = cells.shape[0]
    running = torch.ones(rows, rows, dtype=torch.bool, device=cells.device)
    shared = torch.zeros(rows, rows, device=cells.device)
    for level in range(cells.shape[1]):
        same = (cells[:, None, level] == cells[None, :, level]) & (cells[:, None, level] >= 0)
        running = running & same
        shared = shared + running.float()
    return float(cells.shape[1]) - shared


class Sampler:
    """Cached cell tables, and pairs the hierarchy can order.

    Every returned triplet is ``(anchor, near, far)`` with
    ``T(anchor, near) < T(anchor, far)`` by construction: ``near`` shares the
    sampled level's cell, ``far`` is in a different cell of the same level --
    preferably a sibling branch under the same parent, which is the hard negative.
    """

    def __init__(self, cells: np.ndarray, seed: int = 0, rank: int = 0):
        cells = np.ascontiguousarray(np.asarray(cells, dtype=np.int64))
        self.cells = cells
        self.items, self.columns = cells.shape
        self.levels = list(range(1, self.columns))
        self.members: dict[int, dict[int, np.ndarray]] = {}
        self.siblings: dict[int, dict[int, np.ndarray]] = {}
        for level in self.levels:
            column = cells[:, level]
            groups: dict[int, list[int]] = {}
            for row in np.nonzero(column >= 0)[0]:
                groups.setdefault(int(column[row]), []).append(int(row))
            self.members[level] = {cell: np.asarray(rows, dtype=np.int64)
                                   for cell, rows in groups.items()}
            parents: dict[int, list[int]] = {}
            for cell, rows in self.members[level].items():
                parent = int(cells[int(rows[0]), level - 1])
                parents.setdefault(parent, []).append(int(cell))
            self.siblings[level] = {parent: np.asarray(value, dtype=np.int64)
                                    for parent, value in parents.items()}
        self._rng = np.random.default_rng(seed * 1000 + rank)
        self.coverage: dict[str, dict[int, int]] = {}

    def has_positive(self, row: int, level: int) -> bool:
        cell = int(self.cells[row, level])
        return cell >= 0 and len(self.members[level][cell]) > 1

    def positive(self, row: int, level: int) -> int | None:
        """A genuine non-self partner in the anchor's cell at this level."""
        cell = int(self.cells[row, level])
        members = self.members[level].get(cell)
        if members is None or len(members) < 2:
            return None
        picked = int(members[self._rng.integers(0, len(members))])
        if picked == row:
            # step to the next member: a self pair would make the ordering term vacuous
            picked = int(members[(int(np.searchsorted(members, row)) + 1) % len(members)])
        return picked

    def negative(self, row: int, level: int) -> int | None:
        """A different cell at this level; a sibling branch when one exists."""
        cell = int(self.cells[row, level])
        parent = int(self.cells[row, level - 1])
        candidates = self.siblings[level].get(parent)
        if candidates is not None:
            wanted = candidates[candidates != cell]
        else:
            wanted = np.empty(0, dtype=np.int64)
        if len(wanted) == 0:
            wanted = np.asarray([c for c in self.members[level] if c != cell], dtype=np.int64)
        if len(wanted) == 0:
            return None
        other = int(wanted[self._rng.integers(0, len(wanted))])
        members = self.members[level][other]
        return int(members[self._rng.integers(0, len(members))])

    def prefix_partner(self, row: int, level: int) -> int | None:
        """Shares this cell but not the next one: what the prefix term asks for."""
        if level + 1 >= self.columns:
            return None
        cell = int(self.cells[row, level])
        members = self.members[level].get(cell)
        if members is None or len(members) < 2:
            return None
        deeper = self.cells[members, level + 1] != self.cells[row, level + 1]
        wanted = members[deeper]
        if len(wanted) == 0:
            return None
        return int(wanted[self._rng.integers(0, len(wanted))])

    def sample(self, pool, rng=None, triples: int = 32, pairs: int = 24) -> dict:
        """A supervision set: unique rows plus index tensors into them."""
        rng = self._rng if rng is None else rng
        rows: list[int] = []
        slot: dict[int, int] = {}

        def put(row: int) -> int:
            if row not in slot:
                slot[row] = len(rows)
                rows.append(row)
            return slot[row]

        tree: list[tuple[int, int, int]] = []
        depth: list[int] = []
        skipped = {"tree": 0, "direction": {level: 0 for level in self.levels},
                   "prefix": {level: 0 for level in self.levels}, "rows": 0}
        for _ in range(triples):
            anchor = int(pool[rng.integers(0, len(pool))])
            levels = [level for level in self.levels if self.has_positive(anchor, level)]
            if not levels:
                skipped["tree"] += 1
                continue
            level = int(levels[rng.integers(0, len(levels))])
            near = self.positive(anchor, level)
            far = self.negative(anchor, level)
            if near is None or far is None:
                skipped["tree"] += 1
                continue
            tree.append((put(anchor), put(near), put(far)))
            depth.append(level)

        direction: dict[int, list[tuple[int, int, int]]] = {level: [] for level in self.levels}
        prefix: dict[int, list[tuple[int, int]]] = {level: [] for level in self.levels}
        for level in self.levels:
            for _ in range(pairs):
                anchor = int(pool[rng.integers(0, len(pool))])
                if not self.has_positive(anchor, level):
                    skipped["direction"][level] += 1
                    continue
                near = self.positive(anchor, level)
                far = self.negative(anchor, level)
                if near is None or far is None:
                    skipped["direction"][level] += 1
                    continue
                direction[level].append((put(anchor), put(near), put(far)))
                partner = self.prefix_partner(anchor, level)
                if partner is None:
                    skipped["prefix"][level] += 1
                else:
                    prefix[level].append((put(anchor), put(partner)))

        # the supervision rows are encoded once per step; keep the set bounded.
        # entries are positioned by slot, so truncation only drops the tail.
        keep = min(len(rows), SUPERVISION_ROWS)
        rows = rows[:keep]

        def alive(group) -> bool:
            return all(int(index) < keep for index in group)

        tree_array = np.asarray([group for group in tree if alive(group)], dtype=np.int64).reshape(-1, 3)
        depth = np.asarray([level for group, level in zip(tree, depth) if alive(group)], dtype=np.int64)
        tree_index = (tree_array[:, 0].copy(), tree_array[:, 1].copy(), tree_array[:, 2].copy())
        direction_index = {}
        for level, groups in direction.items():
            kept = np.asarray([group for group in groups if alive(group)], dtype=np.int64).reshape(-1, 3)
            direction_index[level] = (kept[:, 0].copy(), kept[:, 1].copy(), kept[:, 2].copy())
        prefix_index = {}
        for level, groups in prefix.items():
            kept = np.asarray([group for group in groups if alive(group)], dtype=np.int64).reshape(-1, 2)
            prefix_index[level] = (kept[:, 0].copy(), kept[:, 1].copy())
        self.coverage = {"used": {"tree": int(tree_index[0].shape[0]),
                                  "direction": {k: int(v[0].shape[0]) for k, v in direction_index.items()},
                                  "prefix": {k: int(v[0].shape[0]) for k, v in prefix_index.items()},
                                  "rows": len(rows)},
                         "skipped": skipped}
        return {"rows": np.asarray(rows, dtype=np.int64), "tree": tree_index,
                "depth": depth, "direction": direction_index, "prefix": prefix_index}
