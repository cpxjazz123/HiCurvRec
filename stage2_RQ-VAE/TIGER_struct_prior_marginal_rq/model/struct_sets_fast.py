"""Item-indexed candidate sets: the same frozen triangles and edges, without the full scan.

``struct_prior.batch_sets`` keeps every triangle and every edge whose endpoints are
all in the batch, and it finds them by mapping the *whole* frozen pool through a
lookup array on every step: 546438 triangles and 315391 edges per call, six calls
per epoch, 3000 epochs.  Measured on the real pool that is 35.7 ms per step, about
62% of one epoch of this arm.

This module answers the same question from the other side: once, at load time, it
builds an item -> incident-row index (CSR) for both pools; per step it gathers only
the rows incident to the batch's own items (about 68k candidates instead of 861k),
deduplicates with ``np.unique`` (which also restores the ascending row order of the
pool) and then applies the identical membership test.

The returned dict is *identical* -- same arrays, same order, same coverage counts --
to ``batch_sets``; ``tests/test_struct_sets_fast.py`` asserts that over random
batches of several sizes.
"""
from __future__ import annotations

import numpy as np


def _csr(rows: np.ndarray, width: int, item_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row ids grouped by every item that appears in them."""
    flat = rows.reshape(-1)
    order = np.argsort(flat, kind="stable")
    row_of_occurrence = np.repeat(np.arange(rows.shape[0], dtype=np.int64), width)[order]
    sorted_items = flat[order]
    counts = np.bincount(sorted_items, minlength=item_count).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return row_of_occurrence, counts, starts


class SetWindow:
    """The frozen pools plus their item -> incident-row index, built once."""

    def __init__(self, edges: np.ndarray, triangles: np.ndarray, item_count: int):
        self.item_count = int(item_count)
        self.edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.edge_rows, self.edge_counts, self.edge_starts = _csr(self.edges, 2, self.item_count)
        self.triangle_rows, self.triangle_counts, self.triangle_starts = _csr(
            self.triangles, 3, self.item_count)

    @staticmethod
    def _gather(rows: np.ndarray, counts: np.ndarray, starts: np.ndarray,
                ids: np.ndarray) -> np.ndarray:
        keep = counts[ids]
        total = int(keep.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        offsets = np.repeat(starts[ids] - (np.cumsum(keep) - keep), keep)
        return rows[offsets + np.arange(total, dtype=np.int64)]

    def batch_sets(self, item_ids: np.ndarray, layer_of_item: np.ndarray) -> dict:
        ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)
        layers = np.asarray(layer_of_item, dtype=np.int64).reshape(-1)
        if layers.shape[0] != ids.shape[0]:
            raise ValueError("one layer per item id is required")

        triangle_candidates = np.unique(
            self._gather(self.triangle_rows, self.triangle_counts, self.triangle_starts, ids))
        edge_candidates = np.unique(
            self._gather(self.edge_rows, self.edge_counts, self.edge_starts, ids))

        lookup = np.full(self.item_count, -1, dtype=np.int64)
        lookup[ids] = np.arange(ids.size)

        if triangle_candidates.size:
            mapped_triangles = lookup[self.triangles[triangle_candidates]]
            triangle_keep = (mapped_triangles >= 0).all(axis=1)
        else:
            mapped_triangles = np.zeros((0, 3), dtype=np.int64)
            triangle_keep = np.zeros(0, dtype=bool)

        if edge_candidates.size:
            mapped_edges = lookup[self.edges[edge_candidates]]
            edge_keep = (mapped_edges >= 0).all(axis=1)
        else:
            mapped_edges = np.zeros((0, 2), dtype=np.int64)
            edge_keep = np.zeros(0, dtype=bool)

        kept_edges = mapped_edges[edge_keep]
        same_layer = layers[kept_edges[:, 0]] == layers[kept_edges[:, 1]] if kept_edges.size \
            else np.zeros(0, dtype=bool)
        return {
            "triangles": mapped_triangles[triangle_keep],
            "within": kept_edges[same_layer],
            "cross": kept_edges[~same_layer],
            "coverage": {
                "triangles_used": int(triangle_keep.sum()),
                "triangles_total": int(self.triangles.shape[0]),
                "edges_used": int(edge_keep.sum()),
                "edges_total": int(self.edges.shape[0]),
                "within_used": int(kept_edges[same_layer].shape[0]),
                "cross_used": int(kept_edges[~same_layer].shape[0]),
            },
        }


__all__ = ["SetWindow"]
