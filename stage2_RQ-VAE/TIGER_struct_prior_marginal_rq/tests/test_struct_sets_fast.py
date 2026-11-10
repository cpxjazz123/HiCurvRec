"""The item-indexed set window must return exactly what ``batch_sets`` returns.

The fast path is only a different way to *find* the rows: the frozen pools, the
membership test, the ascending row order and the coverage counts must be identical,
otherwise the structural terms would silently see a different graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model.struct_prior import batch_sets as reference_sets          # noqa: E402
from model.struct_sets_fast import SetWindow, SetWindowTorch          # noqa: E402


def synthetic_pool(items: int, edges: int, triangles: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, items, size=(edges, 2))
    keep = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[keep]
    triples = rng.integers(0, items, size=(triangles, 3))
    distinct = (triples[:, 0] != triples[:, 1]) & (triples[:, 1] != triples[:, 2]) \
        & (triples[:, 0] != triples[:, 2])
    return pairs.astype(np.int64), triples[distinct].astype(np.int64)


def layer_table(items: int, groups: int = 4, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, groups, size=items).astype(np.int64)


@pytest.mark.parametrize("batch", [1, 7, 64, 512])
def test_fast_sets_equal_the_reference_on_random_batches(batch):
    items = 4096
    edges, triangles = synthetic_pool(items, edges=20000, triangles=30000)
    layer_of = layer_table(items)
    window = SetWindow(edges, triangles, items)
    rng = np.random.default_rng(batch)
    for _ in range(15):
        ids = rng.choice(items, size=batch, replace=False)
        fast = window.batch_sets(ids, layer_of[ids])
        slow = reference_sets(ids, layer_of[ids], edges, triangles, items)
        for key in ("triangles", "within", "cross"):
            torch_free_equal(fast[key], slow[key], key)
        assert fast["coverage"] == slow["coverage"]


def torch_free_equal(fast, slow, key):
    """Same values, same order, same dtype family."""
    fast = np.asarray(fast)
    slow = np.asarray(slow)
    assert fast.shape == slow.shape, (key, fast.shape, slow.shape)
    assert np.array_equal(fast, slow), key


def test_the_order_is_the_pool_order_not_the_gather_order():
    """``np.unique`` must restore ascending row order, which is what the reference keeps."""
    items = 512
    edges, triangles = synthetic_pool(items, edges=4000, triangles=6000, seed=3)
    layer_of = layer_table(items)
    window = SetWindow(edges, triangles, items)
    ids = np.array([4, 9, 17, 33, 200, 411], dtype=np.int64)
    fast = window.batch_sets(ids, layer_of[ids])
    slow = reference_sets(ids, layer_of[ids], edges, triangles, items)
    assert np.array_equal(fast["triangles"], slow["triangles"])
    # and the rows really are the kept rows of the pool, in pool order
    keep = np.isin(triangles, ids).all(axis=1)
    mapped = np.full(items, -1, dtype=np.int64)
    mapped[ids] = np.arange(ids.size)
    assert np.array_equal(fast["triangles"], mapped[triangles[keep]])


def test_empty_and_singleton_batches_are_handled():
    items = 256
    edges, triangles = synthetic_pool(items, edges=1500, triangles=2000, seed=5)
    layer_of = layer_table(items)
    window = SetWindow(edges, triangles, items)
    for ids in (np.array([], dtype=np.int64), np.array([7], dtype=np.int64),
                np.array([3, 5], dtype=np.int64)):
        fast = window.batch_sets(ids, layer_of[ids])
        slow = reference_sets(ids, layer_of[ids], edges, triangles, items)
        assert fast["coverage"] == slow["coverage"]
        assert np.array_equal(np.asarray(fast["triangles"]), np.asarray(slow["triangles"]))


def test_the_window_is_reusable_and_deterministic():
    """Two calls on the same batch must agree: the index is read-only."""
    items = 1024
    edges, triangles = synthetic_pool(items, edges=8000, triangles=12000, seed=7)
    layer_of = layer_table(items)
    window = SetWindow(edges, triangles, items)
    rng = np.random.default_rng(11)
    first_ids = rng.choice(items, size=256, replace=False)
    one = window.batch_sets(first_ids, layer_of[first_ids])
    other_ids = rng.choice(items, size=256, replace=False)
    _ = window.batch_sets(other_ids, layer_of[other_ids])
    two = window.batch_sets(first_ids, layer_of[first_ids])
    for key in ("triangles", "within", "cross"):
        assert np.array_equal(np.asarray(one[key]), np.asarray(two[key]))
    assert one["coverage"] == two["coverage"]


REAL = TREE.parents[0].parent / "results" / "research" / "issue150" / "structure_prior.npz"


@pytest.mark.skipif(not REAL.is_file(), reason="the frozen #150 prior is not present")
def test_the_real_pool_matches_the_reference():
    payload = np.load(REAL)
    edges = np.asarray(payload["edges"], dtype=np.int64)
    triangles = np.asarray(payload["triangles"], dtype=np.int64)
    items = int(np.load(REAL.with_suffix(".npz"), mmap_mode="r")["item_ids"].max()) + 1 \
        if "item_ids" in payload.files else 24587
    layer_of = np.full(int(items), -1, dtype=np.int64)
    layer_of[np.asarray(payload["item_ids"], dtype=np.int64)] = np.asarray(payload["layer"], dtype=np.int64)
    window = SetWindow(edges, triangles, int(items))
    rng = np.random.default_rng(0)
    for _ in range(3):
        ids = rng.choice(24556, size=1024, replace=False).astype(np.int64)
        fast = window.batch_sets(ids, layer_of[ids])
        slow = reference_sets(ids, layer_of[ids], edges, triangles, int(items))
        assert fast["coverage"] == slow["coverage"]
        for key in ("triangles", "within", "cross"):
            assert np.array_equal(np.asarray(fast[key]), np.asarray(slow[key])), key

def test_the_device_window_matches_the_reference_on_random_batches():
    """The CUDA path must answer exactly what the host path answers.

    The tensor window runs the same two gathers on whatever device it is given, so
    the equality is checkable here with CPU tensors; on a card the kernels differ
    but the quantities do not.
    """
    items = 4096
    edges, triangles = synthetic_pool(items, edges=20000, triangles=30000)
    layer_of = layer_table(items)
    window = SetWindowTorch(edges, triangles, items, "cpu")
    rng = np.random.default_rng(13)
    for batch in (1, 7, 64, 512):
        for _ in range(6):
            ids = rng.choice(items, size=batch, replace=False)
            fast = window.batch_sets(ids, layer_of[ids])
            slow = reference_sets(ids, layer_of[ids], edges, triangles, items)
            assert fast["coverage"] == slow["coverage"]
            for key in ("triangles", "within", "cross"):
                assert np.array_equal(np.asarray(fast[key]), np.asarray(slow[key])), key


@pytest.mark.skipif(not REAL.is_file(), reason="the frozen #150 prior is not present")
def test_the_device_window_matches_the_reference_on_the_real_pool():
    payload = np.load(REAL)
    edges = np.asarray(payload["edges"], dtype=np.int64)
    triangles = np.asarray(payload["triangles"], dtype=np.int64)
    layer_of = np.full(24587, -1, dtype=np.int64)
    layer_of[np.asarray(payload["item_ids"], dtype=np.int64)] = np.asarray(payload["layer"], dtype=np.int64)
    window = SetWindowTorch(edges, triangles, 24587, "cpu")
    rng = np.random.default_rng(17)
    for _ in range(3):
        ids = rng.choice(24556, size=1024, replace=False).astype(np.int64)
        fast = window.batch_sets(ids, layer_of[ids])
        slow = reference_sets(ids, layer_of[ids], edges, triangles, 24587)
        assert fast["coverage"] == slow["coverage"]
        for key in ("triangles", "within", "cross"):
            assert np.array_equal(np.asarray(fast[key]), np.asarray(slow[key])), key
