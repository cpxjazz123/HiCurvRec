"""The direction-shuffle control must change direction and nothing else.

The first version flipped each matrix cell independently, which can turn one
edge into a reciprocal pair or delete an edge outright.  These tests pin the
corrected contract: the undirected skeleton (every unordered pair's edge count)
is identical before and after the shuffle, and the direction count is close to
half but not identical (the shuffle is seeded and random).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from subgraph_predictor import shuffle_directions  # noqa: E402
from directed_role import instance_targets, load_vocabulary  # noqa: E402


def skeleton(a: np.ndarray) -> np.ndarray:
    """Undirected edge count per unordered pair, summed over instances."""
    undirected = a + np.transpose(a, (0, 2, 1))
    return undirected


def test_skeleton_is_preserved():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = (rng.random((64, 4, 4)) < 0.35).astype(np.float32)
        for i in range(4):
            a[:, i, i] = 0.0
        b = shuffle_directions(a, seed=7)
        assert np.array_equal(skeleton(a), skeleton(b))


def test_no_reciprocal_is_created_or_destroyed():
    rng = np.random.default_rng(1)
    a = np.zeros((32, 3, 3), dtype=np.float32)
    a[:, 0, 1] = 1.0                      # single edge 0 -> 1 everywhere
    b = shuffle_directions(a, seed=3)
    reciprocal = (b + np.transpose(b, (0, 2, 1))) == 2
    assert not reciprocal.any(), "a single edge must never become a reciprocal pair"
    assert (b[:, 0, 1] + b[:, 1, 0]).sum() == 32
    # half the instances should have reversed, within binomial noise
    reversed_share = float(b[:, 1, 0].mean())
    assert 0.2 < reversed_share < 0.8


def test_probe_targets_are_global_orbit_ids():
    """The probe fix: labels must be global orbit ids, not local subtype ids."""
    vocab = load_vocabulary("directed")
    subtypes = np.array([0, 1, 0, 1])
    orbits = np.array([[0, 1, 0], [1, 0, 1], [0, 2, 0], [2, 0, 2]])
    targets = instance_targets("P3", subtypes, orbits, vocab)
    assert targets["orbit"].max() < vocab["n_orbits"]
    assert targets["orbit"].max() > vocab["raw"]["P3"].__len__()
