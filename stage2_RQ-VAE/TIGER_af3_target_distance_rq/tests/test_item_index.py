"""The row -> item id map the AF3 losses need (red first)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import item_index as ii  # noqa: E402


def test_round_trip_recovers_every_id():
    embeddings = np.ascontiguousarray(np.random.default_rng(0).normal(size=(64, 8)).astype(np.float32))
    index = ii.build(embeddings)
    rows = torch.from_numpy(embeddings[[3, 17, 63]].copy())
    assert ii.lookup(rows, index).tolist() == [3, 17, 63]


def test_unknown_row_fails_closed():
    embeddings = np.ascontiguousarray(np.random.default_rng(1).normal(size=(16, 8)).astype(np.float32))
    index = ii.build(embeddings)
    stranger = torch.from_numpy(np.full((1, 8), 7.5, dtype=np.float32))
    with pytest.raises(KeyError):
        ii.lookup(stranger, index)


def test_tiny_float_perturbation_is_not_silently_matched():
    embeddings = np.ascontiguousarray(np.random.default_rng(2).normal(size=(8, 8)).astype(np.float32))
    index = ii.build(embeddings)
    perturbed = torch.from_numpy((embeddings[[0]] + np.float32(1e-6)).copy())
    with pytest.raises(KeyError):
        ii.lookup(perturbed, index)
