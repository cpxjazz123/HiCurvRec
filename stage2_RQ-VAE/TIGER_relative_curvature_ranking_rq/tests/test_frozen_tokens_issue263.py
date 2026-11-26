"""Candidate-specific tests for the Issue #263 frozen-token Stage2 path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "stage2_RQ-VAE/TIGER_relative_curvature_ranking_rq"
CACHE = ROOT / "results/stage2_RQ-VAE/TIGER_frozen_topology_gnn_rq/token_cache.npz"

import sys
sys.path.insert(0, str(TREE))

from model.frozen_tokens import (FrozenTokenAttention, FrozenTokenStore, N_ITEMS,  # noqa: E402
                                 TOKEN_CACHE, TOKEN_DIM, TOKEN_TYPES)


def test_cache_identity_shape_and_mask():
    assert CACHE.is_file(), "run pretrain_gnn.py before this test"
    with np.load(CACHE, allow_pickle=False) as blob:
        tokens = blob["tokens"]
        mask = blob["mask"]
        ids = blob["item_ids"]
    assert tokens.shape == (N_ITEMS, len(TOKEN_TYPES), TOKEN_DIM)
    assert mask.shape == (N_ITEMS, len(TOKEN_TYPES))
    assert np.array_equal(ids, np.arange(N_ITEMS))
    assert np.isfinite(tokens).all()
    assert mask.any()


def test_gate_reads_frozen_tokens_without_gnn():
    gate = FrozenTokenAttention(cache_path=CACHE)
    ids = np.array([0, 17, 24586, 17], dtype=np.int64)
    values, weights = gate.gates_for_items(FrozenTokenStore(), ids, "cpu")
    assert values.shape == (4, len(TOKEN_TYPES), TOKEN_DIM)
    assert torch.equal(values[1], values[3])
    assert weights.shape == (4, 3, 3)
    assert torch.allclose(weights.sum(-1), torch.ones(4, 3), atol=1e-6)
    assert gate.gnn_forward_calls == 0 and gate.gnn_backward_calls == 0
    assert gate.frozen_tokens.requires_grad is False
    assert gate.attention.query.requires_grad is True


def test_stage2_mechanism_has_no_gnn_instantiation():
    text = (TREE / "model/mechanism.py").read_text()
    assert "EgoStore" not in text
    assert "SubgraphTokens" not in text
    assert "FrozenTokenAttention" in text
    assert "FrozenTokenStore" in text
