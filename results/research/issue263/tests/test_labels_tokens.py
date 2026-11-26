"""Deterministic tests for Issue #263 labels and frozen-token interface."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "results/research/issue263"))
sys.path.insert(0, str(ROOT / "stage2_RQ-VAE/TIGER_frozen_topology_gnn_rq"))

from pretrain_gnn import N_ITEMS, load_targets  # noqa: E402
from model.frozen_tokens import FrozenTokenAttention, FrozenTokenStore, N_ITEMS as FROZEN_N  # noqa: E402


def test_label_shapes_masks_and_normalization():
    t = load_targets()
    assert t["own"].shape[0] == N_ITEMS and t["dgdv"].shape[0] == N_ITEMS
    assert t["curvature"].shape[0] == N_ITEMS
    assert t["own_mask"].sum() == 12266
    assert t["dgdv_mask"].sum() == 17389
    assert int(t["curve_mask"].sum()) > 0
    covered = t["own_mask"]
    assert np.allclose(t["own"][covered].sum(1), 1.0, atol=1e-5)
    dg = t["dgdv_mask"]
    assert np.allclose(t["dgdv"][dg].sum(1), 1.0, atol=1e-5)
    assert np.allclose(t["curvature"][t["curve_mask"]].sum(1), 1.0, atol=1e-5)
    assert len(t["curve_edges"]) == t["curvature"].shape[1] + 1
    assert not t["own_mask"].all()  # missing evidence is explicit, not imputed


def test_frozen_token_attention_reads_exact_ids_without_gnn(tmp_path):
    rng = np.random.default_rng(42)
    tokens = rng.normal(size=(N_ITEMS, 4, 32)).astype(np.float32)
    mask = np.ones((N_ITEMS, 4), dtype=bool)
    path = tmp_path / "cache.npz"
    np.savez_compressed(path, tokens=tokens, mask=mask,
                        item_ids=np.arange(N_ITEMS, dtype=np.int64))
    gate = FrozenTokenAttention(cache_path=path)
    ids = np.array([0, 17, 24586, 17], dtype=np.int64)
    values, weights = gate.gates_for_items(FrozenTokenStore(), ids, "cpu")
    assert values.shape == (4, 4, 32)
    assert torch.equal(values[1], values[3])
    assert weights.shape == (4, 3, 3)
    assert torch.allclose(weights.sum(-1), torch.ones(4, 3), atol=1e-6)
    assert gate.gnn_forward_calls == 0 and gate.gnn_backward_calls == 0
    assert gate.attention.query.requires_grad
    assert gate.frozen_tokens.requires_grad is False


def test_mechanism_source_has_no_online_gnn():
    source = (ROOT / "stage2_RQ-VAE/TIGER_frozen_topology_gnn_rq/model/mechanism.py").read_text()
    assert "EgoStore" not in source
    assert "SubgraphTokens" not in source
    assert "FrozenTokenAttention" in source
    assert "FrozenTokenStore" in source
