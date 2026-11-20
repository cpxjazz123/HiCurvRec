"""Issue #220 CGV 6: reject contaminated pools/graphs and collapsed runs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

import trq  # noqa: E402
from graphlet_roles import ORBIT_TO_GLOBAL, ROLE_MASK, TYPE_INDEX, load_instances  # noqa: E402


def test_pools_are_strictly_induced(monkeypatch):
    """Every instance has exactly the edge pattern its type demands."""
    from scipy.sparse import csr_matrix
    payload = np.load(trq.REPO / "results" / "research" / "behavior_graph" / "graph.npz",
                      allow_pickle=False)
    n_items = int(payload["nodes"])
    adjacency = csr_matrix((np.ones(len(payload["rows"])), (payload["rows"], payload["cols"])),
                           shape=(n_items, n_items)).maximum(
        csr_matrix((np.ones(len(payload["rows"])), (payload["cols"], payload["rows"])),
                   shape=(n_items, n_items)))
    adjacency = adjacency.tocsr()
    grouped = load_instances("heldout")
    for name, (nodes, _) in grouped.items():
        for instance in nodes[:200]:
            edges = sum(1 for i in range(len(instance)) for j in range(i + 1, len(instance))
                        if adjacency[instance[i], instance[j]] > 0)
            assert edges == {"K3": 3, "P3": 2, "C4": 4, "P4": 3, "K1_3": 3}[name], (name, instance)


def test_splits_do_not_overlap():
    train = {tuple(sorted(row.tolist())) for name in ("K3", "P3", "C4", "P4", "K1_3")
             for row in load_instances("train")[name][0]}
    heldout = {tuple(sorted(row.tolist())) for name in ("K3", "P3", "C4", "P4", "K1_3")
               for row in load_instances("heldout")[name][0]}
    assert train and heldout and not (train & heldout)


def test_graph_matches_metadata():
    graph = trq.REPO / "results" / "research" / "behavior_graph" / "graph.npz"
    recorded = json.loads((graph.parent / "metadata.json").read_text())["outputs"]["graph.npz"]
    assert hashlib.sha256(graph.read_bytes()).hexdigest() == recorded["sha256"]


def test_arms_are_the_pre_registered_five():
    assert trq.ARMS == ("A", "B", "C", "D", "E")
    assert trq.ARM_VARIANT in trq.ARMS


def test_role_masks_exclude_other_types_orbit():
    for name in ("K3", "P3", "C4", "P4", "K1_3"):
        mask = ROLE_MASK[TYPE_INDEX[name]]
        allowed = set(ORBIT_TO_GLOBAL[name].tolist())
        assert {index for index, flag in enumerate(mask.tolist()) if flag} == allowed


def _codebook(scale: float, n: int = 8, dim: int = 4) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n, dim) * scale


def test_health_monitor_rejects_collapse():
    base = _codebook(1.0)
    initial = trq.codeword_spacing(base)
    codes = np.stack([np.arange(24) % 8, np.zeros(24, dtype=np.int64)], 1)
    assert trq.health_verdict(codes, base, initial)["spacing_ratio"] > 0.5
    with pytest.raises(RuntimeError, match="spacing"):
        trq.health_verdict(codes, _codebook(0.01), initial)


def test_health_monitor_rejects_full_collision():
    base = _codebook(1.0)
    initial = trq.codeword_spacing(base)
    codes = np.zeros((200, 2), dtype=np.int64)
    with pytest.raises(RuntimeError, match="collision|max bucket"):
        trq.health_verdict(codes, base, initial)
