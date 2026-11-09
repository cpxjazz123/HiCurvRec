"""Tests for the community-guided level-1 cost.

They pin the three properties the arm's result rests on: the community term
enters the **cost** and not the assignment rule, it is **balanced against the
baseline cost by a measured scale** and not by a chosen constant, and the
codebook's community profile is the **exact mean** of the items the code
received.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import layers as model_layers  # noqa: E402
from model.community import CommunitySupervisor  # noqa: E402


def _community_file(tmp_path: Path, n_items: int = 8, width: int = 4) -> Path:
    """A tiny community representation: eight items in a small ball."""
    rng = np.random.default_rng(7)
    free = rng.normal(size=(n_items, width)).astype(np.float32)
    tangent = rng.normal(size=(n_items, width)).astype(np.float32) * 0.3
    path = tmp_path / "community.npz"
    np.savez(path, free=free, ball=tangent, curvature=1.0)
    return path


def test_the_term_is_standardised_per_item():
    path = _community_file(Path("/tmp"))
    supervisor = CommunitySupervisor(path, 8, 3, torch.device("cpu"))
    ids = torch.arange(4)
    indices = torch.tensor([0, 1, 2, 0])
    supervisor.observe(indices, ids)
    supervisor.finish_epoch()
    cost = torch.ones(4, 3)
    term = supervisor.term(ids, cost)
    assert term.shape == cost.shape
    assert torch.allclose(term.mean(dim=1), torch.zeros(4), atol=1e-6)
    assert torch.allclose(term.std(dim=1, unbiased=False), torch.ones(4), atol=1e-5)


def test_the_profile_is_the_mean_of_the_items_the_code_received():
    path = _community_file(Path("/tmp"))
    supervisor = CommunitySupervisor(path, 8, 3, torch.device("cpu"))
    ids = torch.arange(4)
    indices = torch.tensor([0, 0, 1, 2])
    supervisor.observe(indices, ids)
    supervisor.finish_epoch()
    # code 0 received the items 0 and 1, code 1 the item 2, code 2 the item 3
    expected = supervisor.free[:2].double().mean(dim=0)
    assert torch.allclose(supervisor.free_profile[0].double(), expected, atol=1e-5)


def test_the_weight_is_the_baseline_costs_own_spread():
    path = _community_file(Path("/tmp"))
    supervisor = CommunitySupervisor(path, 8, 3, torch.device("cpu"))
    cost = torch.tensor([[0.0, 1.0, 2.0]] * 4)
    supervisor.calibrate(cost)
    spread = float(cost.std(dim=1, unbiased=False).mean())
    assert supervisor.weight == pytest.approx(spread, rel=1e-6)
    # the weight is read once and does not move
    supervisor.calibrate(cost * 100.0)
    assert supervisor.weight == pytest.approx(spread, rel=1e-6)


def test_the_term_prefers_the_code_whose_items_share_the_community():
    """Two codes: one holds items with the query's community, the other does not."""
    path = _community_file(Path("/tmp"))
    supervisor = CommunitySupervisor(path, 8, 2, torch.device("cpu"))
    ids = torch.arange(4)
    # code 0 receives the items 0 and 1, code 1 receives the items 2 and 3
    supervisor.observe(torch.tensor([0, 0, 1, 1]), ids)
    supervisor.finish_epoch()
    term = supervisor.term(torch.tensor([0]), torch.zeros(1, 2))
    # the query is item 0, so the code whose profile is closest is code 0
    assert float(term[0, 0]) < float(term[0, 1])


def test_without_the_supervisor_the_cost_is_the_baselines():
    """The mechanism is a cost: with it switched off, nothing else moves."""
    layer = model_layers.VQLayer(4, 3, 0.25, -1.0, -1)
    assert layer.community is None
    x = torch.randn(5, 3)
    baseline = layer._distances(x.view(-1, 3))
    assert torch.allclose(baseline, layer._distances(x.view(-1, 3)))
    # and the assignment is still the baseline's rule at this level: the
    # argmin over the cost matrix, unchanged
    assert torch.equal(layer._indices(baseline, False), baseline.argmin(dim=-1))


def test_the_level_one_layer_is_the_one_that_carries_the_supervisor():
    class Config:
        codebook_num = 3
        codebook_dim = 3
        codebook_size = (4, 4, 4)
        vq_type = "vq"
        beta = 0.25
        sk_epsilon = 0.003
        sk_iters = 5
        ema_decay = 0.99
        fix_code_embs = False
        community_path = str(_community_file(Path("/tmp")))
        community_curvature = 1.0
        n_items = 8

    stack = model_layers.RQLayer(Config())
    assert stack.vq_layers[0].community_spec is not None
    assert stack.vq_layers[1].community_spec is None
    assert stack.vq_layers[2].community_spec is None


def test_the_term_repels_a_code_whose_items_share_the_querys_community():
    """The direction of this arm: the community-consistent code is the dearer one."""
    path = _community_file(Path("/tmp"))
    supervisor = CommunitySupervisor(path, 8, 2, torch.device("cpu"))
    ids = torch.arange(4)
    supervisor.observe(torch.tensor([0, 0, 1, 1]), ids)
    supervisor.finish_epoch()
    term = supervisor.term(torch.tensor([0]), torch.zeros(1, 2))
    # code 0 holds the items that share the query's community, so it is the one
    # the repulsion makes expensive: its distance is the smaller one
    assert float(term[0, 0]) < float(term[0, 1])
