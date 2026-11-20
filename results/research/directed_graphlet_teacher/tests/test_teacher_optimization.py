"""Issue #228: the training-mechanics and gate regression tests.

Every test here targets a failure the issue names explicitly, so a broken
harness fails before any long run starts:

  * `optimizer.step()` counts must be real updates, not epoch-multiplied
    microbatch counters;
  * the fixed 1,024-pair gate must actually be able to separate a separable
    set (an architecture that cannot fit 1,024 pairs is a hard stop);
  * the message graph may not contain a target pair in either direction;
  * sampled negatives may not be true edges in either direction;
  * every variant sees the same pair batches given the same seed;
  * degree-preserving rewiring keeps in- and out-degree and reports what it
    had to skip;
  * checkpoints reload bitwise;
  * the encoder receives gradients (a detached graph state would make every
    long run meaningless);
  * edge-only runs carry edge supervision only, so an attribution across
    losses is impossible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TESTS = Path(__file__).resolve().parent
TEACHER = TESTS.parent
ROLE = TEACHER.parent / "directed_graphlet_role"
sys.path.insert(0, str(TEACHER))
sys.path.insert(0, str(ROLE))

import graph_teacher3 as gt3  # noqa: E402


# ---------------------------------------------------------------- step counting

def test_step_counter_counts_updates_not_microbatches():
    torch.manual_seed(0)
    features, positives, negatives = _separable_set(64, dim=16)
    variants = gt3.build_variant("A_embedding", in_dim=16, hidden=32, node_dim=8)
    counter = gt3.StepCounter()
    result = gt3.run_steps(
        variants, features, None, positives, negatives, None, None,
        steps=3, accum=4, micro=16, lr=1e-3, weight_decay=0.0,
        device=torch.device("cpu"), counter=counter, loss_terms=("edge",),
        batch_seed=7, eval_fn=None, patience=None, eval_every=None)
    assert counter.steps == 3
    assert counter.microbatches == 12
    assert counter.epochs >= 3
    assert result["steps"] == 3
    assert result["microbatches"] == 12
    assert result["param_sha_after"] != result["param_sha_before"]


def test_gradient_reaches_the_encoder():
    torch.manual_seed(1)
    features, positives, negatives = _separable_set(64, dim=16)
    variant = gt3.build_variant("C_residual", in_dim=16, hidden=32, node_dim=8)
    adjacency = torch.eye(64).to_sparse()
    before = gt3.param_sha(variant.encoder.parameters())
    gt3.run_steps(variant, features, adjacency, positives, negatives, None, None,
                  steps=1, accum=1, micro=16, lr=1e-2, weight_decay=0.0,
                  device=torch.device("cpu"), counter=gt3.StepCounter(),
                  loss_terms=("edge",), batch_seed=3)
    after = gt3.param_sha(variant.encoder.parameters())
    assert before != after, "encoder parameters never moved"


# --------------------------------------------------------------- overfit gate

def test_overfit_gate_separates_a_separable_set():
    features, positives, negatives = _separable_set(128, dim=32)
    gate = gt3.overfit_gate(features, positives, negatives,
                            device=torch.device("cpu"), variant="A_embedding",
                            max_steps=300, target_auc=0.95, in_dim=32,
                            hidden=32, node_dim=8)
    assert gate["passed"], gate
    assert gate["train_auc"] >= 0.95
    assert gate["steps"] >= 1


def test_overfit_gate_reports_failure_without_raising():
    # random labels: no architecture can pass, the gate must report, not raise
    rng = np.random.default_rng(0)
    features = torch.from_numpy(rng.normal(size=(64, 16)).astype(np.float32))
    positives = rng.integers(0, 64, size=(1024, 2)).astype(np.int64)
    negatives = rng.integers(0, 64, size=1024).astype(np.int64)
    gate = gt3.overfit_gate(features, positives, negatives,
                            device=torch.device("cpu"), variant="A_embedding",
                            max_steps=5, target_auc=0.999, in_dim=16,
                            hidden=16, node_dim=8)
    assert gate["passed"] is False
    assert "diagnostics" in gate


# -------------------------------------------------------------- shared batches

def test_variants_see_identical_batches():
    left = gt3.PairStream(n=100, seed=11, micro=8)
    right = gt3.PairStream(n=100, seed=11, micro=8)
    for _ in range(10):
        assert np.array_equal(left.next(), right.next())
    other = gt3.PairStream(n=100, seed=12, micro=8)
    assert not np.array_equal(left.next(), other.next())


def test_loss_terms_are_declared_per_task():
    assert gt3.TASKS["edge"] == ("edge",)
    assert gt3.TASKS["edge+direction"] == ("edge", "direction")


# ------------------------------------------------------------- leakage checks

@pytest.fixture(scope="module")
def real_data():
    return gt3.prepare_data(device=torch.device("cpu"), load_features=False)


def test_message_graph_excludes_targets_and_reverse(real_data):
    message = real_data["message_edges"]
    assert len(message), "the message graph is empty"
    pairs = set(map(tuple, message.tolist()))
    targets = np.concatenate([real_data["train_positive"],
                              real_data["valid_positive"],
                              real_data["test_positive"]])
    for a, b in targets.tolist():
        assert (a, b) not in pairs, (a, b)
        assert (b, a) not in pairs, (b, a)


def test_negatives_are_not_edges_in_either_direction(real_data):
    edges = real_data["full_directed"]
    for positives, negatives in ((real_data["train_positive"], real_data["train_negative"]),
                                 (real_data["valid_positive"], real_data["valid_negative"]),
                                 (real_data["test_positive"], real_data["test_negative"])):
        assert len(positives) and len(positives) == len(negatives)
        for (a, _b), d in zip(positives.tolist(), negatives.tolist()):
            assert (int(a), int(d)) not in edges
            assert (int(d), int(a)) not in edges
            assert int(d) not in (int(a), int(_b))


def test_pass_rate_is_reported(real_data):
    for key in ("train", "valid", "test"):
        rate = real_data["negative_success"][key]["rate"]
        assert 0.0 < rate <= 1.0


# ------------------------------------------------------------------- rewiring

def test_degree_preserving_rewire_keeps_in_and_out_degrees():
    rng = np.random.default_rng(0)
    n, e = 40, 160
    rows = rng.integers(0, n, size=e)
    cols = rng.integers(0, n, size=e)
    keep = rows != cols
    index = np.stack([rows[keep], cols[keep]])
    rewired, stats = gt3.degree_preserving_rewire(index, n, forbidden=set(), seed=5)
    assert stats["edges_out"] == stats["edges_in"]
    assert stats["in_degree_l1"] == 0
    assert stats["out_degree_l1"] == 0
    assert not np.any(rewired[0] == rewired[1])
    assert stats["swaps"] > 0
    assert 0.0 <= stats["blocked_attempts"] / max(stats["attempts"], 1) <= 1.0


def test_graph_features_read_edges_as_pairs():
    index = np.array([[0, 1], [1, 2], [2, 0], [0, 3]])  # 0->1, 1->2, 2->0, 0->3
    pairs = np.array([[0, 2]])
    degrees = np.bincount(index.ravel(), minlength=4).astype(np.float64)
    features = gt3._graph_features(index, 4, pairs, degrees)
    assert features[0, 0] == pytest.approx(np.log1p(degrees[0]))
    assert features[0, 1] == pytest.approx(np.log1p(degrees[2]))
    assert features[0, 2] == pytest.approx(np.log1p(1.0))   # CN(0, 2) = {1}
    assert features[0, 4] == pytest.approx(np.log1p(1.0))   # directed 0 -> 1 -> 2


def test_rewire_refuses_to_recreate_a_forbidden_pair():
    index = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    forbidden = {(0, 5), (5, 0)}
    rewired, stats = gt3.degree_preserving_rewire(index, n=8, forbidden=forbidden, seed=9)
    keys = set(map(tuple, rewired.tolist()))
    assert all((a, b) not in forbidden for a, b in keys)
    assert stats["forbidden_blocked"] >= 0


# ----------------------------------------------------------------- checkpoint

def test_checkpoint_roundtrip_is_bitwise(tmp_path):
    variant = gt3.build_variant("A_embedding", in_dim=16, hidden=32, node_dim=8)
    path = tmp_path / "ckpt.pth"
    gt3.save_checkpoint(path, variant, {"variant": "A_embedding"})
    reloaded = gt3.load_checkpoint(path, variant)
    for key, value in variant.state_dict().items():
        assert torch.equal(value, reloaded[key]), key


# ------------------------------------------------------------------- helpers

def _separable_set(n: int, dim: int):
    """Positives share a node cluster, negatives cross clusters: trivially separable."""
    rng = np.random.default_rng(0)
    groups = 8
    centroid = rng.normal(size=(groups, dim)).astype(np.float32) * 3.0
    label = rng.integers(0, groups, size=n)
    features = torch.from_numpy(
        (centroid[label] + rng.normal(size=(n, dim)) * 0.1).astype(np.float32))
    positives = np.stack([np.arange(n), (np.arange(n) + groups) % n], axis=1).astype(np.int64)
    # same-cluster positives: pick a partner with the same group label
    partner = np.zeros(n, dtype=np.int64)
    for group in range(groups):
        members = np.flatnonzero(label == group)
        partner[members] = np.roll(members, -1)
    positives = np.stack([np.arange(n), partner], axis=1).astype(np.int64)
    negatives = np.zeros(n, dtype=np.int64)
    for group in range(groups):
        members = np.flatnonzero(label == group)
        others = np.flatnonzero(label != group)
        negatives[members] = rng.choice(others, size=len(members))
    keep = positives[:, 0] != positives[:, 1]
    return features, positives[keep], negatives[keep]
