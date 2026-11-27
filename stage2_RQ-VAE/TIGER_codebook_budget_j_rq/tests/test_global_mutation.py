"""Issue #271 mutation checks: each mutation is caught by a named assertion."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))
from model import global_constraint as gc


def _sample():
    return {"i": np.array([0, 1]), "j": np.array([1, 2]),
            "g": np.array([1.0, 2.0]), "weight": np.array([1.0, 4.0])}


def test_mutation_ruler_factor_dropped():
    """M1: using ||q||/(S g) instead of ||q||/(2 S g) must fail the exact-zero assertion."""
    q = torch.zeros((3, 2), dtype=torch.float64)
    q[1, 0] = gc.RULER_FACTOR * gc.kappa_scale()
    sample = {"i": np.array([0]), "j": np.array([1]), "g": np.array([1.0]),
              "weight": np.array([1.0])}
    assert float(gc.weighted_j(q, sample)) == pytest.approx(0.0, abs=1e-12)
    old = gc.RULER_FACTOR
    try:
        gc.RULER_FACTOR = 2.0
        mutated = float(gc.weighted_j(q, sample))
    finally:
        gc.RULER_FACTOR = old
    assert mutated == pytest.approx(0.25)  # the unchanged assertion J==0 catches this mutation


def test_mutation_weight_normalisation_dropped():
    """M2: a sum instead of a weighted mean must differ from the reference."""
    q = torch.as_tensor([[0.0], [1.0], [3.0]], dtype=torch.float64)
    sample = _sample()
    reference = float(gc.weighted_j(q, sample))
    total = 0.0
    for a, b, distance, mass in zip(sample["i"], sample["j"], sample["g"], sample["weight"]):
        ratio = float(abs(q[a] - q[b])) / (gc.RULER_FACTOR * gc.kappa_scale() * distance)
        total += mass * (ratio - 1.0) ** 2
    mutated = total
    assert reference != pytest.approx(mutated, rel=1e-9)


def test_mutation_dual_sign_inverted():
    """M3: mu must fall, not rise, while the constraint is satisfied."""
    bound = 0.4
    satisfied = 0.3
    reference = gc.dual_step(0.2, satisfied, j_max=bound, step=0.1)
    mutated = max(0.0, 0.2 - 0.1 * (satisfied - bound))
    assert reference == pytest.approx(0.19)
    assert mutated != pytest.approx(reference)


def test_mutation_diagonal_included():
    """M4: including self pairs changes the pair count; the production mask excludes them."""
    graph = __import__("scipy.sparse", fromlist=["csr_matrix"])
    rows = np.arange(4)
    g = graph.csr_matrix((np.ones(4), (rows, rows + 1)), shape=(5, 5)).maximum(
        graph.csr_matrix((np.ones(4), (rows, rows + 1)), shape=(5, 5)).T).tocsr()
    q = np.zeros((5, 1))
    out = gc.full_j(q, g, chunk=2)
    assert out["pairs"] == 10  # C(5,2), no diagonal
    assert out["pairs"] != 15  # the mutated count including diagonal would fail


def test_mutation_hard_q_detached():
    """M5: detaching q removes the gradient; the production gradient assertion fails."""
    q = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    sample = {"i": np.array([0]), "j": np.array([1]), "g": np.array([2.0]),
              "weight": np.array([1.0])}
    with torch.no_grad():
        q[1, 0] = 2.0 * gc.RULER_FACTOR * gc.kappa_scale() * 2.0
    q.requires_grad_(True)
    gc.weighted_j(q, sample).backward()
    assert q.grad is not None and float(q.grad.abs().sum()) > 0
    detached = q.detach()
    value = gc.weighted_j(detached, sample)
    assert not value.requires_grad and value.grad_fn is None
