"""Issue #271: deterministic and independent-reference checks for the global ruler."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))
from model import global_constraint as gc
from model.stereographic import squared as stereo_squared


def _path_graph(n: int = 6):
    import scipy.sparse as sp

    rows = np.arange(n - 1)
    graph = sp.csr_matrix((np.ones(n - 1), (rows, rows + 1)), shape=(n, n))
    return graph.maximum(graph.T).tocsr()


def test_weighted_j_exact_zero_and_one():
    """Hand-chosen coordinates: ratio exactly 1 gives J=0, ratio 2 gives J=1."""
    scale = gc.kappa_scale()
    factor = gc.RULER_FACTOR
    q = torch.zeros((2, 2), dtype=torch.float64)
    g = 3.0
    q[1, 0] = factor * scale * g
    sample = {"i": np.array([0]), "j": np.array([1]), "g": np.array([g]),
              "weight": np.array([1.0])}
    assert float(gc.weighted_j(q, sample)) == pytest.approx(0.0, abs=1e-12)
    q[1, 0] *= 2.0
    assert float(gc.weighted_j(q, sample)) == pytest.approx(1.0, abs=1e-12)


def test_weighted_j_reference_loop():
    rng = np.random.default_rng(7)
    q = torch.as_tensor(rng.normal(size=(8, 3)))
    i = np.array([0, 1, 2, 3, 4, 5, 6])
    j = np.array([1, 2, 3, 4, 5, 6, 7])
    g = rng.uniform(1, 4, size=7)
    w = rng.uniform(0.5, 3.0, size=7)
    got = float(gc.weighted_j(q, {"i": i, "j": j, "g": g, "weight": w}))
    total = 0.0
    weight = 0.0
    for a, b, distance, mass in zip(i, j, g, w):
        delta = q[a] - q[b]
        length = float(torch.sqrt((delta * delta).sum()))
        ratio = length / (gc.RULER_FACTOR * gc.kappa_scale() * distance)
        total += mass * (ratio - 1.0) ** 2
        weight += mass
    assert got == pytest.approx(total / weight, rel=1e-12)


def test_kappa_zero_limit():
    """The common ruler is the kappa=0 member of the #269 w=q/(2S) map."""
    q = torch.as_tensor([[0.0, 0.0], [0.3, -0.2]], dtype=torch.float64)
    w = q / (2.0 * gc.kappa_scale())
    zero = stereo_squared(w[0][None, :], w[1][None, :], torch.zeros(1, dtype=torch.float64))
    direct = 4.0 * (((q[0] - q[1]) / (2.0 * gc.kappa_scale())) ** 2).sum()
    assert float(zero.reshape(-1)[0]) == pytest.approx(float(direct), rel=1e-12)
    assert float(torch.sqrt(zero.reshape(-1)[0])) == pytest.approx(
        float(torch.linalg.vector_norm(q[0] - q[1]) / gc.kappa_scale()), rel=1e-12)


def test_full_j_known_answer_and_masks():
    graph = _path_graph(3)
    scale = gc.kappa_scale()
    q = np.zeros((3, 2))
    q[1, 0] = gc.RULER_FACTOR * scale
    q[2, 0] = 2 * gc.RULER_FACTOR * scale
    out = gc.full_j(q, graph, chunk=2)
    assert out["pairs"] == 3
    assert out["mean_j"] == pytest.approx(0.0, abs=1e-12)
    assert set(out["strata"]) == {"1-1", "2-2", "3-3", "4-5", "6-inf"}


def test_pair_table_is_deterministic_and_weighted():
    graph = _path_graph(6)
    first, receipt = gc.build_pair_table(graph, centres=4, seed=11)
    second, _ = gc.build_pair_table(graph, centres=4, seed=11)
    for key in ("i", "j", "g", "weight"):
        assert np.array_equal(first[key], second[key])
    assert np.all(first["g"] > 0)
    assert np.all(first["weight"] >= 1)
    assert receipt["pairs"] == len(first["i"])
    assert len({(int(a), int(b)) for a, b in zip(first["i"], first["j"])}) == len(first["i"])


def test_dual_step_sign_and_cap():
    bound = 0.4
    assert gc.dual_step(0.2, 0.5, j_max=bound, step=0.1) == pytest.approx(0.21)
    assert gc.dual_step(0.2, 0.3, j_max=bound, step=0.1) == pytest.approx(0.19)
    assert gc.dual_step(0.01, 0.0, j_max=bound, step=1.0) == 0.0
    assert gc.dual_step(0.0, 10.0, j_max=bound, step=1.0, cap=2.0) <= 2.0


def test_gradient_reaches_q():
    scale = gc.kappa_scale()
    q = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    sample = {"i": np.array([0]), "j": np.array([1]), "g": np.array([2.0]),
              "weight": np.array([1.0])}
    value = gc.weighted_j(q, sample)
    value.backward()
    assert q.grad is not None
    assert bool(torch.isfinite(q.grad).all())
    assert float(q.grad.abs().sum()) == 0.0  # coincident points: finite zero subgradient
    q.grad = None
    with torch.no_grad():
        q[1, 0] = 2.0 * gc.RULER_FACTOR * scale * 2.0
    q.requires_grad_(True)
    q2 = q.clone().detach().requires_grad_(True)
    gc.weighted_j(q2, sample).backward()
    assert q2.grad is not None and float(q2.grad.abs().sum()) > 0
    detached = q.detach()
    assert detached.grad is None
    assert gc.RULER_FACTOR == 1.0
    assert scale > 0
