"""The four QDA metrics must measure what the report claims they measure.

Each case pairs a hand-computable answer with the implementation, and each case
is chosen so that a plausible wrong implementation fails it: an AUC that ignores
ties, a lift that normalises by the wrong base rate, a CKA that normalises by the
wrong norm, and a rank proxy that is not scale-free.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TREE = HERE / "trees" / "F0_seed42"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze = _load("qda_analyze", HERE / "analyze.py")
observer = _load("qda_observer", TREE / "qda_observer.py")

# An identity map so `pair_distance` reduces to the Euclidean half plus zero ball
# term: the metric under test stays the real one, only its geometry is trivial.
IDENTITY = (lambda x, c: x)


def _squared_distance(a, b, c):
    import torch
    return ((a - b) ** 2).sum(-1)


def _sample(edges, negatives):
    return {"edges": np.asarray(edges, dtype=np.int32),
            "negatives": np.asarray(negatives, dtype=np.int32)}


def test_auc_counts_ties_as_one_half():
    """A metric that never separates anything must score 0.5, not 0 or 1."""
    vectors = np.zeros((4, 16), dtype=np.float32)
    sample = _sample([[0, 1]], [[2, 3]])
    mask = np.array([True])
    # `expmap0` is never called on this path: the identity map plus a pure
    # Euclidean norm leaves only the Euclidean half of the mixed metric.
    score = analyze.edge_auc(vectors, sample, mask, IDENTITY, _squared_distance)
    assert score == 0.5, score


def test_auc_is_one_when_every_edge_is_closer_than_its_non_edge():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(8, 16)).astype(np.float32)
    # Edges join near-identical points, non-edges join far-apart ones.
    vectors[1] = vectors[0]
    vectors[3] = -vectors[2]
    sample = _sample([[0, 1]], [[2, 3]])
    score = analyze.edge_auc(vectors, sample, np.array([True]), IDENTITY, _squared_distance)
    assert score == 1.0, score


def test_prefix_lift_is_the_difference_of_two_rates():
    codes = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [2, 0, 0]], dtype=np.int64)
    # one edge agreeing on L1, one non-edge disagreeing at L1 but agreeing at L2
    sample = _sample([[0, 1]], [[0, 2]])
    lift = analyze.prefix_lift(codes, sample, np.array([True]), level=1)
    assert lift["positive_share"] == 1.0
    assert lift["negative_share"] == 0.0
    assert lift["lift"] == 1.0
    # Prefix agreement is monotone in depth: agreeing on the first two tokens
    # forces agreement on the first one, so deeper levels can only lose edges.
    lift2 = analyze.prefix_lift(codes, sample, np.array([True]), level=2)
    assert lift2["positive_share"] <= lift["positive_share"]
    assert lift2["negative_share"] <= lift["negative_share"]


def test_prefix_lift_drops_when_the_deeper_token_disagrees():
    codes = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]], dtype=np.int64)
    sample = _sample([[0, 1]], [[2, 3]])
    # All four share the first token, so L1 cannot separate edge from non-edge.
    assert analyze.prefix_lift(codes, sample, np.array([True]), level=1)["lift"] == 0.0
    # At L2 every pair differs, so there is nothing left to share either.
    assert analyze.prefix_lift(codes, sample, np.array([True]), level=2)["lift"] == 0.0
    # A pair that agrees at L2 lifts there while the others stay flat.
    codes[1] = [0, 1, 9]
    assert analyze.prefix_lift(codes, sample, np.array([True]), level=2)["lift"] == 1.0


def test_cka_is_one_for_a_rotation_and_zero_for_an_independent_sample():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert abs(analyze.linear_cka(x, x @ rotation) - 1.0) < 1e-9
    independent = rng.normal(size=(200, 8))
    assert abs(analyze.linear_cka(x, independent)) < 0.05


def test_effective_rank_counts_rank_and_ignores_scale():
    """exp(entropy of the squared spectrum): k exactly when the spectrum is flat,
    and strictly below k when it is not — that is the point of the proxy."""
    rng = np.random.default_rng(2)
    basis = np.linalg.qr(rng.normal(size=(100, 5)))[0]
    flat = basis * 2.0
    assert abs(observer._effective_rank(flat) - 5.0) < 1e-6
    # Same directions, different magnitudes: the proxy is scale-free.
    skewed = basis * np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(observer._effective_rank(skewed)
               - observer._effective_rank(skewed * 1e6)) < 1e-12
    assert abs(observer._effective_rank(skewed) - 5.0) > 0.1
    # A single direction is rank one regardless of magnitude.
    assert abs(observer._effective_rank(np.ones((100, 1)) * 3.0) - 1.0) < 1e-9
