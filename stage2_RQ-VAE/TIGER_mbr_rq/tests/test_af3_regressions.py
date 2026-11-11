"""Regressions for the Issue #139 structural cache and acceptance guards.

The cache oracle below uses a small dense probability table and explicit
neighbour sums. It never calls the sparse engine or af3_structure's arithmetic.
The trainer check executes only the outer-loop callback on four synthetic items;
it does not start a trainer, fit an arm, or supply training-admission evidence.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))

from model.af3_engine import StructuralState  # noqa: E402
from model import af3_selector  # noqa: E402


def _dense_oracle(phi, pairs, is_edge, target, tau, intercept, scale):
    vectors = phi.detach().cpu().numpy().astype(np.float64)
    n_nodes = len(vectors)
    delta = []
    probability = []
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, j in pairs:
        denominator = max(np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]), 1e-8)
        distance = 1.0 - np.dot(vectors[i], vectors[j]) / denominator
        value = 1.0 / (1.0 + np.exp(-(intercept - distance / tau)))
        delta.append(distance)
        probability.append(value)
        matrix[i, j] = matrix[j, i] = value
    triangles = []
    predicted = []
    for position, (i, j) in enumerate(pairs):
        partners = [k for k in range(n_nodes) if k not in (i, j)]
        triangle = sum(matrix[i, k] * matrix[j, k] for k in partners)
        # Conditional on the pair being present, its two endpoints each
        # already have degree one. All other supported edges remain Bernoulli.
        conditional_degree_i = 1.0 + sum(matrix[i, k] for k in partners)
        conditional_degree_j = 1.0 + sum(matrix[j, k] for k in partners)
        predicted.append(probability[position] * (
            4.0 - conditional_degree_i - conditional_degree_j + 3.0 * triangle))
        triangles.append(triangle)
    predicted = np.asarray(predicted)
    error = (predicted - np.where(is_edge, target, 0.0)) / scale
    absolute = np.abs(error)
    loss = np.where(absolute < 1.0, 0.5 * error * error, absolute - 0.5)
    distortion = 0.5 * loss[is_edge].mean() + 0.5 * loss[~is_edge].mean()
    return {"delta": np.asarray(delta), "probability": np.asarray(probability),
            "triangle": np.asarray(triangles), "predicted": predicted,
            "distortion": float(distortion)}


def _assert_dense_match(state, phi):
    reference = _dense_oracle(phi, state.pairs, state.is_edge, state.kappa,
                              state.tau, state.intercept, state.scale)
    for name, actual in state.state().items():
        np.testing.assert_allclose(actual, reference[name], rtol=0, atol=2e-12,
                                   err_msg="incremental cache mismatch: " + name)
    np.testing.assert_allclose(state.summary()["distortion"], reference["distortion"],
                               rtol=0, atol=2e-12)


def test_changed_degree_reaches_an_edge_outside_the_inner_neighbourhood():
    """Changing node 0 on support 0--1--2 must also update mu(1,2)."""
    pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
    state = StructuralState(pairs, np.array([True, False]), np.array([2.0, 0.0]),
                            3, tau=0.4, intercept=0.3, scale=1.0)
    phi = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=torch.float64)
    state.full(phi)
    before = state.state()
    phi[0] = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    state.update(phi, np.array([0], dtype=np.int64))
    # Edge 1--2's own probability and triangle count have NOT changed;
    # only the degree of its endpoint 1 has changed through edge 0--1.
    np.testing.assert_allclose(state.probability[1], before["probability"][1], atol=1e-15)
    expected = state.probability[1] * (2.0 - state.probability[0])
    assert abs(expected - before["predicted"][1]) > 1e-3
    np.testing.assert_allclose(state.predicted[1], expected, rtol=0, atol=2e-12)
    _assert_dense_match(state, phi)


def test_repeated_updates_and_rejected_candidate_rollbacks_match_dense_oracle():
    rng = np.random.default_rng(139)
    n_nodes = 18
    candidates = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    selected = rng.choice(len(candidates), size=51, replace=False)
    pairs = np.asarray(sorted(candidates[i] for i in selected), dtype=np.int64)
    is_edge = np.arange(len(pairs)) % 3 == 0
    observed = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, j in pairs[is_edge]:
        observed[i, j] = observed[j, i] = 1.0
    degree = observed.sum(axis=1)
    target = np.zeros(len(pairs), dtype=np.float64)
    for position, (i, j) in enumerate(pairs):
        if is_edge[position]:
            target[position] = 4.0 - degree[i] - degree[j] + 3.0 * np.dot(observed[i], observed[j])
    state = StructuralState(pairs, is_edge, target, n_nodes,
                            tau=0.7, intercept=0.2, scale=2.0)
    phi = torch.as_tensor(rng.normal(size=(n_nodes, 7)), dtype=torch.float64)
    state.full(phi)
    _assert_dense_match(state, phi)
    for step in range(12):
        original_phi = phi.clone()
        original_state = state.state()
        original_distortion = state.summary()["distortion"]
        changed = np.sort(rng.choice(n_nodes, size=1 + step % 4, replace=False))
        phi[changed] = torch.as_tensor(rng.normal(size=(len(changed), 7)), dtype=torch.float64)
        state.update(phi, changed)
        _assert_dense_match(state, phi)
        if step % 2 == 0:
            # A rejected candidate has to restore the whole dependency closure,
            # not just the probabilities on pairs incident to the changed items.
            phi = original_phi
            state.update(phi, changed)
            _assert_dense_match(state, phi)
            for name, actual in state.state().items():
                np.testing.assert_allclose(actual, original_state[name], rtol=0, atol=2e-12)
            np.testing.assert_allclose(state.summary()["distortion"], original_distortion,
                                       rtol=0, atol=2e-12)
    before = state.state()
    state.update(phi, np.array([], dtype=np.int64))
    for name, actual in state.state().items():
        np.testing.assert_array_equal(actual, before[name])


def _guard_fixture():
    pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 3]], dtype=np.int64)
    is_edge = np.array([True, False, False, False])
    original_degree = np.array([1.0, 1.0, 0.0, 0.0])
    probability = np.array([0.8, 0.2, 0.4, 0.6])
    return probability, pairs, is_edge, original_degree


def test_structural_guards_use_class_balance_and_original_node_degrees():
    probability, pairs, is_edge, degree = _guard_fixture()
    result = af3_selector.structural_guard_metrics(probability, pairs, is_edge, degree)
    # Brier = .5*(.8-1)^2 + .5*(.2^2+.4^2+.6^2)/3 = 17/150.
    np.testing.assert_allclose(result["adjacency_brier"], 17.0 / 150.0, rtol=0, atol=1e-14)
    # Soft degrees [1.4,1.4,.2,1.0]; original degrees [1,1,0,0].
    # Mean absolute node error is .5 and mean original node degree is .5.
    np.testing.assert_allclose(result["degree_error"], 1.0, rtol=0, atol=1e-14)
    candidate_probability = np.array([0.8, 0.9, 0.9, 0.9])
    candidate = af3_selector.structural_guard_metrics(candidate_probability, pairs, is_edge, degree)
    np.testing.assert_allclose(candidate["degree_error"], 2.5, rtol=0, atol=1e-14)
    assert candidate["degree_error"] != result["degree_error"]


def test_structural_guards_reject_nonfinite_or_misaligned_probability_vectors():
    probability, pairs, is_edge, degree = _guard_fixture()
    invalid_probabilities = [probability[:-1], probability.reshape(2, 2)]
    for bad in (float("nan"), float("inf"), -0.1, 1.1):
        values = probability.copy()
        values[0] = bad
        invalid_probabilities.append(values)
    for values in invalid_probabilities:
        try:
            af3_selector.structural_guard_metrics(values, pairs, is_edge, degree)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid probabilities must be rejected, not used as guard evidence")


def _valid_feasibility_inputs():
    resolution = {"used_codes": [4, 4, 4], "occupied_prefixes": [4, 8, 12],
                  "excess_collision": 0.05, "max_bucket": 2}
    start = {"semantic": 1.0, "adjacency_brier": 0.2, "degree_error": 0.1,
             "resolution": resolution}
    return start, copy.deepcopy(resolution)


def test_feasibility_rejects_nonfinite_metrics_and_resolution_values():
    # NaN comparisons and negative infinity must not silently pass guards.
    for metric in ("semantic", "adjacency_brier", "degree_error"):
        for bad in (float("nan"), float("inf"), -float("inf")):
            start, candidate = _valid_feasibility_inputs()
            values = {"semantic": 1.0, "adjacency_brier": 0.2, "degree_error": 0.1}
            values[metric] = bad
            accepted, reason = af3_selector.within_feasible_region(start, candidate, **values)
            assert not accepted, ("invalid candidate metric accepted", metric, bad)
            assert reason
            start[metric] = bad
            accepted, reason = af3_selector.within_feasible_region(start, candidate, 1.0, 0.2, 0.1)
            assert not accepted, ("invalid start metric accepted", metric, bad)
            assert reason
    for field in ("excess_collision", "max_bucket", "used_codes", "occupied_prefixes"):
        for side in ("start", "candidate"):
            start, candidate = _valid_feasibility_inputs()
            resolution = start["resolution"] if side == "start" else candidate
            if isinstance(resolution[field], list):
                resolution[field][1] = float("nan")
            else:
                resolution[field] = float("nan")
            accepted, reason = af3_selector.within_feasible_region(start, candidate, 1.0, 0.2, 0.1)
            assert not accepted, ("invalid resolution accepted", side, field)
            assert reason


def test_feasibility_rejects_degree_only_regression_against_round_start():
    start, candidate = _valid_feasibility_inputs()
    original = copy.deepcopy(start)
    accepted, _ = af3_selector.within_feasible_region(start, candidate, 1.0, 0.2, 0.102)
    assert not accepted
    assert start == original
    accepted, _ = af3_selector.within_feasible_region(start, candidate, 1.0, 0.2, 0.1005)
    assert accepted


def test_outer_loop_evaluates_start_and_candidate_with_the_same_guard_definitions():
    """Exercise the real trainer callback without running a training entry point."""
    probability, pairs, is_edge, original_degree = _guard_fixture()
    candidate_probability = np.array([0.8, 0.9, 0.9, 0.9])
    spec = importlib.util.spec_from_file_location("issue139_guard_trainer_probe", TREE / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    class FakeState:
        n_nodes = 4

        def __init__(self):
            self.pairs = pairs
            self.is_edge = is_edge
            self.probability = probability.copy()
            # Deliberately incompatible with probabilities: the old trainer
            # mistook the AF3 expectation for an adjacency probability.
            self.predicted = np.full(len(pairs), 17.0)

        def full(self, _phi):
            self.probability = probability.copy()
            return self.summary()

        def summary(self):
            return {"distortion": 1.0}

    state = FakeState()
    tokens = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]])
    from model.mixed_rq import AlphaTable
    rq = SimpleNamespace(levels=[SimpleNamespace(n_embed=2) for _ in range(3)],
                         alpha_table=AlphaTable())
    raw = SimpleNamespace(encoder=lambda x: x, decoder=lambda x: x,
                          rq=rq, _structure_state=state)
    graph = {"n_nodes": 4, "rows": np.array([0]), "cols": np.array([1]),
             "item_ids": np.arange(4, dtype=np.int64)}
    support = {"pairs": pairs, "is_edge": is_edge}
    bundle = (graph, support, np.array([2.0, 0.0, 0.0, 0.0]))
    features = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    captured = []

    def capture_guard(start, _candidate, semantic, adjacency_brier, degree_error):
        captured.append((copy.deepcopy(start), adjacency_brier, degree_error))
        return False, "fixture rejects the candidate"

    def propose_candidate(*_args, **kwargs):
        state.probability = candidate_probability.copy()
        kwargs["evaluate_feasibility"](tokens, {"distortion": 0.5})
        state.probability = probability.copy()
        return {"accepted": None, "attempts": [], "distortion": 1.0}

    from model import mixed_rq
    with patch.object(trainer, "ARM_VARIANT", "L"), \
            patch.object(af3_selector, "choose_weight", propose_candidate), \
            patch.object(af3_selector, "within_feasible_region", capture_guard), \
            patch.object(af3_selector, "_readout_of", lambda z, *_args: z), \
            patch.object(mixed_rq, "encode_tokens", lambda *_args, **_kwargs: tokens.clone()):
        trainer.run_outer_loop(1500, raw, features, features, torch.device("cpu"),
                               bundle, {"tau": 0.5, "intercept": 0.0},
                               lambda *_args, **_kwargs: None)
    assert len(captured) == 1
    start, candidate_brier, candidate_degree_error = captured[0]
    np.testing.assert_allclose(start["adjacency_brier"], 17.0 / 150.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(start["degree_error"], 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(candidate_brier, 0.425, rtol=0, atol=1e-14)
    np.testing.assert_allclose(candidate_degree_error, 2.5, rtol=0, atol=1e-14)
