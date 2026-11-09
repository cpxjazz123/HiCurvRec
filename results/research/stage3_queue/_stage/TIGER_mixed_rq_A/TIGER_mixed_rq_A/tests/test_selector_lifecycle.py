"""Transactional tests of the real discrete selector on a small mixed RQ.

These exercise hard assignments, inherited alpha values, the real structural
cache, guard callbacks and commit/rollback. They neither train an arm nor stand
in for the repository's four-GPU training-admission checks.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))

from model import af3_selector as selector  # noqa: E402
from model.af3_engine import StructuralState  # noqa: E402
from model.mixed_rq import AlphaTable, MixedRQ, encode_tokens  # noqa: E402


def _fixture():
    generator = torch.Generator().manual_seed(1)
    rq = MixedRQ([4, 4, 4], AlphaTable(), curvature=1.0).double().eval()
    with torch.no_grad():
        for layer in rq.levels:
            layer.embed.weight.copy_(torch.randn(layer.embed.weight.shape,
                                                 generator=generator,
                                                 dtype=torch.float64) * 0.06)
    z = (torch.randn((9, 32), generator=generator, dtype=torch.float64) * 0.07).requires_grad_()
    pairs = np.array([(i, j) for i in range(9) for j in range(i + 1, 9)], dtype=np.int64)
    observed = {(0, 1), (1, 2), (0, 2), (2, 3), (3, 4),
                (4, 5), (3, 5), (5, 6), (6, 7), (7, 8)}
    is_edge = np.array([tuple(pair) in observed for pair in pairs])
    graph = np.zeros((9, 9), dtype=np.float64)
    for i, j in observed:
        graph[i, j] = graph[j, i] = 1.0
    degree = graph.sum(axis=1)
    kappa = np.array([4.0 - degree[i] - degree[j] + 3.0 * np.dot(graph[i], graph[j])
                      if edge else 0.0 for (i, j), edge in zip(pairs, is_edge)])
    state = StructuralState(pairs, is_edge, kappa, 9, tau=0.8, intercept=0.3, scale=2.0)
    tokens = encode_tokens(z, rq)
    state.full(selector._readout_of(z, tokens, rq))
    return z, rq, state, tokens


def _dense_reference(phi, state):
    """Independent neighbour enumeration, not the engine's sparse update/full."""
    vectors = phi.detach().cpu().numpy()
    n = len(vectors)
    matrix = np.zeros((n, n), dtype=np.float64)
    deltas = []
    for i, j in state.pairs:
        denominator = max(np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]), 1e-8)
        delta = 1.0 - np.dot(vectors[i], vectors[j]) / denominator
        deltas.append(delta)
        matrix[i, j] = matrix[j, i] = 1.0 / (1.0 + np.exp(delta / state.tau - state.intercept))
    triangles = []
    predicted = []
    probability = []
    for i, j in state.pairs:
        others = [k for k in range(n) if k not in (i, j)]
        triangles.append(sum(matrix[i, k] * matrix[j, k] for k in others))
        # Condition on this pair being present before taking the expectation.
        degree_i = 1.0 + sum(matrix[i, k] for k in others)
        degree_j = 1.0 + sum(matrix[j, k] for k in others)
        probability.append(matrix[i, j])
        predicted.append(matrix[i, j] * (4.0 - degree_i - degree_j + 3.0 * triangles[-1]))
    predicted = np.asarray(predicted)
    errors = (predicted - np.where(state.is_edge, state.kappa, 0.0)) / state.scale
    huber = np.where(np.abs(errors) <= 1.0, 0.5 * errors ** 2, np.abs(errors) - 0.5)
    distortion = 0.5 * huber[state.is_edge].mean() + 0.5 * huber[~state.is_edge].mean()
    return {"delta": np.asarray(deltas), "probability": np.asarray(probability),
            "triangle": np.asarray(triangles), "predicted": predicted,
            "distortion": float(distortion)}


def _assert_state_matches_tokens(z, rq, state, tokens):
    expected = _dense_reference(selector._readout_of(z, tokens, rq), state)
    for name, actual in state.state().items():
        np.testing.assert_allclose(actual, expected[name], rtol=0, atol=2e-12,
                                   err_msg="selector left stale " + name)
    np.testing.assert_allclose(state.summary()["distortion"], expected["distortion"],
                               rtol=0, atol=2e-12)


def _choose(z, rq, state, tokens, evaluate, **kwargs):
    return selector.choose_weight(z, rq, rq.alpha_table, state,
                                   level=1, prefix=(), items=np.arange(len(z)),
                                   current_tokens=tokens, start={},
                                   evaluate_feasibility=evaluate, **kwargs)


def test_real_selector_commits_the_best_feasible_trial_including_inherited_weights():
    z, rq, state, incumbent_tokens = _fixture()
    table_before = copy.deepcopy(rq.alpha_table.to_dict())
    parameters_before = [parameter.detach().clone() for parameter in rq.parameters()]
    gradients_before = []
    for parameter in rq.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
        gradients_before.append(parameter.grad.clone())
    trials = {}
    for alpha in (0.25, 0.5, 0.75):
        tokens = encode_tokens(z, rq, override={(1, ()): alpha})
        expected = _dense_reference(selector._readout_of(z, tokens, rq), state)
        trials[alpha] = (tokens.clone(), expected["distortion"])
    best = min(trials, key=lambda value: trials[value][1])
    assert best == 0.75  # Fixed synthetic fixture, not a search over training runs.
    assert trials[best][1] < trials[0.5][1] - 1e-5
    callbacks = []

    def feasible(tokens, outcome):
        assert rq.alpha_table.to_dict() == table_before, "a trial changed the committed table"
        _assert_state_matches_tokens(z, rq, state, tokens)
        callbacks.append(tokens.clone())
        return True, ""

    result = _choose(z, rq, state, incumbent_tokens, feasible)
    assert result["accepted"]["alpha"] == best
    assert len(callbacks) == 2
    committed = encode_tokens(z, rq)
    assert torch.equal(committed, trials[best][0]), "committed codes differ from the evaluated trial"
    assert any(torch.equal(committed, candidate) for candidate in callbacks)
    # At least one item keeps its first token but changes a descendant token:
    # the winning root value must also control inherited deeper assignments.
    assert ((committed[:, 0] == incumbent_tokens[:, 0])
            & (committed[:, 1:] != incumbent_tokens[:, 1:]).any(dim=1)).any()
    for item in committed.tolist():
        assert rq.alpha_table.get(2, (item[0],)) == best
        assert rq.alpha_table.get(3, (item[0], item[1])) == best
    _assert_state_matches_tokens(z, rq, state, committed)
    np.testing.assert_allclose(result["distortion"], trials[best][1], rtol=0, atol=2e-12)
    for parameter, original, gradient in zip(rq.parameters(), parameters_before, gradients_before):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert z.grad is None


def test_infeasible_improvements_restore_every_cached_array_and_do_not_commit():
    z, rq, state, tokens = _fixture()
    table_before = copy.deepcopy(rq.alpha_table.to_dict())
    arrays_before = state.state()
    distortion_before = state.summary()["distortion"]
    callbacks = []

    def infeasible(candidate_tokens, outcome):
        assert outcome["distortion"] < distortion_before
        _assert_state_matches_tokens(z, rq, state, candidate_tokens)
        callbacks.append(candidate_tokens.clone())
        return False, "synthetic reconstruction constraint"

    result = _choose(z, rq, state, tokens, infeasible)
    assert len(callbacks) == 2
    assert result["accepted"] is None
    assert rq.alpha_table.to_dict() == table_before
    assert torch.equal(encode_tokens(z, rq), tokens)
    for name, actual in state.state().items():
        np.testing.assert_allclose(actual, arrays_before[name], rtol=0, atol=2e-12)
    _assert_state_matches_tokens(z, rq, state, tokens)


def test_guard_exception_restores_trial_state_before_it_propagates():
    z, rq, state, tokens = _fixture()
    table_before = copy.deepcopy(rq.alpha_table.to_dict())
    arrays_before = state.state()

    class GuardFailure(RuntimeError):
        pass

    def abort(candidate_tokens, _outcome):
        assert not torch.equal(candidate_tokens, tokens)
        _assert_state_matches_tokens(z, rq, state, candidate_tokens)
        raise GuardFailure("guard execution failed")

    try:
        _choose(z, rq, state, tokens, abort)
    except GuardFailure:
        pass
    else:
        raise AssertionError("selector must propagate a guard execution failure")
    assert rq.alpha_table.to_dict() == table_before
    for name, actual in state.state().items():
        np.testing.assert_allclose(actual, arrays_before[name], rtol=0, atol=2e-12,
                                   err_msg="aborted trial was left in the cache")
    _assert_state_matches_tokens(z, rq, state, tokens)


def test_feasible_candidate_still_needs_a_strict_improvement_beyond_tolerance():
    z, rq, state, tokens = _fixture()
    original_table = copy.deepcopy(rq.alpha_table.to_dict())
    result = _choose(z, rq, state, tokens, lambda *_args: (True, ""), tolerance=1.0)
    assert result["accepted"] is None
    assert rq.alpha_table.to_dict() == original_table
    _assert_state_matches_tokens(z, rq, state, tokens)


def test_nonfinite_candidate_distortions_never_reach_feasibility_or_commit():
    for bad in (float("nan"), float("inf"), -float("inf")):
        z, rq, state, tokens = _fixture()
        table_before = copy.deepcopy(rq.alpha_table.to_dict())
        initial_probability = state.probability.copy()
        real_summary = state.summary
        callbacks = []

        def summary():
            result = real_summary()
            if not np.array_equal(state.probability, initial_probability):
                result["distortion"] = bad
            return result

        def feasible(*args):
            callbacks.append(args)
            return True, ""

        with patch.object(state, "summary", summary):
            result = _choose(z, rq, state, tokens, feasible)
        assert result["accepted"] is None, "a non-finite distortion was accepted"
        assert not callbacks, "invalid objective values must be rejected before consulting guards"
        assert rq.alpha_table.to_dict() == table_before
        _assert_state_matches_tokens(z, rq, state, tokens)


def test_nonfinite_incumbent_fails_closed_without_changing_the_table():
    z, rq, state, tokens = _fixture()
    before = copy.deepcopy(rq.alpha_table.to_dict())
    with patch.object(state, "summary", lambda: {"distortion": float("nan")}):
        try:
            _choose(z, rq, state, tokens, lambda *_args: (True, ""))
        except ValueError:
            pass
        else:
            raise AssertionError("a non-finite incumbent must stop the selector")
    assert rq.alpha_table.to_dict() == before
    _assert_state_matches_tokens(z, rq, state, tokens)


def _check_commit_failure_is_atomic(failure_point):
    z, rq, state, tokens = _fixture()
    table_object = rq.alpha_table
    table_before = copy.deepcopy(table_object.to_dict())
    arrays_before = state.state()
    real_encode = selector.encode_tokens
    real_update = state.update
    failure_raised = False

    class CommitFailure(RuntimeError):
        pass

    def encode(*args, **kwargs):
        nonlocal failure_raised
        if (failure_point == "encode" and not failure_raised
                and table_object.to_dict() != table_before):
            failure_raised = True
            raise CommitFailure("encoding failed after alpha was committed")
        return real_encode(*args, **kwargs)

    def update(*args, **kwargs):
        nonlocal failure_raised
        result = real_update(*args, **kwargs)
        if (failure_point == "update" and not failure_raised
                and table_object.to_dict() != table_before):
            # Failure AFTER mutating the cache is deliberate: rollback must
            # repair both stateful objects, not just their alpha entry.
            failure_raised = True
            raise CommitFailure("structural update failed after mutating the committed cache")
        return result

    with patch.object(selector, "encode_tokens", encode), patch.object(state, "update", update):
        try:
            _choose(z, rq, state, tokens, lambda *_args: (True, ""))
        except CommitFailure:
            pass
        else:
            raise AssertionError("commit execution failures must be propagated")
    assert failure_raised, "fixture never reached the final commit section"
    assert rq.alpha_table is table_object, "rollback disconnected the caller's shared alpha table"
    assert table_object.to_dict() == table_before, "failed commit left a new alpha table installed"
    assert torch.equal(encode_tokens(z, rq), tokens)
    for name, actual in state.state().items():
        np.testing.assert_allclose(actual, arrays_before[name], rtol=0, atol=2e-12,
                                   err_msg="failed commit left candidate " + name + " cached")
    _assert_state_matches_tokens(z, rq, state, tokens)


def test_failure_in_commit_encoding_restores_the_original_alpha_table():
    _check_commit_failure_is_atomic("encode")


def test_failure_after_commit_cache_mutation_restores_both_alpha_and_state():
    _check_commit_failure_is_atomic("update")


def test_resolution_counts_do_not_allocate_the_complete_sid_address_space():
    tokens = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 1, 5],
                           [3, 4, 7], [3, 4, 7], [255, 255, 255]])
    # A six-item diagnostic must not allocate 256**3 int64 bins (~128 MiB).
    # The assertion is a memory contract; all metric expectations below are
    # obtained by counting this fixture's occupied codes, not by bincount.
    with patch.object(torch, "bincount", side_effect=AssertionError(
            "resolution reporting must count occupied SIDs, not the 256**3 address space")):
        result = selector.resolution_report(tokens, 256)
    assert result["used_codes"] == [3, 4, 4]
    assert result["occupied_prefixes"] == [3, 4, 4]
    assert result["max_bucket"] == 2
    assert result["unique_sids"] == 4
    np.testing.assert_allclose(result["excess_collision"], 1.0 / 3.0, rtol=0, atol=1e-15)
