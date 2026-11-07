"""The AF3 surrogate against a brute-force enumeration of every Bernoulli subgraph.

The oracle does not reuse the implementation. On a four-node graph it walks all
``2^6`` subgraphs, computes ``4 - d_i - d_j + 3 T_ij`` for each of them, and
takes the exact expectation under the per-edge Bernoulli measure. The surrogate
must match it for arbitrary probabilities, and with binary probabilities it must
recover the target of the graph itself.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.af3_structure import (  # noqa: E402
    calibrate,
    huber,
    readout_delta,
    readout_probability,
    uncertainty_term,
)

NODES = 4
PAIRS = [(i, j) for i in range(NODES) for j in range(i + 1, NODES)]


def _kappa_of_subgraph(present: set) -> dict:
    """``4 - d_i - d_j + 3 T_ij`` of the subgraph that contains ``present``.

    The degrees are the plain subgraph degrees, which count the edge (i, j)
    itself once at each end: an earlier draft subtracted one from each, which is
    the *conditional* form of the expectation and not the definition of kappa.
    """
    adjacency = {node: set() for node in range(NODES)}
    for i, j in present:
        adjacency[i].add(j)
        adjacency[j].add(i)
    values = {}
    for i, j in present:
        triangles = len(adjacency[i] & adjacency[j])
        values[(i, j)] = 4 - len(adjacency[i]) - len(adjacency[j]) + 3 * triangles
    return values


def _exact_expectation(probabilities: dict) -> dict:
    """The expectation of the edge-weighted target over every subgraph."""
    totals = {pair: 0.0 for pair in PAIRS}
    for mask in range(1 << len(PAIRS)):
        weight = 1.0
        present = set()
        for bit, pair in enumerate(PAIRS):
            p = probabilities[pair]
            if mask >> bit & 1:
                weight *= p
                present.add(pair)
            else:
                weight *= 1.0 - p
        if weight == 0.0:
            continue
        values = _kappa_of_subgraph(present)
        for pair in PAIRS:
            # The target is edge-weighted: an absent edge contributes nothing,
            # which is the definition of mu = A_ij * kappa_ij.
            totals[pair] += weight * values.get(pair, 0.0)
    return totals


def _surrogate(probabilities: dict) -> dict:
    pairs = np.array(PAIRS, dtype=np.int64)
    p = np.array([probabilities[pair] for pair in PAIRS], dtype=np.float64)
    term = uncertainty_term(pairs, p, NODES)
    return {pair: probabilities[pair] * term[index] for index, pair in enumerate(PAIRS)}


def test_the_surrogate_matches_the_exact_expectation() -> str:
    """Arbitrary probabilities: the closed form equals the enumeration."""
    rng = np.random.default_rng(7)
    for trial in range(3):
        probabilities = {pair: float(rng.uniform(0.05, 0.95)) for pair in PAIRS}
        exact = _exact_expectation(probabilities)
        surrogate = _surrogate(probabilities)
        for pair in PAIRS:
            if not math.isclose(exact[pair], surrogate[pair], rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError(
                    f"trial {trial} pair {pair}: exact {exact[pair]!r} "
                    f"surrogate {surrogate[pair]!r}")
    return "the expectation surrogate matches the enumeration for all pairs"


def test_binary_probabilities_recover_the_graph_target() -> str:
    """With p in {0, 1} the surrogate is the graph's own ``kappa``."""
    present = {PAIRS[0], PAIRS[3], PAIRS[4]}
    probabilities = {pair: (1.0 if pair in present else 0.0) for pair in PAIRS}
    target = _kappa_of_subgraph(present)
    surrogate = _surrogate(probabilities)
    for pair in PAIRS:
        want = target.get(pair, 0.0)
        if not math.isclose(want, surrogate[pair], rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"pair {pair}: target {want!r} surrogate {surrogate[pair]!r}")
    return "binary probabilities recover 4 - d_i - d_j + 3 T_ij exactly"


def test_the_huber_is_the_specified_two_piece_function() -> str:
    """``Huber_1``: half the square inside, the absolute value minus a half outside."""
    values = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    got = huber(values)
    want = np.array([2.5, 0.5, 0.125, 0.0, 0.125, 0.5, 2.5])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    return "the Huber matches its definition at the join and outside it"


def test_the_readout_is_a_decreasing_function_of_the_cosine_gap() -> str:
    """A larger gap must lower the predicted edge probability."""
    import torch
    delta = torch.linspace(0.0, 2.0, 11, dtype=torch.float64)
    probability = readout_probability(delta, tau=0.3, intercept=-1.0).numpy()
    if not np.all(np.diff(probability) < 0):
        raise AssertionError("the readout is not strictly decreasing in the gap")
    if not math.isclose(float(probability[0]), float(1.0 / (1.0 + math.exp(-(-1.0)))), rel_tol=1e-9):
        raise AssertionError("the readout does not start at sigmoid(intercept)")
    return "the readout falls strictly as the cosine gap grows"


def test_the_calibration_matches_the_edge_density() -> str:
    """The calibrated intercept makes the mean probability equal ``|E|/|S|``."""
    import torch

    generator = torch.Generator().manual_seed(11)
    n_nodes = 40
    phi = torch.randn(n_nodes, 32, generator=generator)
    phi[1::2] = phi[0::2] * 0.9 + 0.1 * phi[1::2]
    pairs = np.array([[i, j] for i in range(n_nodes) for j in range(i + 1, n_nodes)][:120],
                     dtype=np.int64)
    is_edge = np.zeros(len(pairs), dtype=bool)
    is_edge[:24] = True
    support = {"pairs": pairs, "is_edge": is_edge, "edges": int(is_edge.sum())}
    fitted = calibrate(phi, support)
    if abs(fitted["mean_probability"] - fitted["density"]) > 1e-6:
        raise AssertionError("the calibrated mean does not match the edge density: "
                             + str(fitted))
    delta = readout_delta(phi, pairs)
    probability = readout_probability(delta, fitted["tau"], fitted["intercept"]).mean()
    if abs(float(probability) - fitted["density"]) > 1e-6:
        raise AssertionError("the saved parameters do not reproduce the calibration")
    return "the calibrated readout reproduces the true edge density"


def main() -> int:
    failures = 0
    for test in (
        test_the_surrogate_matches_the_exact_expectation,
        test_binary_probabilities_recover_the_graph_target,
        test_the_huber_is_the_specified_two_piece_function,
        test_the_readout_is_a_decreasing_function_of_the_cosine_gap,
        test_the_calibration_matches_the_edge_density,
    ):
        try:
            print("PASS " + test.__name__ + ": " + test())
        except AssertionError as exc:
            failures += 1
            print("FAIL " + test.__name__ + ": " + str(exc))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("ERROR " + test.__name__ + ": " + type(exc).__name__ + ": " + str(exc))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
