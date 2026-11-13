"""`af3_potential.py`: the four formulas that turn the training graph into `s_i`.

Every expected number below is computed by hand from math.json, not by calling the
function twice:

    kappa_ij = 4 - d_i - d_j + 3 T_ij
    s_i      = mean_{j in N(i)} kappa_ij            (d_i = 0 -> graph mean)
    s_i     <- (s_i - mu_s) / max(sigma_s, eps)     (sigma_s = 0 -> 0)
    H_delta  = 0.5 x^2  if |x| <= delta, else delta (|x| - 0.5 delta)

The tiny graph is

    0 - 1 - 2 - 3        and item 4 is isolated

so the degrees are d = (1, 2, 2, 1, 0), every edge of the path has one common
neighbour (T = 1), and the edge curvatures are

    (0,1): 4 - 1 - 2 + 3 = 4
    (1,2): 4 - 2 - 2 + 3 = 3
    (2,3): 4 - 2 - 1 + 3 = 4
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import af3_potential as af3


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "history": [[0, 1], [2], [1, 2], [3]],
            "target": [2, 3, 3, 3],
        }
    )


def _graph(num_items: int = 5):
    items = af3.sequences(_frame())
    rows, cols = af3.edge_arrays(items)
    matrix = af3.adjacency(rows, cols, num_items)
    curvature = af3.edge_curvatures(matrix, rows, cols)
    return matrix, rows, cols, curvature


def test_edge_arrays_drop_self_loops_and_duplicates():
    items = [np.asarray([0, 0, 1], dtype=np.int64), np.asarray([1, 0], dtype=np.int64)]
    rows, cols = af3.edge_arrays(items)
    assert len(rows) == 1 and len(cols) == 1
    assert (int(rows[0]), int(cols[0])) == (0, 1)


def test_sequences_append_target_and_skip_degenerate_rows():
    items = af3.sequences(_frame())
    assert [list(map(int, sequence)) for sequence in items] == [[0, 1, 2], [2, 3], [1, 2, 3], [3, 3]]
    # the last row is `[3, 3]`: one item twice, so it contributes no edge after the
    # self-loop filter
    assert af3.sequences(pd.DataFrame({"history": [[]], "target": [0]})) == []


def test_curvature_formula_on_the_tiny_graph():
    matrix, rows, cols, curvature = _graph()
    degrees = np.asarray(matrix.sum(1)).ravel()
    assert degrees.tolist() == [1.0, 2.0, 2.0, 1.0, 0.0]
    edges = {(int(a), int(b)): float(k) for a, b, k in zip(rows, cols, curvature)}
    assert edges == {(0, 1): 4.0, (1, 2): 3.0, (2, 3): 4.0}


def test_structural_potential_is_the_incident_mean():
    matrix, rows, cols, curvature = _graph()
    potential = af3.structural_potentials(matrix, rows, cols, curvature)
    # item 0 has one neighbour (kappa 4); item 1 averages 4 and 3; item 3 has one
    # neighbour (kappa 4); item 4 has none and takes the mean of the four above.
    assert potential[:4].tolist() == pytest.approx([4.0, 3.5, 3.5, 4.0])
    assert float(potential[4]) == pytest.approx(3.75)


def test_degree_zero_uses_the_mean_of_the_connected_items_not_the_including_zero():
    matrix, rows, cols, curvature = _graph()
    potential = af3.structural_potentials(matrix, rows, cols, curvature)
    assert float(potential.mean()) == pytest.approx(3.75)
    # if the isolated item had been folded in as 0 before the mean, the value would be
    # 3.0 -- the formula says "mean over the graph", not "mean including absent items"
    assert float(potential[4]) != pytest.approx(3.0)


def test_standardisation_uses_the_population_sigma():
    values = np.asarray([4.0, 3.5, 3.5, 4.0, 3.75])
    result = af3.standardise(values)
    mean = values.mean()
    sigma = values.std(ddof=0)          # the convention this tree fixed
    assert sigma == pytest.approx(0.223606797749979, rel=1e-12)
    assert result.tolist() == pytest.approx(((values - mean) / sigma).tolist())
    assert float(result.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(result.std(ddof=0)) == pytest.approx(1.0, rel=1e-12)


def test_constant_potential_standardises_to_exact_zeros():
    # sigma_s = 0 is the degenerate case: the formula returns 0, not a division
    result = af3.standardise(np.full(7, 2.5))
    assert np.array_equal(result, np.zeros(7))
    assert not np.isnan(result).any()


def test_huber_is_quadratic_inside_linear_outside_and_continuous_at_delta():
    x = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    assert af3.huber(x).tolist() == pytest.approx([1.5, 0.5, 0.125, 0.0, 0.125, 0.5, 1.5])
    # |x| == delta belongs to the quadratic branch and the two branches agree there
    assert float(af3.huber(np.asarray([1.0, -1.0]))[0]) == pytest.approx(0.5)
    # outside, the slope is delta = 1 (linear), not x
    assert float(af3.huber(np.asarray([3.0]))[0]) == pytest.approx(2.5)


def test_huber_tensor_matches_the_array_version():
    x = torch.tensor([-3.0, -0.25, 0.0, 0.75, 4.0])
    assert af3.huber(x).tolist() == pytest.approx(af3.huber(x.numpy()).tolist())
    assert af3.huber(x).dtype == x.dtype


def test_calibrate_lambda_divides_two_medians():
    den, scale = af3.calibrate_lambda(
        distances_to_assigned=np.asarray([1.0, 3.0]),
        potentials=np.asarray([0.0, 2.0]),
        tokens=np.asarray([0, 1]),
        assigned_potentials=np.asarray([0.0, 1.0]),
    )
    # residual = [0, 1] -> median 0.5; distances median 2 -> lambda = 4
    assert den == pytest.approx(0.5)
    assert scale == pytest.approx(4.0)


def test_calibrate_lambda_is_zero_when_the_denominator_vanishes():
    den, scale = af3.calibrate_lambda(
        distances_to_assigned=np.asarray([2.0, 2.0]),
        potentials=np.asarray([1.0, 1.0]),
        tokens=np.asarray([0, 0]),
        assigned_potentials=np.asarray([1.0, 1.0]),
    )
    assert den == pytest.approx(0.0)
    assert scale == 0.0


def test_build_returns_the_standardised_table_and_the_metrics():
    table = af3.build(_frame(), num_items=5)
    assert table["num_items"] == 5
    expected = af3.standardise(table["raw_potential"])
    assert table["potential"].tolist() == pytest.approx(expected.tolist())
    assert table["metrics"]["graph_edges"] == 3
    assert table["metrics"]["graph_nodes_with_edges"] == 4
    tensor = af3.potential_tensor(table)
    assert tensor.shape == (5,) and tensor.dtype == torch.float32


def test_the_frozen_table_is_finite_and_unit_scaled():
    table = af3.build(_frame(), num_items=5)
    assert np.isfinite(table["potential"]).all()
    assert float(np.abs(table["potential"]).max()) > 0.0
