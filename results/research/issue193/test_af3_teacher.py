"""P0-2 of issue #193: the AF3 teacher's triangle count, pinned by graph theory rather than by the implementation.

``AF3(i, j) = 4 - deg(i) - deg(j) + 3 m_ij`` with ``m_ij`` the number of common neighbours.  The shipped module added the constant 3 instead, which is
``3 m_ij`` only when the edge sits in exactly one triangle; on a triangle-free graph it inflates every edge by three, and because the same constant
entered both clipping quantiles it cancelled in ``W = 1 + clip((AF3 - q05)/(q95 - q05), 0, 1)``, leaving a weight that is a function of the degree sum
alone.

``AF3_TARGET=shipped`` points these tests at the arm's own module so the defect is reproduced (red); the default points them at
``af3_teacher_fixed.py`` (green).  Each case is stated with the truth value computed by hand from the graph, never taken from either implementation.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SHIPPED = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_af3_structdistill_rq" / "model" / "af3_teacher.py"
FIXED = HERE / "af3_teacher_fixed.py"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"af3_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TARGET = os.environ.get("AF3_TARGET", "fixed").strip().lower()
MODULE = load_module(SHIPPED if TARGET == "shipped" else FIXED)

CYCLE4 = [(0, 1), (1, 2), (2, 3), (3, 0)]          # no triangle anywhere
TRIANGLE = [(0, 1), (1, 2), (2, 0)]                # one triangle, m = 1 on every edge
K4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]   # every edge sits in two triangles
PATH_CHORD = [(0, 1), (1, 2), (2, 3), (1, 3)]      # one triangle (1, 2, 3), one edge outside it


def degrees(edges):
    counts: dict[int, int] = {}
    for i, j in edges:
        counts[i] = counts.get(i, 0) + 1
        counts[j] = counts.get(j, 0) + 1
    return counts


def call(edges):
    return MODULE.af3_values(edges, degrees(edges))


def test_a_cycle_without_triangles_has_af3_zero_on_every_edge():
    # 4 - 2 - 2 + 3*0 = 0
    assert all(abs(v - 0.0) < 1e-12 for v in call(CYCLE4).values())


def test_a_single_triangle_has_af3_three_on_every_edge():
    # 4 - 2 - 2 + 3*1 = 3
    assert all(abs(v - 3.0) < 1e-12 for v in call(TRIANGLE).values())


def test_k4_has_af3_four_on_every_edge():
    # 4 - 3 - 3 + 3*2 = 4
    assert all(abs(v - 4.0) < 1e-12 for v in call(K4).values())


def test_path_with_one_chord_has_the_hand_computed_values():
    values = call(PATH_CHORD)
    # (0,1): 4 - 1 - 3 + 0 = 0 ; (1,2): 4 - 3 - 2 + 3 = 2 ; (1,3): 4 - 3 - 2 + 3 = 2 ; (2,3): 4 - 2 - 2 + 3 = 3
    assert values[(0, 1)] == pytest.approx(0.0)
    assert values[(1, 2)] == pytest.approx(2.0)
    assert values[(1, 3)] == pytest.approx(2.0)
    assert values[(2, 3)] == pytest.approx(3.0)


def test_quantiles_of_a_triangle_free_graph_are_zero():
    assert MODULE.quantiles(call(CYCLE4)) == pytest.approx((0.0, 0.0))


def test_triangles_change_the_weights_when_the_degree_sequence_is_held_fixed():
    """The property the constant-3 defect destroyed: same degrees, different triangles => different weights.

    Both graphs have degree 2 at every node, so any weight that depends only on the degree sum gives the same answer for both; the true AF3 does not.
    """
    cycle_weight = MODULE.weights(CYCLE4, 4, *MODULE.quantiles(call(CYCLE4))).to_dense()
    triangle_weight = MODULE.weights(TRIANGLE, 4, *MODULE.quantiles(call(TRIANGLE))).to_dense()
    cycle_edge = float(cycle_weight[0, 1])
    triangle_edge = float(triangle_weight[0, 1])
    assert cycle_edge == pytest.approx(1.0)          # u = 0 at the only value, so W = 1
    assert triangle_edge == pytest.approx(1.0)       # and here too -- because the clipping span collapses on a uniform graph
    # the discriminating case: one graph with mixed degrees and triangles, its triangle-free counterpart
    chorded = MODULE.weights(PATH_CHORD, 4, *MODULE.quantiles(call(PATH_CHORD))).to_dense()
    triangle_inside = float(chorded[1, 2])
    triangle_free_edge = float(chorded[0, 1])
    assert triangle_inside > 1.0
    assert triangle_free_edge == pytest.approx(1.0)


def test_weights_are_strictly_positive_and_symmetric_on_observed_edges():
    w = MODULE.weights(PATH_CHORD, 4, *MODULE.quantiles(call(PATH_CHORD))).to_dense()
    observed = {(i, j) for i, j in PATH_CHORD} | {(j, i) for i, j in PATH_CHORD}
    for i in range(4):
        for j in range(4):
            value = float(w[i, j])
            if (i, j) in observed:
                assert value > 1.0 - 1e-12
                assert value == pytest.approx(float(w[j, i]))
            elif i != j:
                assert value == pytest.approx(0.0)
