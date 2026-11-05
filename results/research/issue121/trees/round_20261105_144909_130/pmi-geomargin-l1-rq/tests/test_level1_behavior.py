"""The level-1 behaviour-adjusted assignment of the frozen specification.

Written before the implementation: every test below fails until
``model/level1_behavior.py`` exists and implements the specification as frozen
(round_20261105_144909_130, sha256 ef97f29b1c02de39).

The properties are the specification's own: LIFT(0) = 0, DIST(x, x) = 0, the
c = 0 arm equal to 2||x - y||, the lift inside the ball, score = D - off, a
constant demand giving b_k = log 2, w = 0 during epoch 1, an item-weighted
mean_scale rather than a mean of batch means, and the switch arms doing what
they claim.
"""
from __future__ import annotations

import math

import pytest
import torch

from model.level1_behavior import LevelOneMechanism


def _mechanism(**kwargs) -> LevelOneMechanism:
    torch.manual_seed(0)
    settings = dict(scale=2.0, rho=0.5, rho_off=0.10, eta=0.1, tau=1.0)
    settings.update(kwargs)
    return LevelOneMechanism(**settings)


# ---- the geometry ----------------------------------------------------------

def test_lift_of_the_origin_is_exactly_zero():
    mechanism = _mechanism()
    lifted = mechanism.lift(torch.zeros(3, 5))
    assert torch.equal(lifted, torch.zeros(3, 5))


def test_lift_is_the_identity_at_zero_curvature():
    mechanism = _mechanism()
    points = torch.randn(4, 5)
    assert torch.allclose(mechanism.lift(points, curvature=0.0), points)


def test_lift_stays_inside_the_ball():
    mechanism = _mechanism()
    points = torch.randn(64, 5) * 10.0
    lifted = mechanism.lift(points)
    assert float((lifted.norm(dim=-1) * math.sqrt(mechanism.curvature)).max()) < 1.0


def test_distance_of_a_point_to_itself_is_zero():
    mechanism = _mechanism()
    points = torch.randn(6, 5) * 0.3
    diagonal = mechanism.dist(points, points)
    assert float(diagonal.abs().max()) < 1e-6


def test_the_euclidean_arm_is_twice_the_distance():
    mechanism = _mechanism()
    left = torch.randn(3, 5) * 0.3
    right = torch.randn(4, 5) * 0.3
    curved_zero = mechanism.dist(left[:, None, :], right[None, :, :], curvature=0.0)
    expected = 2.0 * (left[:, None, :] - right[None, :, :]).norm(dim=-1)
    assert torch.allclose(curved_zero, expected, atol=1e-6)


def test_the_distance_is_not_nan_at_the_boundary():
    mechanism = _mechanism()
    points = torch.zeros(2, 5)
    points[0, 0] = 1e6
    distances = mechanism.dist(points[:, None, :], points[None, :, :])
    assert torch.isfinite(distances).all()


# ---- the score and the switches --------------------------------------------

def test_the_score_is_the_distance_minus_the_offset():
    mechanism = _mechanism()
    residual = torch.randn(4, 5)
    codebook = torch.randn(6, 5) * 0.4
    score, distance, offset, _ = mechanism.score(residual, codebook)
    assert torch.allclose(score, distance - offset[None, :], atol=1e-6)


def test_behaviour_off_gives_pure_geodesic_nearest_neighbour():
    mechanism = _mechanism(behaviour_off=True)
    residual = torch.randn(4, 5)
    codebook = torch.randn(6, 5) * 0.4
    score, distance, offset, chosen = mechanism.score(residual, codebook)
    assert torch.equal(offset, torch.zeros_like(offset))
    assert torch.equal(chosen, torch.argmin(distance, dim=-1))


def test_geometry_off_uses_the_euclidean_score():
    mechanism = _mechanism(geometry_off=True)
    residual = torch.randn(4, 5)
    codebook = torch.randn(6, 5) * 0.4
    score, distance, _, _ = mechanism.score(residual, codebook)
    expected = 2.0 * (residual[:, None, :] - codebook[None, :, :]).norm(dim=-1)
    assert torch.allclose(distance, expected, atol=1e-5)


# ---- the demand ------------------------------------------------------------

def test_a_constant_demand_gives_log_two():
    mechanism = _mechanism()
    demand = torch.ones(6)
    b = mechanism.normalise(demand)
    assert torch.allclose(b, torch.full_like(b, math.log(2.0)), atol=1e-6)


def test_the_first_epoch_has_no_offset():
    """mean_scale does not exist before epoch 1 ends, so w = 0 there."""
    mechanism = _mechanism()
    mechanism.observe(min_distance=torch.full((4,), 1.5), assignment=torch.softmax(
        -torch.randn(4, 6), dim=-1), mass=torch.rand(4) + 0.5)
    assert mechanism.epoch_index == 1
    assert float(mechanism.weight()) == 0.0
    assert torch.equal(mechanism.offset(), torch.zeros(6))


def test_mean_scale_is_item_weighted_not_a_mean_of_batch_means():
    mechanism = _mechanism()
    # first batch: four items at 1.0; second batch: one item at 5.0
    mechanism.observe(min_distance=torch.ones(4), assignment=torch.full((4, 6), 1 / 6),
                      mass=torch.ones(4))
    mechanism.observe(min_distance=torch.full((1,), 5.0),
                      assignment=torch.full((1, 6), 1 / 6), mass=torch.ones(1))
    mechanism.end_epoch()
    # item-weighted: (4*1 + 1*5) / 5 = 1.8; a mean of batch means would be 3.0
    assert abs(mechanism.mean_scale() - 1.8) < 1e-4


def test_the_offset_separates_only_from_the_second_epoch():
    mechanism = _mechanism()
    # the two items prefer different codewords, and their masses differ, so the
    # demand separates the codewords
    assignment = torch.tensor([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                                [0.1, 0.9, 0.0, 0.0, 0.0, 0.0]])
    mechanism.observe(min_distance=torch.full((2,), 1.0), assignment=assignment,
                      mass=torch.tensor([3.0, 1.0]))
    mechanism.end_epoch()
    offsets = mechanism.offset()
    assert torch.isfinite(offsets).all()
    assert float(offsets.max() - offsets.min()) > 0


def test_the_demand_is_detached():
    mechanism = _mechanism()
    mass = torch.rand(3, requires_grad=True) + 0.5
    assignment = torch.softmax(torch.randn(3, 6), dim=-1)
    mechanism.observe(min_distance=torch.ones(3), assignment=assignment, mass=mass)
    mechanism.end_epoch()
    assert mechanism._b_raw.requires_grad is False


# ---- the wiring the trainer depends on -------------------------------------

def test_the_mechanism_assigns_with_the_offset_it_computed():
    mechanism = _mechanism()
    residual = torch.randn(5, 5)
    codebook = torch.randn(8, 5) * 0.4
    score, _, offset, chosen = mechanism.score(residual, codebook)
    assert torch.equal(chosen, torch.argmin(score, dim=-1))
    assert offset.shape == (8,)


def test_a_huge_residual_does_not_produce_nan_scores():
    mechanism = _mechanism()
    residual = torch.full((2, 5), 1e4)
    codebook = torch.zeros(4, 5)
    score, _, _, chosen = mechanism.score(residual, codebook)
    assert torch.isfinite(score).all()
    assert chosen.shape == (2,)
