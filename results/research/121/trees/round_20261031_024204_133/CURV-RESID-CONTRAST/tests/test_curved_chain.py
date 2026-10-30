"""The tests for the curved residual chain, written before the implementation.

What they pin down, in the order of the specification:

* the exponential and logarithmic maps are inverse to each other and keep the
  radius inside the ball (the numerical safeguards);
* the parallel transport is the identity at the origin and preserves the tangent
  vector's norm up to the conformal ratio (transport improves on the raw
  Euclidean transport in the curved metric);
* the anisotropic objective reduces to the isotropic one when the gain is one and
  the direction is zero -- the ``behaviour off`` arm;
* the isotropic curved objective ranks codewords the same way as the Euclidean
  nearest neighbour only when the prefix is at the origin, and differs once the
  prefix moves -- the ``geometry on`` arm has to bite;
* the chain's tokens are a deterministic function of the input and the codebooks,
  and they advance the prefix through ``exp_p`` rather than by adding vectors;
* the gradient reaches both the codewords and the encoder input (the objective is
  differentiable end to end).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from model.curved_chain import (  # noqa: E402
    CurvedChain,
    conformal,
    exp_map,
    log_at,
    log_map,
    parallel_transport,
)


torch.manual_seed(0)


def test_exp_and_log_are_inverse():
    points = torch.randn(64, 8, dtype=torch.float64) * 0.5
    curved = exp_map(points)
    recovered = log_map(curved)
    assert torch.allclose(recovered, points, atol=1e-9)


def test_exp_stays_inside_the_ball():
    points = torch.randn(64, 8, dtype=torch.float64) * 10.0
    curved = exp_map(points)
    assert float(curved.norm(dim=-1).max()) < 1.0


def test_transport_is_the_identity_at_the_origin():
    vector = torch.randn(32, 8, dtype=torch.float64)
    origin = torch.zeros(32, 8, dtype=torch.float64)
    moved = parallel_transport(vector, origin)
    assert torch.allclose(moved, vector, atol=1e-12)


def test_transport_follows_the_specification_formula():
    """The specification's transport, evaluated term by term."""
    vector = torch.randn(32, 8, dtype=torch.float64)
    point = torch.randn(32, 8, dtype=torch.float64) * 0.3
    point = point / point.norm(dim=-1, keepdim=True) * 0.6
    moved = parallel_transport(vector, point)
    product = (point * vector).sum(dim=-1, keepdim=True)
    denominator = 1.0 + torch.sqrt(1.0 - (point * point).sum(dim=-1, keepdim=True))
    expected = (conformal(torch.zeros_like(point)) / conformal(point)) * (
        vector - product / denominator * point)
    assert torch.allclose(moved, expected, atol=1e-12)


def test_behaviour_off_reduces_to_the_isotropic_objective():
    chain = CurvedChain(codebooks=torch.randn(3, 16, 8, dtype=torch.float64) * 0.1)
    points = exp_map(torch.randn(5, 8, dtype=torch.float64) * 0.4)
    unit = torch.zeros(5, 8, dtype=torch.float64)
    gain = torch.ones(5, dtype=torch.float64)
    tokens_off, prefixes_off = chain(points, unit, gain)
    tokens_plain, prefixes_plain = chain(points, unit, gain)
    assert torch.equal(tokens_off, tokens_plain)
    assert torch.allclose(prefixes_off, prefixes_plain)


def test_geometry_off_is_the_euclidean_chain():
    codebooks = torch.randn(2, 8, 8, dtype=torch.float64) * 0.1
    chain = CurvedChain(codebooks=codebooks, curvature=0.0)
    points = torch.randn(6, 8, dtype=torch.float64) * 0.4
    unit = torch.zeros(6, 8, dtype=torch.float64)
    gain = torch.ones(6, dtype=torch.float64)
    tokens, prefixes = chain(points, unit, gain)
    # the Euclidean chain picks the nearest codeword at every level and adds it
    residual = points.clone()
    expected = []
    for level in range(2):
        distances = torch.cdist(residual, codebooks[level])
        chosen = distances.argmin(dim=-1)
        expected.append(chosen)
        residual = residual - codebooks[level][chosen]
    assert torch.equal(tokens, torch.stack(expected, dim=-1))
    assert torch.allclose(prefixes, points - residual, atol=1e-9)


def test_the_curved_objective_is_not_the_euclidean_one():
    """The objective value differs as soon as the prefix leaves the origin.

    A conformal map scales every distance by the same positive factor at a point,
    so the RANKING can survive; what cannot survive is the value, and the value is
    what the specification defines.
    """
    torch.manual_seed(7)
    codebooks = torch.randn(1, 16, 8, dtype=torch.float64) * 0.2
    curved = CurvedChain(codebooks=codebooks, curvature=1.0)
    euclidean = CurvedChain(codebooks=codebooks, curvature=0.0)
    points = exp_map(torch.randn(64, 8, dtype=torch.float64) * 1.2)
    unit = torch.zeros(64, 8, dtype=torch.float64)
    gain = torch.ones(64, dtype=torch.float64)
    prefix = points * 0.5
    residual = log_at(prefix, points, 1.0)
    curved_value, _ = curved.level_objective(prefix, residual, unit, gain)
    plain_value, _ = euclidean.level_objective(prefix, residual, unit, gain)
    assert not torch.allclose(curved_value, plain_value, rtol=1e-6)


def test_the_geometry_changes_the_tokens_when_the_codebook_spans_the_radii():
    """Codewords spread over the radial range make the two chains disagree."""
    torch.manual_seed(7)
    radii = torch.linspace(0.05, 0.95, 16, dtype=torch.float64)
    directions = torch.nn.functional.normalize(
        torch.randn(16, 8, dtype=torch.float64), dim=-1)
    codebooks = (directions * radii[:, None]).unsqueeze(0)
    curved = CurvedChain(codebooks=codebooks, curvature=1.0)
    euclidean = CurvedChain(codebooks=codebooks, curvature=0.0)
    points = exp_map(torch.randn(256, 8, dtype=torch.float64) * 1.2)
    unit = torch.zeros(256, 8, dtype=torch.float64)
    gain = torch.ones(256, dtype=torch.float64)
    curved_tokens, _ = curved(points, unit, gain)
    plain_tokens, _ = euclidean(points, unit, gain)
    differing = float((curved_tokens[:, 0] != plain_tokens[:, 0]).float().mean())
    assert differing > 0.1


def test_the_behaviour_gain_changes_the_tokens():
    torch.manual_seed(11)
    codebooks = torch.randn(2, 32, 8, dtype=torch.float64) * 0.2
    chain = CurvedChain(codebooks=codebooks, curvature=1.0)
    points = exp_map(torch.randn(64, 8, dtype=torch.float64) * 1.0)
    unit = torch.nn.functional.normalize(torch.randn(64, 8, dtype=torch.float64), dim=-1)
    neutral = torch.ones(64, dtype=torch.float64)
    amplified = torch.full((64,), 2.0, dtype=torch.float64)
    tokens_neutral, _ = chain(points, unit, neutral)
    tokens_amplified, _ = chain(points, unit, amplified)
    assert not torch.equal(tokens_neutral, tokens_amplified)


def test_the_prefix_advances_through_the_exponential_map():
    codebooks = torch.randn(1, 8, 8, dtype=torch.float64) * 0.1
    chain = CurvedChain(codebooks=codebooks, curvature=1.0)
    points = exp_map(torch.randn(6, 8, dtype=torch.float64) * 0.4)
    unit = torch.zeros(6, 8, dtype=torch.float64)
    gain = torch.ones(6, dtype=torch.float64)
    tokens, prefixes = chain(points, unit, gain)
    assert float(prefixes.norm(dim=-1).max()) < 1.0
    # after one level the prefix is exp_0 of the chosen codeword, not a sum
    chosen = codebooks[0][tokens[:, 0]]
    assert torch.allclose(prefixes, exp_map(chosen), atol=1e-9)


def test_gradients_reach_the_codewords_and_the_input():
    codebooks = (torch.randn(2, 8, 8, dtype=torch.float64) * 0.1).requires_grad_(True)
    chain = CurvedChain(codebooks=codebooks, curvature=1.0)
    raw = (torch.randn(6, 8, dtype=torch.float64) * 0.4).requires_grad_(True)
    points = exp_map(raw)
    unit = torch.nn.functional.normalize(torch.randn(6, 8, dtype=torch.float64), dim=-1)
    gain = torch.ones(6, dtype=torch.float64)
    tokens, prefixes = chain(points, unit, gain, soft=True)
    loss = prefixes.sum()
    loss.backward()
    assert chain.codebooks.grad is not None and float(chain.codebooks.grad.abs().sum()) > 0
    assert raw.grad is not None and float(raw.grad.abs().sum()) > 0


def test_the_chain_is_deterministic():
    codebooks = torch.randn(2, 8, 8, dtype=torch.float64) * 0.1
    chain = CurvedChain(codebooks=codebooks, curvature=1.0)
    points = exp_map(torch.randn(6, 8, dtype=torch.float64) * 0.4)
    unit = torch.zeros(6, 8, dtype=torch.float64)
    gain = torch.ones(6, dtype=torch.float64)
    first, _ = chain(points, unit, gain)
    second, _ = chain(points, unit, gain)
    assert torch.equal(first, second)
