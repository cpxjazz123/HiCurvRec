"""The parallel-transport verifier's tests (issue #112).

The fixtures that decide whether a transport is the real thing: the isometry, the round trip,
the identity at the origin, the curvature limit (finding F1 -- at a fixed normalised radius the
transport's deviation from the identity does not depend on c, because the curvature is a
rescaling), and the misuse that a candidate's probe actually made.

Needs torch, so run it under the project interpreter:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 -m pytest \
        tools/harness2_genrec/tests/test_transport_verifier.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools" / "hyperbolic_verifier") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "hyperbolic_verifier"))

torch = pytest.importorskip("torch")

import transport as T  # noqa: E402

DIM = 6
C = 1.0


def _points(seed: int, count: int = 24):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, DIM, dtype=torch.float64, generator=generator) * 0.2
    y = torch.randn(count, DIM, dtype=torch.float64, generator=generator) * 0.2
    v = torch.randn(count, DIM, dtype=torch.float64, generator=generator) * 0.3
    return x, y, v


def test_the_transport_is_an_isometry():
    """The Riemannian norm is preserved: lambda(y)||PT(v)|| = lambda(x)||v||."""
    x, y, v = _points(11)
    moved = T.parallel_transport(v, x, y, C)
    before = T.riemannian_norm(v, x, C)
    after = T.riemannian_norm(moved, y, C)
    assert float((before - after).abs().max()) < 1e-9


def test_the_round_trip_returns_the_vector():
    x, y, v = _points(12)
    moved = T.parallel_transport(v, x, y, C)
    back = T.parallel_transport(moved, y, x, C)
    assert float((back - v).abs().max()) < 1e-9


def test_the_transport_from_the_origin_is_the_frame_change():
    """A residual computed at the origin is expressed at another point by THIS map."""
    y, _, v = _points(13)
    moved = T.transport_from_origin(v, y, C)
    assert float((moved - T.parallel_transport(v, torch.zeros_like(y), y, C)).abs().max()) == 0.0
    assert float(T.riemannian_norm(moved, y, C).min()) > 0.0


def test_transporting_a_vector_from_the_wrong_point_gives_a_different_vector():
    """The misuse a candidate's probe made: an ORIGIN-frame residual transported "from x".

    The isometry does NOT detect it -- the transport is a scaled isometry, so it preserves the
    Riemannian norm of ANY input -- and a probe that checked only the norm therefore proved
    nothing. What distinguishes the misuse is the VECTOR: transporting from the wrong point
    rotates the residual by the wrong gyration.
    """
    x, y, _ = _points(14)
    origin_residual = torch.randn(24, DIM, dtype=torch.float64) * 0.1
    wrong = T.parallel_transport(origin_residual, x, y, C)
    right = T.transport_from_origin(origin_residual, y, C)
    assert float((wrong - right).abs().max()) > 1e-3, "the misuse must change the vector"
    # and the isometry holds for the misuse too, which is why the norm check cannot be the test
    before = T.riemannian_norm(origin_residual, x, C)
    after = T.riemannian_norm(wrong, y, C)
    assert float((before - after).abs().max()) < 1e-9


def test_the_curvature_limit_is_a_rescaling():
    """F1: at a fixed NORMALISED radius the transport's deviation from the identity is the same
    for every curvature -- the curvature is a scale, not a shape."""
    generator = torch.Generator().manual_seed(15)
    direction = torch.randn(24, DIM, dtype=torch.float64, generator=generator)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    vector = torch.randn(24, DIM, dtype=torch.float64, generator=generator) * 0.1
    deviations = {}
    for c in (0.25, 1.0, 4.0):
        root = c ** 0.5
        x = direction / root * 0.3          # a fixed normalised radius sqrt(c)||x|| = 0.3
        y = -x
        moved = T.parallel_transport(vector, x, y, c)
        deviations[c] = float((moved - vector).abs().max())
    spread = max(deviations.values()) - min(deviations.values())
    assert spread < 1e-8, deviations
