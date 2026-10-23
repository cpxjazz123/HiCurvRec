"""The Busemann / horosphere verifier's tests, and the R10 audit they establish (issue #112, P1).

The R10 arm calls its operator "a signed Busemann level set" whose level sets are horospheres.
The verifier's own implementation of the Busemann function lets that claim be tested instead of
accepted. The measurements below are the ones recorded in
``results/hyperbolic_research_runs/issue112_busemann_audit.json``.

Needs torch, so run it under the project interpreter:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 -m pytest \
        tools/harness2_genrec/tests/test_busemann_verifier.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools" / "hyperbolic_verifier") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "hyperbolic_verifier"))

torch = pytest.importorskip("torch")

import busemann as B  # noqa: E402

DIM = 6
C = 1.0


def _fixture():
    omega = torch.zeros(DIM, dtype=torch.float64)
    omega[0] = 1.0
    xi = torch.zeros(DIM, dtype=torch.float64)
    xi[1] = 0.3                      # NOT collinear with omega
    return omega, xi


def test_the_busemann_function_is_constant_on_a_horosphere():
    """The verifier's Busemann function is checked against a horosphere built by construction."""
    omega, _ = _fixture()
    for radius in (0.5, 0.3):
        points = B.horosphere_points(omega, radius=radius, count=64)
        levels = B.busemann_boundary(points, omega)
        assert float(levels.max() - levels.min()) < 1e-12
        # the level of the horosphere at fraction radius is log((1-radius)/radius)
        expected = math.log((1.0 - radius) / radius)
        assert float(levels.mean()) == pytest.approx(expected, abs=1e-9)


def test_the_busemann_function_is_signed_inside_the_ball():
    """A genuine Busemann function takes both signs; the level says which side you are on."""
    omega, _ = _fixture()
    grid = torch.rand(2000, DIM, dtype=torch.float64) * 1.6 - 0.8
    inside = grid[(C * (grid * grid).sum(-1)) < 0.9]
    levels = B.busemann_boundary(inside, omega)
    negative = float((levels < 0).double().mean())
    assert 0.5 < negative < 1.0, negative


def test_the_r10_level_is_never_negative():
    """The R10 operator's level is NOT signed: it is clamped into a positive range."""
    omega, xi = _fixture()
    grid = torch.rand(2000, DIM, dtype=torch.float64) * 1.6 - 0.8
    inside = grid[(C * (grid * grid).sum(-1)) < 0.9]
    levels = B.r10_declared_level(inside, xi, C)
    assert float(levels.min()) > 0.0
    assert float((levels < 0).double().mean()) == 0.0


def test_the_r10_level_is_a_near_monotone_function_of_the_distance_to_the_base_point():
    """Which is what makes it an equidistance constraint rather than a horosphere family."""
    _, xi = _fixture()
    grid = torch.rand(2000, DIM, dtype=torch.float64) * 1.6 - 0.8
    inside = grid[(C * (grid * grid).sum(-1)) < 0.9]
    distance = B.poincare_distance(inside, xi.expand_as(inside), C)
    levels = B.r10_declared_level(inside, xi, C)

    def rank(values):
        order = torch.argsort(values)
        ranks = torch.empty(len(values), dtype=torch.float64)
        ranks[order] = torch.arange(len(values), dtype=torch.float64)
        return ranks

    left, right = rank(levels), rank(distance)
    left, right = left - left.mean(), right - right.mean()
    rho = float((left * right).sum() / (left.norm() * right.norm()))
    assert rho < -0.99, rho


def test_the_r10_level_is_not_constant_on_a_horosphere():
    """The decisive one: a horosphere is a level set of the Busemann function, not of this."""
    omega, xi = _fixture()
    points = B.horosphere_points(omega, radius=0.5, count=64)
    levels = B.r10_declared_level(points, xi, C)
    relative_spread = float((levels.max() - levels.min()) / levels.abs().mean())
    assert relative_spread > 0.1, (
        f"the R10 level varies by {relative_spread:.3f} of its mean across a genuine horosphere, "
        "so its level sets are not horospheres")


def test_the_geodesic_sphere_is_not_tangent_and_the_audit_fixture_is_honest():
    """A geodesic sphere has |centre| + radius < 1; a horosphere built by construction is 1."""
    omega, xi = _fixture()
    sphere = B.geodesic_sphere(xi, distance=1.2, count=64, c=C)
    distances = B.poincare_distance(sphere, xi.expand_as(sphere), C)
    assert float(distances.std()) < 1e-12, "the samples must be equidistant"
    centre, radius = B.fit_euclidean_sphere(sphere)
    assert float(centre.norm()) + radius < 1.0


# --------------------------------------------------------------------------------------
# the candidate's own fixtures: the Euclidean twin and the guard
# --------------------------------------------------------------------------------------
def test_the_euclidean_level_set_crosses_the_boundary_and_the_hyperbolic_one_touches_it():
    """H2/H3: the mechanism rests on the curvature CLASS, not on the magnitude of c (F4).

    The Busemann level set is a CLOSED sphere tangent to the boundary from inside; the Euclidean
    level set is a disc whose rim lies ON the boundary. That is the structural difference the
    mechanism uses: a horosphere has an inside and an outside, a half-space does not.
    """
    import horosphere as H

    omega = torch.zeros(DIM, dtype=torch.float64)
    omega[0] = 1.0
    tangent = B.horosphere_points(omega, radius=0.5, count=64)
    on_boundary = float((tangent.norm(dim=-1) > 1.0 - 1e-9).double().mean())

    # the Euclidean level set at the same value: the plane <x, omega> = 0.5 cut by the ball
    rim = []
    for index in range(64):
        angle = 2 * math.pi * index / 64
        direction = torch.zeros(DIM, dtype=torch.float64)
        direction[1] = math.cos(angle)
        direction[2] = math.sin(angle)
        rim.append(0.5 * omega + math.sqrt(0.75) * direction)
    rim = torch.stack(rim)
    euclid_on_boundary = float((rim.norm(dim=-1) > 1.0 - 1e-9).double().mean())

    assert on_boundary < 0.05, on_boundary
    assert euclid_on_boundary > 0.95, euclid_on_boundary
    # and the hyperbolic level is exactly constant on its level set
    assert H.level_spread_on(tangent, lambda p: B.busemann_boundary(p, omega)) < 1e-12


def test_the_hyperbolic_level_is_asymmetric_where_the_euclidean_one_is_not():
    """The Busemann level has an inside and an outside; the Euclidean linear functional splits
    the ball roughly in half, so it cannot say 'which side of the horosphere'."""
    import horosphere as H

    omega = torch.zeros(DIM, dtype=torch.float64)
    omega[0] = 1.0
    grid = torch.rand(2000, DIM, dtype=torch.float64) * 1.6 - 0.8
    inside = grid[(grid * grid).sum(-1) < 0.9]
    hyperbolic = float((B.busemann_boundary(inside, omega) < 0).double().mean())
    euclidean = float((H.euclidean_twin_level(inside, omega) < 0).double().mean())
    assert hyperbolic > 0.9, hyperbolic
    assert 0.35 < euclidean < 0.65, euclidean


def test_the_guard_skips_the_rows_that_leave_the_band():
    """H5: an out-of-band level is skipped and counted, never clamped to a constant."""
    import horosphere as H

    levels = torch.tensor([[0.5, -1.0], [30.0, 0.2], [0.1, -0.3]], dtype=torch.float64)
    keep = H.guard_levels(levels, bound=20.0)
    assert keep.tolist() == [True, False, True]


def test_the_boundary_point_lies_on_the_boundary():
    import horosphere as H

    direction = torch.zeros(DIM, dtype=torch.float64)
    direction[2] = 3.0
    omega = H.boundary_point(direction, c=1.0)
    assert float(omega.norm()) == pytest.approx(1.0, abs=1e-12)
    omega4 = H.boundary_point(direction, c=4.0)
    assert float(omega4.norm()) == pytest.approx(0.5, abs=1e-12)
