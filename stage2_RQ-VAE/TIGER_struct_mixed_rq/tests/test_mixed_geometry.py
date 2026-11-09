"""Red-first checks for the mixed E16 x H16 geometry.

Run with: genrec_env_v2/bin/python3.9 tests/test_mixed_geometry.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G


def test_split_sizes():
    z = torch.zeros(3, 32)
    euclid, hyperbolic = G.split(z)
    assert euclid.shape == (3, 16) and hyperbolic.shape == (3, 16)


def test_ball_projection_is_inside_the_ball():
    h = torch.randn(5, 16) * 10
    ball = G.to_ball(h)
    assert (ball.norm(dim=-1) < 1).all()


def test_poincare_sq_matches_closed_form():
    zero = torch.zeros(1, 16, dtype=torch.float64)
    point = torch.zeros(1, 16, dtype=torch.float64)
    point[0, 0] = 0.25
    r = float(G.to_ball(point).norm())
    expected = (2.0 * torch.atanh(torch.tensor(r, dtype=torch.float64)).item()) ** 2
    got = G.poincare_sq(G.to_ball(zero), G.to_ball(point)).item()
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-18)


def test_zero_distance_is_exactly_zero():
    a = torch.randn(5, 16, dtype=torch.float64) * 0.7
    assert torch.equal(G.poincare_sq(a, a.clone()), torch.zeros(5, dtype=torch.float64))


def test_small_distance_is_not_floored():
    # origin to the tangent point 1e-4: d^2 must be (2 atanh(tanh(1e-4)*guard))^2
    a = torch.zeros(1, 16, dtype=torch.float64)
    b = torch.zeros(1, 16, dtype=torch.float64)
    b[0, 0] = 1e-4
    got = G.poincare_sq(G.to_ball(a), G.to_ball(b)).item()
    assert got < 1e-6 and got > 0


def test_near_diagonal_gradient_is_finite():
    a = torch.zeros(1, 16, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, 16, dtype=torch.float64)
    b.data[0, 0] = 1e-6
    value = G.poincare_sq(G.to_ball(a), G.to_ball(b))
    value.backward()
    assert torch.isfinite(a.grad).all()


def test_poincare_sq_is_finite_at_touching_points():
    a = torch.zeros(1, 16)
    assert torch.isfinite(G.poincare_sq(G.to_ball(a), G.to_ball(a))).all()


def test_mixed_sq_is_symmetric_and_separates_halves():
    a = torch.zeros(1, 32)
    b = torch.zeros(1, 32)
    b[0, 0] = 0.5          # euclidean half only
    euclid_only = G.mixed_sq(a, b).item()
    c = torch.zeros(1, 32)
    c[0, 16] = 0.9         # hyperbolic half only
    hyp_only = G.mixed_sq(a, c).item()
    assert math.isclose(G.mixed_sq(a, b).item(), G.mixed_sq(b, a).item(), rel_tol=1e-6)
    assert euclid_only > 0 and hyp_only > 0
    assert euclid_only != hyp_only


def test_mixed_sq_is_zero_for_identical_points():
    a = torch.randn(2, 32)
    assert G.mixed_sq(a, a).abs().max().item() < 1e-6


def test_euclidean_mode_ignores_the_ball():
    a = torch.zeros(1, 32)
    b = torch.zeros(1, 32)
    b[0, 16] = 3.0
    mixed = G.mixed_sq(a, b).item()
    plain = G.mixed_sq(a, b, geometry_on=False).item()
    assert math.isclose(plain, 9.0, rel_tol=1e-6)
    assert mixed != plain


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL ", name, type(exc).__name__, exc)
    raise SystemExit(1 if failures else 0)
