"""The full-space statements must agree with the native geometry: the correct
cross-ratio direction and the Mobius residual, not a reciprocal or a plain
Euclidean subtraction."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mixed_geometry as G
import native_product as N

D = torch.float64


def _expected(tangent_a, tangent_b):
    ball_a, ball_b = G.to_ball(tangent_a), G.to_ball(tangent_b)
    num = float(((ball_a - ball_b) ** 2).sum())
    den = float((1 - (ball_a ** 2).sum()) * (1 - (ball_b ** 2).sum()))
    return float(G.phi(torch.tensor(num / den, dtype=D)))


def _inputs():
    a = torch.zeros(1, 32, dtype=D)
    b = torch.zeros(1, 32, dtype=D)
    a[0, 16] = 0.2
    b[0, 17] = 0.1
    return a, b


def test_the_real_distance_matches_the_independent_cross_ratio():
    a, b = _inputs()
    expected = _expected(a[0, 16:], b[0, 16:])
    got = float(G.mixed_sq(a, b))
    assert abs(got - expected) < 1e-12
    assert abs(got - 0.05052627184756516) < 1e-12
    ball_a = torch.zeros(1, 16, dtype=D)
    ball_b = torch.zeros(1, 16, dtype=D)
    ball_a[0, 0] = 0.2
    ball_b[0, 1] = 0.1
    num = float(((ball_a - ball_b) ** 2).sum())
    den = float((1 - (ball_a ** 2).sum()) * (1 - (ball_b ** 2).sum()))
    produced = float(G.phi(torch.tensor(num / den)))
    assert abs(produced - 0.05171188712120056) < 1e-12
    assert abs(produced - 0.051711885865117525) < 5e-9  # series truncation


def test_a_reciprocal_distance_variant_is_rejected_by_the_same_check():
    a, b = _inputs()
    expected = _expected(a[0, 16:], b[0, 16:])
    saved = G.poincare_sq
    ball_a, ball_b = G.to_ball(a[0, 16:]), G.to_ball(b[0, 16:])
    num = ((ball_a - ball_b) ** 2).sum()
    den = (1 - (ball_a ** 2).sum()) * (1 - (ball_b ** 2).sum())
    G.poincare_sq = lambda x, y: 4.0 * G.phi((den / num).reshape(-1))
    try:
        got = float(G.mixed_sq(a, b))
    finally:
        G.poincare_sq = saved
    assert abs(got - expected) > 1e-2


def test_the_h_residual_is_mobius_not_subtraction():
    latent = torch.zeros(1, 32, dtype=D)
    code = torch.zeros(1, 32, dtype=D)
    latent[0, 16] = 0.4
    latent[0, 17] = -0.1
    code[0, 16] = 0.2
    code[0, 17] = 0.1
    out = N.residuals(latent, [code])[1][0, 16:]
    naive = torch.tensor([0.2, -0.2], dtype=D)
    assert abs(float(out[0]) - 0.1868567811974333) < 1e-12
    assert float((out[:2] - naive).abs().max()) > 1e-2

    ball = G.to_ball(latent[0, 16:])
    recovered = G.log_zero(G.mobius_add(G.code_ball(code[0]), G.to_ball(out)))
    assert float((recovered[:16] - latent[0, 16:]).abs().max()) < 1e-12

    recovered_naive = G.log_zero(G.mobius_add(G.code_ball(code[0]), G.to_ball(
        torch.cat([naive, torch.zeros(14, dtype=D)]))))
    error_naive = float((recovered_naive[:16] - latent[0, 16:]).abs().max())
    assert error_naive > 1e-2


if __name__ == "__main__":
    test_the_real_distance_matches_the_independent_cross_ratio()
    test_a_reciprocal_distance_variant_is_rejected_by_the_same_check()
    test_the_h_residual_is_mobius_not_subtraction()
    print("ok")
