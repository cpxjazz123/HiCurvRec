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
import structure as S

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
    produced = float(G.phi(torch.tensor(num / den, dtype=D)))
    assert abs(produced - 0.051711885865117525) < 1e-12


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


def test_the_real_distance_has_a_live_h_gradient():
    a, b = _inputs()
    a.requires_grad_(True)
    b.requires_grad_(True)
    d = G.mixed_sq(a, b)
    ga, gb = torch.autograd.grad(d, (a, b))
    assert float(torch.cat([ga, gb])[:, 16:].norm()) > 1e-6

    saved = G.poincare_sq
    G.poincare_sq = lambda x, y: saved(x, y).detach()
    try:
        a2, b2 = _inputs()
        a2.requires_grad_(True)
        b2.requires_grad_(True)
        d2 = G.mixed_sq(a2, b2)
        ga2, gb2 = torch.autograd.grad(d2, (a2, b2))
    finally:
        G.poincare_sq = saved
    assert float(torch.cat([ga2, gb2])[:, 16:].norm()) < 1e-12


def test_the_prefix_reference_is_the_three_layer_stress():
    torch.manual_seed(0)
    qs = [torch.randn(2, 3, 32, dtype=D) for _ in range(3)]
    h = torch.ones(2, 3, dtype=D)
    h[:, 2] = torch.tensor([1.0, 2.0], dtype=D)
    s = torch.ones(2, 1, dtype=D)
    for q in qs:
        q.requires_grad_(True)
    loss = S.reference_prefix_surface(qs, h, s)

    manual = None
    for q in qs:
        d = [G.mixed_sq(q[:, 0], q[:, 1]), G.mixed_sq(q[:, 0], q[:, 2]),
             G.mixed_sq(q[:, 1], q[:, 2])]
        e = [(d[k] / (s[:, 0] ** 2 * h[:, k] ** 2) - 1) ** 2 for k in range(3)]
        layer = (e[0] + e[1] + e[2]) / 3
        manual = layer if manual is None else manual + layer
    manual = manual / 3
    assert torch.allclose(loss, manual, atol=1e-12)
    g = torch.autograd.grad(loss.sum(), qs)
    assert all(float(x[:, :, 16:].norm()) > 1e-6 for x in g)

    zero = [torch.zeros(1, 3, 32, dtype=D) for _ in range(3)]
    collapsed = S.reference_prefix_surface(zero, torch.ones(1, 3, dtype=D),
                                            torch.ones(1, 1, dtype=D))
    assert abs(float(collapsed) - 1.0) < 1e-9


if __name__ == "__main__":
    test_the_real_distance_matches_the_independent_cross_ratio()
    test_a_reciprocal_distance_variant_is_rejected_by_the_same_check()
    test_the_real_distance_has_a_live_h_gradient()
    test_the_prefix_reference_is_the_three_layer_stress()
    test_the_h_residual_is_mobius_not_subtraction()
    print("ok")
