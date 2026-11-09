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


def test_the_cross_ratio_direction_matches_the_native_distance():
    a = torch.zeros(1, 16, dtype=D)
    b = torch.zeros(1, 16, dtype=D)
    a[0, 0] = 0.2
    b[0, 1] = 0.1
    num = float(((a - b) ** 2).sum())
    den = float((1 - (a ** 2).sum()) * (1 - (b ** 2).sum()))
    correct = num / den
    reciprocal = den / num
    phi_prod = float(G.phi(torch.tensor(correct, dtype=D)))
    assert abs(phi_prod - 0.051711885865117525) < 1e-12
    assert abs(correct - 0.05260942760942763) < 1e-12
    assert abs(reciprocal - 19.007999999999996) < 1e-9


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
    test_the_cross_ratio_direction_matches_the_native_distance()
    test_the_h_residual_is_mobius_not_subtraction()
    print("ok")
