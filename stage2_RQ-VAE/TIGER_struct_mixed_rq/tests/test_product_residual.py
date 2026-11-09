"""The native product residual: Mobius left inverse, nested composition, log readback."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G


def test_mobius_inverse_returns_the_origin():
    a = torch.randn(5, 16, dtype=torch.float64) * 0.3
    origin = G.mobius_add(G.to_ball(a), G.mobius_neg(G.to_ball(a)))
    assert origin.abs().max() < 1e-12


def test_log_is_the_inverse_of_the_ball_map():
    h = torch.randn(6, 16, dtype=torch.float64) * 0.8
    back = G.log_zero(G.to_ball(h))
    assert torch.allclose(back, h, rtol=1e-9, atol=1e-12)


def test_nested_composition_is_right_nested():
    torch.manual_seed(0)
    c1 = torch.randn(3, 32, dtype=torch.float64) * 0.3
    c2 = torch.randn(3, 32, dtype=torch.float64) * 0.3
    c3 = torch.randn(3, 32, dtype=torch.float64) * 0.3
    left = G.nested_compose([c1, c2, c3])
    manual = G.mobius_add(G.code_ball(c1), G.mobius_add(G.code_ball(c2), G.code_ball(c3)))
    assert torch.allclose(left, manual, rtol=1e-10, atol=1e-12)


def test_residual_step_subtracts_euclidean_half_and_uses_the_inverse_in_the_ball():
    torch.manual_seed(1)
    residual = torch.randn(4, 32, dtype=torch.float64) * 0.2
    code = torch.randn(4, 32, dtype=torch.float64) * 0.2
    e, h = G.residual_step(code, residual[..., :G.E_DIM], G.to_ball(residual[..., G.E_DIM:]))
    assert torch.allclose(e, residual[..., :G.E_DIM] - code[..., :G.E_DIM], rtol=1e-12)
    expected_h = G.mobius_add(G.mobius_neg(G.code_ball(code)), G.to_ball(residual[..., G.E_DIM:]))
    assert torch.allclose(h, expected_h, rtol=1e-12)


def test_residual_and_composition_are_not_the_tangent_sum():
    torch.manual_seed(2)
    code = torch.randn(1, 32, dtype=torch.float64) * 0.9
    residual = torch.randn(1, 32, dtype=torch.float64) * 0.9
    _, h_residual = G.residual_step(code, residual[..., :G.E_DIM], G.to_ball(residual[..., G.E_DIM:]))
    tangent_difference = G.to_ball(residual[..., G.E_DIM:] - code[..., G.E_DIM:])
    assert not torch.allclose(h_residual, tangent_difference, atol=1e-3)


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
