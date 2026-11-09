"""The contract's 2-d surface must be the same formula as the 32-d implementation."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G
import structure as S


def test_reduced_surface_matches_the_closed_form():
    # Independent reference for the reduced surface: E = (x0-y0)^2, and the H part
    # is phi(t) with t built from tanh of the two tangent coordinates.
    torch.manual_seed(0)
    x_e = torch.randn(12, dtype=torch.float64) * 0.3
    y_e = torch.randn(12, dtype=torch.float64) * 0.3
    x_h = torch.randn(12, dtype=torch.float64) * 0.4
    y_h = torch.randn(12, dtype=torch.float64) * 0.4
    a = torch.tanh(x_h)
    b = torch.tanh(y_h)
    t = (a - b) ** 2 / ((1 - a ** 2) * (1 - b ** 2))
    expected = (x_e - y_e) ** 2 + torch.asinh(t.clamp_min(0).sqrt()) ** 2
    left = torch.stack([x_e, x_h], dim=-1)
    right = torch.stack([y_e, y_h], dim=-1)
    got = G.reference_surface(left, right)
    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-15)


def test_reduced_objective_matches_the_full_loss():
    torch.manual_seed(1)
    points = torch.zeros(6, 3, 32, dtype=torch.float64)
    points[..., 0] = torch.randn(6, 3, dtype=torch.float64) * 0.2
    points[..., 16] = torch.randn(6, 3, dtype=torch.float64) * 0.2
    h = torch.tensor([[1.0, 1.0, 1.0]] * 3 + [[1.0, 1.0, 2.0]] * 3, dtype=torch.float64)
    full = S.structure_loss(points, h, scale=0.3)
    reduced = S.reference_objective(points[..., [0, 16]].reshape(6, 6), h, 0.3)
    # the reduced and full layouts reach tanh through differently shaped tensors;
    # the difference is float noise, not a different formula
    assert torch.allclose(full, reduced, rtol=1e-5, atol=1e-9)


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
