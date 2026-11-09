"""The contract's 2-d surface must be the same formula as the 32-d implementation."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G
import structure as S


def test_reduced_surface_matches_the_full_distance():
    torch.manual_seed(0)
    # Radial inputs: everything on the first Euclidean coordinate and on the
    # first hyperbolic one, which is the configuration the 2-d surface states.
    left = torch.zeros(12, 32, dtype=torch.float64)
    right = torch.zeros(12, 32, dtype=torch.float64)
    left[:, 0] = torch.randn(12, dtype=torch.float64) * 0.3
    right[:, 0] = torch.randn(12, dtype=torch.float64) * 0.3
    left[:, 16] = torch.randn(12, dtype=torch.float64) * 0.4
    right[:, 16] = torch.randn(12, dtype=torch.float64) * 0.4
    # mixed_sq expects the 32-d layout; place the two coordinates where the split
    # reads them (index 0 for E, index 16 for the first H coordinate).
    full_l = torch.zeros(12, 32, dtype=torch.float64)
    full_r = torch.zeros(12, 32, dtype=torch.float64)
    full_l[:, 0], full_l[:, 16] = left[:, 0], left[:, 1]
    full_r[:, 0], full_r[:, 16] = right[:, 0], right[:, 1]
    full = G.mixed_sq(full_l, full_r)
    reduced = G.reference_surface(left[:, [0, 16]], right[:, [0, 16]])
    # the reduced and full layouts reach tanh through differently shaped tensors;
    # the difference is float noise, not a different formula
    assert torch.allclose(full, reduced, rtol=1e-5, atol=1e-9)


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
