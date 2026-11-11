"""The contract's positive-target form must agree with the training objective."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import structure as S


def test_tanh_form_matches_the_objective():
    torch.manual_seed(3)
    points = torch.zeros(5, 6, dtype=torch.float64)
    points[:, 0] = torch.randn(5, dtype=torch.float64) * 0.2
    points[:, 3] = torch.randn(5, dtype=torch.float64) * 0.2
    points[:, 1] = torch.randn(5, dtype=torch.float64) * 0.3
    points[:, 5] = torch.randn(5, dtype=torch.float64) * 0.3
    u = torch.randn(5, 3, dtype=torch.float64) * 0.5
    contract = S.reference_objective_surface(points, u, torch.full((5, 1), 0.3, dtype=torch.float64))
    h = 1.0 + torch.tanh(u)
    direct = S.reference_objective(points, h, 0.3)
    assert torch.allclose(contract.mean(), direct, rtol=1e-9, atol=1e-12)


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
