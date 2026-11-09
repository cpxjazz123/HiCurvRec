"""The collapse loss value remains one; the rest of the former metadata checks
now live in the formula-reference value/gradient tests."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structure as S


def test_coincident_points_give_one_not_zero():
    points = torch.zeros(1, 6, dtype=torch.float64)
    u = torch.zeros(1, 3, dtype=torch.float64)
    s = torch.ones(1, dtype=torch.float64)
    loss = S.reference_objective_surface(points, u, s)
    assert abs(float(loss) - 1.0) < 1e-9


if __name__ == "__main__":
    test_coincident_points_give_one_not_zero()
    print("ok")
