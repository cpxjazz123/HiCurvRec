"""The reduced 2-d surface must be the same distance as the 32-d one.

Fixed inputs, printed on failure: the E term and the H term are checked
separately so a mismatch says which one moved.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G

XE, YE, XH, YH = 0.3, -0.1, 0.2, 0.4


def test_reduced_surface_equals_the_full_distance():
    full_l = torch.zeros(1, 32, dtype=torch.float64)
    full_r = torch.zeros(1, 32, dtype=torch.float64)
    full_l[0, 0], full_l[0, 16] = XE, XH
    full_r[0, 0], full_r[0, 16] = YE, YH
    full = G.mixed_sq(full_l, full_r).item()
    left = torch.tensor([[XE, XH]], dtype=torch.float64)
    right = torch.tensor([[YE, YH]], dtype=torch.float64)
    reduced = G.reference_surface(left, right).item()
    e_full = XE ** 2 - 2 * XE * YE + YE ** 2
    print("E", e_full, "H full", full - e_full, "H reduced", reduced - e_full)
    assert abs(full - reduced) < 1e-12, (full, reduced)


if __name__ == "__main__":
    test_reduced_surface_equals_the_full_distance()
    print("ok")
