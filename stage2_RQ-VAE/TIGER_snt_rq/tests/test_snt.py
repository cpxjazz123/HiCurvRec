"""Issue #191: median self-normalisation, its scale invariance and the two off states."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import self_normalised as sn  # noqa: E402
from model import hyperbolic_target as ht  # noqa: E402


def test_median_normalisation_makes_the_mean_ratio_one():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    out, record = sn.median_normalise(values)
    assert torch.allclose(out, values / record["median"])
    assert record["degenerate"] == 0


def test_a_vanishing_median_returns_zeros_and_counts():
    values = torch.zeros(4, dtype=torch.float64)
    out, record = sn.median_normalise(values)
    assert float(out.abs().max()) == 0.0 and record["degenerate"] == 1


def test_scale_invariance_holds_in_the_linear_regime_and_is_bounded_outside_it():
    """The median cancels a common factor exactly only where d_kappa is linear in the positions.

    asinh/atanh are nonlinear, so halving every position does NOT halve every distance; the normalisation removes the common factor it can and the
    residual grows with curvature-times-norm. Both facts are asserted rather than the invariance claim alone.
    """
    a = torch.randn(6, 4, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 0.1
    b = torch.randn(6, 4, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.1

    def stress(x, y, k):
        d = ht.distance(x, y, k)
        normalised, _ = sn.median_normalise(d)
        return ((normalised - 1.0) ** 2).mean()

    # sqrt(k) = 1e-6 against |x| ~ 1e-1 leaves asinh(x)/x - 1 at 5e-15, so the linear regime is exact to float precision
    k_small = torch.full((6,), 1e-12, dtype=torch.float64)
    base = stress(a, b, k_small)
    scaled = stress(a * 0.5, b * 0.5, k_small)
    assert abs(float(base) - float(scaled)) < 1e-9

    k_big = torch.full((6,), 1.4, dtype=torch.float64)
    outside = abs(float(stress(a, b, k_big)) - float(stress(a * 0.5, b * 0.5, k_big)))
    assert outside > 1e-6, "the nonlinear regime must show a measurable difference, not a claimed invariance"


def test_the_normalisation_keeps_the_kappa_dependence():
    a = torch.randn(8, 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 0.1
    b = torch.randn(8, 4, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 0.1
    low, _ = sn.median_normalise(ht.distance(a, b, torch.full((8,), 0.5, dtype=torch.float64)))
    high, _ = sn.median_normalise(ht.distance(a, b, torch.full((8,), 4.0, dtype=torch.float64)))
    assert not torch.allclose(low, high)


def test_the_switches_default_to_the_arm():
    assert sn.GEOMETRY_OFF is False and sn.NORM_OFF is False and sn.BEHAVIOUR_OFF is False


def test_structure_loss_without_kappa_is_struct_mixed_bitwise():
    import structure as S
    import mixed_geometry as G
    torch.manual_seed(9)
    points = torch.randn(4, 3, 6, dtype=torch.float64) * 0.2
    h = torch.ones(4, 3, dtype=torch.float64)
    got = S.structure_loss(points, h, scale=1.0, kappa_pairs=None)
    ratios = torch.stack([G.mixed_sq(points[:, a], points[:, b]) / (1.0 * 1.0 * h[:, i] ** 2)
                          for i, (a, b) in enumerate(S.PAIR_INDEX)], dim=-1)
    assert torch.equal(got, ((ratios - 1.0) ** 2).mean())


def test_the_normalised_state_drops_the_scale_entirely():
    import structure as S
    torch.manual_seed(10)
    points = torch.randn(6, 3, 6, dtype=torch.float64) * 0.2
    h = torch.ones(6, 3, dtype=torch.float64)
    kappa_pairs = [torch.full((6,), k, dtype=torch.float64) for k in (0.7, 1.0, 1.3)]
    one = S.structure_loss(points, h, scale=1.0, kappa_pairs=kappa_pairs)
    ten = S.structure_loss(points, h, scale=10.0, kappa_pairs=kappa_pairs)
    assert torch.equal(one, ten), "the scale argument must be ignored in the normalised state"
