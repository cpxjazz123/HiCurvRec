"""Issue #188: the radial distance, the two-block gradient requirement and the dimensionless term."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import rcs_loss as rl  # noqa: E402


def test_kappa_one_matches_the_hyperbolic_radial_distance():
    x = torch.rand(6, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 0.7
    mine = rl.radial(x, torch.ones(6, dtype=torch.float64))
    theirs = 2.0 * torch.atanh(x.norm(dim=-1).clamp(max=rl.BOUNDARY))
    assert torch.allclose(mine, theirs, rtol=1e-6)


def test_zero_gives_zero():
    x = torch.zeros(3, 4, dtype=torch.float64)
    assert float(rl.radial(x, torch.ones(3, dtype=torch.float64)).abs().max()) < 1e-3   # the eps under the sqrt sets this scale


def test_the_boundary_clamp_keeps_values_and_gradients_finite():
    for norm in (1e-6, 0.5, 1.0 - 1e-9, 1.0):
        x = torch.full((2, 3), norm / 3.0, dtype=torch.float64).requires_grad_(True)
        out = rl.radial(x, torch.full((2,), 2.0, dtype=torch.float64))
        out.sum().backward()
        assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(x.grad).all()), norm


def test_every_curvature_in_the_table_range_is_finite():
    x = torch.rand(4, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.5
    for k in (1e-6, 1e-3, 0.1, 1.0, 8.2671):
        out = rl.radial(x, torch.full((4,), k, dtype=torch.float64))
        assert bool(torch.isfinite(out).all()), k


def test_both_blocks_receive_gradient_the_rad_failure_point():
    encoder = torch.nn.Linear(4, 3, bias=False, dtype=torch.float64)
    codebook = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
    pos = encoder(torch.randn(5, 4, dtype=torch.float64))
    codes = codebook(pos)
    kappa = torch.ones(5, dtype=torch.float64)
    term, _ = rl.loss(pos, codes, kappa)
    blocks = {"encoder": list(encoder.parameters()), "codebooks": list(codebook.parameters())}
    lam, record = rl.calibrate_budget(blocks, (pos.pow(2).sum() + codes.pow(2).sum()), term)
    assert lam > 0.0
    assert record["ratios"]["encoder"] > 0.0 and record["ratios"]["codebooks"] > 0.0


def test_the_loss_is_dimensionless_but_not_scale_invariant():
    # the ratio of a squared radial residual to the squared mean radius has no units, but rho itself is not linear in the norm, so halving
    # every position does NOT halve the residuals: exact scale invariance would be a false claim about atanh
    pos = torch.tensor([[0.2, 0.1], [0.4, 0.3], [0.1, 0.5]], dtype=torch.float64)
    codes = torch.tensor([[0.15, 0.12], [0.35, 0.28], [0.12, 0.45]], dtype=torch.float64)
    kappa = torch.ones(3, dtype=torch.float64)
    base, record = rl.loss(pos, codes, kappa)
    scaled, _ = rl.loss(pos * 0.5, codes * 0.5, kappa)
    assert bool(torch.isfinite(base)) and bool(torch.isfinite(scaled))
    assert float(base) > 0.0 and float(scaled) > 0.0
    assert abs(float(base) - float(scaled)) > 1e-9          # they genuinely differ, and that is the documented behaviour
    assert record["scale"] > 0.0


def test_the_term_is_zero_when_item_and_code_share_a_radius():
    pos = torch.tensor([[0.3, 0.0], [0.0, 0.3]], dtype=torch.float64)
    codes = pos.clone()
    term, _ = rl.loss(pos, codes, torch.ones(2, dtype=torch.float64))
    assert float(term) < 1e-20


def test_the_switch_off_returns_exactly_zero():
    pos = torch.zeros(2, 3, dtype=torch.float64)
    rl.TERM_OFF = True
    term, _ = rl.loss(pos, pos, torch.ones(2, dtype=torch.float64))
    rl.TERM_OFF = False
    assert float(term) == 0.0


def test_no_pairwise_machinery_is_present():
    assert rl.is_pairwise_free(Path(rl.__file__).read_text())
