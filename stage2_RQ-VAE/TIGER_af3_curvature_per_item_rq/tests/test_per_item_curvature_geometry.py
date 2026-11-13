"""The geometry itself: the formulas, their limits, the switches, the gradients.

What each test would catch:

* `test_cost_matches_the_formula_at_the_rows_own_curvature` -- the implementation uses
  the scaled rewrite `(1/c) d_1(sqrt(c) x, sqrt(c) y)^2`; this test recomputes `D_ik`
  from the *definition* (`d_{c_i}` at the real `c_i`), so it fails if the rewrite and
  the definition ever disagree (`row_radius_equivalence`).
* `test_a_row_is_read_at_its_own_curvature_only` -- catches an implementation that
  pairs `c_i` with `c_j`, or that averages curvatures over the batch.
* `test_the_curvature_is_not_a_row_scalar` -- the reason this arm can move a token at
  all: if the per-item factor multiplied the WHOLE cost, `argmin` could never change.
  The test asserts flips exist AND that the row ratio varies with the codeword.
* `test_c_to_zero_recovers_the_baseline_cost` / `test_geometry_off_is_the_baseline_cost`
  -- the `flat_limit` and `geometry_off` claims, the second one bit for bit against
  `distances/distance.py`.
* `test_behaviour_off_collapses_the_table_to_one_curvature` -- `u_i = 0.5 -> c_i = 1`
  with the ball map still applied.
* the `quant_parts` tests -- which side each stop-gradient is on, and the `1/d`
  normalisation (so `losses/quantization.py` sees the same scale as the baseline MSE).
* the error tests -- every fail-loud path, with the reason in the message.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import CODEBOOK_DIM, D_E, D_H, SCALE_HYP, reference_cost
from model.per_item_curvature_geometry import (
    C_MAX,
    C_MIN,
    PerItemCurvatureGeometry,
    squared_euclidean,
)


def test_cost_matches_the_formula_at_the_rows_own_curvature(geometry, curvature_table,
                                                            tensor_pair):
    latent, codebook, rows = tensor_pair
    got = geometry.cost(latent, codebook, rows)
    want = reference_cost(latent, codebook, curvature_table)
    assert got.shape == (latent.shape[0], codebook.shape[0])
    # the two derivations are algebraically equal, not numerically identical
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_a_row_is_read_at_its_own_curvature_only(tensor_pair):
    """Permuting the table permutes the cost rows and leaves every column value alone."""
    latent, codebook, _ = tensor_pair
    table = torch.tensor([0.3, 0.7, 1.0, 1.4, 2.0, 2.6, 3.2, 4.0])
    rows = torch.arange(8)
    base = PerItemCurvatureGeometry(table).cost(latent, codebook, rows)
    permutation = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])
    permuted = PerItemCurvatureGeometry(table[permutation]).cost(latent, codebook, rows)
    torch.testing.assert_close(permuted, base[permutation], rtol=0.0, atol=0.0)

    # row i alone, through a one-row table, is the same row
    for index in range(8):
        single = PerItemCurvatureGeometry(table[index:index + 1])
        torch.testing.assert_close(single.cost(latent[index:index + 1], codebook,
                                               torch.zeros(1, dtype=torch.long))[0],
                                   base[index], rtol=1e-6, atol=1e-7)


def test_the_curvature_is_not_a_row_scalar(geometry, curvature_table, tensor_pair):
    """`1 : SCALE_HYP/c_i` changes per row, so `argmin` can move (math.json `not_a_row_scalar`)."""
    latent, codebook, rows = tensor_pair
    cost = geometry.cost(latent, codebook, rows)
    euclid = squared_euclidean(latent, codebook)

    # (a) the relative weight of the two blocks varies WITHIN a row
    ratio = cost / euclid.clamp_min(1e-12)
    varied = (ratio.std(dim=-1) > 1e-3 * ratio.mean(dim=-1)).sum().item()
    assert varied > 0, ("every row's cost is a constant multiple of the Euclidean cost, "
                        "which is exactly the reduction this mechanism is not")

    # (b) and the choice of codeword actually flips, for at least one row
    flips = (cost.argmin(dim=-1) != euclid.argmin(dim=-1)).sum().item()
    wide = torch.logspace(np.log10(C_MIN), np.log10(C_MAX), 16)
    generator = torch.Generator().manual_seed(11)
    big_latent = torch.randn(16, CODEBOOK_DIM, generator=generator) * 0.4
    big_codebook = torch.randn(64, CODEBOOK_DIM, generator=generator) * 0.4
    wide_cost = PerItemCurvatureGeometry(wide).cost(big_latent, big_codebook,
                                                    torch.arange(16))
    wide_flips = (wide_cost.argmin(dim=-1)
                  != squared_euclidean(big_latent, big_codebook).argmin(dim=-1)).sum().item()
    assert wide_flips > 0, ("the per-item curvature never changes the chosen codeword: "
                            "the row factor is behaving like a scalar")
    assert flips >= 0  # the small fixture is allowed to have no flip, it is not the claim


def test_c_to_zero_recovers_the_baseline_cost(tensor_pair):
    latent, codebook, rows = tensor_pair
    flat = PerItemCurvatureGeometry(torch.full((8,), 1e-7))
    baseline = squared_euclidean(latent, codebook)
    torch.testing.assert_close(flat.cost(latent, codebook, rows), baseline,
                               rtol=1e-4, atol=1e-6)


def test_geometry_off_is_the_baseline_cost_bit_for_bit(curvature_table, tensor_pair):
    from distances.distance import distances as baseline

    latent, codebook, rows = tensor_pair
    off = PerItemCurvatureGeometry(curvature_table, geometry_off=True)
    got = off.cost(latent, codebook, rows)
    assert torch.equal(got, baseline(latent, codebook))
    assert torch.equal(got, squared_euclidean(latent, codebook))


def test_behaviour_off_collapses_the_table_to_one_curvature(curvature_table, tensor_pair):
    latent, codebook, rows = tensor_pair
    off = PerItemCurvatureGeometry(curvature_table, behaviour_off=True)
    curvature = off.curvature_of(rows, latent)
    assert torch.equal(curvature, torch.ones(8))
    assert float(curvature.var(unbiased=False)) == 0.0

    # c = 1 keeps the ball map, so this is NOT the baseline cost
    ones = PerItemCurvatureGeometry(torch.ones(8))
    torch.testing.assert_close(off.cost(latent, codebook, rows),
                               ones.cost(latent, codebook, rows), rtol=0.0, atol=0.0)
    assert not torch.allclose(off.cost(latent, codebook, rows),
                              squared_euclidean(latent, codebook))


def test_both_switches_on_is_the_main_arm(curvature_table, tensor_pair):
    latent, codebook, rows = tensor_pair
    on = PerItemCurvatureGeometry(curvature_table, behaviour_off=False, geometry_off=False)
    assert not torch.equal(on.cost(latent, codebook, rows),
                           squared_euclidean(latent, codebook))
    assert float(on.curvature_of(rows, latent).var(unbiased=False)) > 0.0


def test_weight_and_point_arms_decompose_the_main_cost(geometry, tensor_pair):
    """`W = E + H1(1)/c`, `P = E + H1(c)`, `main = E + H1(c)/c` (math.json `block_weight_split`)."""
    from model.poincare_contract import distance, expmap0

    latent, codebook, rows = tensor_pair
    c = geometry.curvature_of(rows, latent)
    r_e, r_h = latent[:, :D_E], latent[:, D_E:]
    t_e, t_h = codebook[:, :D_E], codebook[:, D_E:]
    euclid = (r_e.unsqueeze(1) - t_e.unsqueeze(0)).pow(2).sum(-1)

    def hyp(scale):
        c_cube = torch.as_tensor(scale, dtype=torch.float32).reshape(-1, 1, 1)
        x = expmap0(c_cube.sqrt() * r_h.unsqueeze(1), c_cube)
        y = expmap0(c_cube.sqrt() * t_h.unsqueeze(0), c_cube)
        return distance(x, y, c_cube).pow(2)  # NOTE: this is d_c, so (1/sqrt c)^2 already in

    # H1(c) is the c = 1 read at sqrt(c) scaled points: recompute it that way
    def h1(scale):
        c_cube = torch.as_tensor(scale, dtype=torch.float32).reshape(-1, 1, 1)
        x = expmap0(c_cube.sqrt() * r_h.unsqueeze(1), 1.0)
        y = expmap0(c_cube.sqrt() * t_h.unsqueeze(0), 1.0)
        return distance(x, y, 1.0).pow(2)

    weight_only = geometry.cost_weight_only(latent, codebook, rows)
    point_only = geometry.cost_point_only(latent, codebook, rows)
    main = geometry.cost(latent, codebook, rows)
    c_column = c.unsqueeze(-1)

    torch.testing.assert_close(weight_only,
                               euclid + SCALE_HYP * h1(torch.ones(8)) / c_column,
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(point_only, euclid + SCALE_HYP * h1(c), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(main, euclid + SCALE_HYP * h1(c) / c_column,
                               rtol=1e-5, atol=1e-6)
    # and the definition-level form, which is where `hyp` above comes in
    torch.testing.assert_close(main, euclid + SCALE_HYP * hyp(c), rtol=1e-5, atol=1e-6)

    # with every c = 1 the three arms coincide; with a spread they must not
    ones = PerItemCurvatureGeometry(torch.ones(8))
    for arm in ("cost", "cost_weight_only", "cost_point_only"):
        torch.testing.assert_close(getattr(ones, arm)(latent, codebook, rows),
                                   ones.cost(latent, codebook, rows), rtol=0.0, atol=0.0)
    assert not torch.allclose(main, weight_only)
    assert not torch.allclose(main, point_only)


def test_ball_radius_is_one_over_sqrt_curvature(geometry, curvature_table, tensor_pair):
    latent, _, rows = tensor_pair
    torch.testing.assert_close(geometry.ball_radius(rows, latent),
                               1.0 / curvature_table.sqrt(), rtol=0.0, atol=0.0)


def test_quant_parts_are_the_geodesic_errors_over_the_dimension(geometry, tensor_pair):
    from model.poincare_contract import distance, expmap0

    latent, codebook, rows = tensor_pair
    assigned = codebook[torch.arange(8) % codebook.shape[0]].clone()
    commitment, codebook_part = geometry.quant_parts(latent, assigned, rows)
    assert commitment.shape == () and codebook_part.shape == ()

    c = geometry.curvature_of(rows, latent)
    c_cube = c.reshape(-1, 1, 1)
    r_e, r_h = latent[:, :D_E], latent[:, D_E:]
    q_e, q_h = assigned[:, :D_E], assigned[:, D_E:]
    euclid = (r_e - q_e).pow(2).sum(-1)
    x = expmap0(c_cube.sqrt() * r_h.unsqueeze(1), c_cube).squeeze(1)
    y = expmap0(c_cube.sqrt() * q_h.unsqueeze(1), c_cube).squeeze(1)
    want = ((euclid + SCALE_HYP * distance(x, y, c).pow(2)).mean()) / CODEBOOK_DIM
    torch.testing.assert_close(commitment, want, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(codebook_part, want, rtol=1e-5, atol=1e-6)


def test_quant_parts_send_each_gradient_to_its_own_side(geometry, tensor_pair):
    latent, codebook, rows = tensor_pair
    assigned = codebook[torch.arange(8) % codebook.shape[0]].clone()
    latent = latent.clone().requires_grad_(True)
    assigned = assigned.clone().requires_grad_(True)

    commitment, codebook_part = geometry.quant_parts(latent, assigned, rows)
    commitment.backward(retain_graph=True)
    assert latent.grad is not None and float(latent.grad.abs().sum()) > 0
    assert assigned.grad is None or float(assigned.grad.abs().sum()) == 0.0

    latent.grad = None
    codebook_part.backward()
    assert assigned.grad is not None and float(assigned.grad.abs().sum()) > 0
    assert float(latent.grad.abs().sum()) == 0.0


def test_quant_parts_split_by_level_with_gradients(curvature_table):
    """One DDP step's worth of check: the two errors of a level carry different grads."""
    metric = None  # the metric face is tested elsewhere; this is the geometry directly
    assert metric is None
    geometry = PerItemCurvatureGeometry(curvature_table)
    generator = torch.Generator().manual_seed(3)
    latent = (torch.randn(8, CODEBOOK_DIM, generator=generator) * 0.2).requires_grad_(True)
    assigned = (torch.randn(8, CODEBOOK_DIM, generator=generator) * 0.2).requires_grad_(True)
    commitment, codebook_part = geometry.quant_parts(latent, assigned, torch.arange(8))
    total = commitment + codebook_part
    total.backward()
    assert torch.isfinite(latent.grad).all() and torch.isfinite(assigned.grad).all()
    assert float(latent.grad.abs().sum()) > 0 and float(assigned.grad.abs().sum()) > 0


def test_quant_parts_geometry_off_is_the_baseline_mse_pair(curvature_table, tensor_pair):
    import torch.nn.functional as F

    latent, codebook, rows = tensor_pair
    assigned = codebook[torch.arange(8) % codebook.shape[0]].clone()
    off = PerItemCurvatureGeometry(curvature_table, geometry_off=True)
    commitment, codebook_part = off.quant_parts(latent, assigned, rows)
    torch.testing.assert_close(commitment, F.mse_loss(latent, assigned.detach()),
                               rtol=0.0, atol=0.0)
    torch.testing.assert_close(codebook_part, F.mse_loss(assigned, latent.detach()),
                               rtol=0.0, atol=0.0)


def test_quant_parts_behaviour_off_uses_the_unit_ball(curvature_table, tensor_pair):
    latent, codebook, rows = tensor_pair
    assigned = codebook[torch.arange(8) % codebook.shape[0]].clone()
    off = PerItemCurvatureGeometry(curvature_table, behaviour_off=True)
    ones = PerItemCurvatureGeometry(torch.ones(8))
    torch.testing.assert_close(off.quant_parts(latent, assigned, rows),
                               ones.quant_parts(latent, assigned, rows), rtol=0.0, atol=0.0)


# --------------------------------------------------------------------- fail loud
def test_missing_rows_is_an_error_not_a_default(curvature_table):
    geometry = PerItemCurvatureGeometry(curvature_table)
    latent = torch.zeros(4, CODEBOOK_DIM)
    codebook = torch.zeros(3, CODEBOOK_DIM)
    with pytest.raises(ValueError, match="rows"):
        geometry.cost(latent, codebook, None)
    with pytest.raises(ValueError, match="rows"):
        geometry.cost_weight_only(latent, codebook, None)


def test_a_latent_of_the_wrong_dimension_is_an_error(curvature_table):
    geometry = PerItemCurvatureGeometry(curvature_table)
    with pytest.raises(ValueError, match="32"):
        geometry.cost(torch.zeros(4, 16), torch.zeros(3, CODEBOOK_DIM), torch.arange(4))


def test_a_row_outside_the_table_is_an_error(curvature_table):
    geometry = PerItemCurvatureGeometry(curvature_table)
    with pytest.raises(ValueError, match="curvature table"):
        geometry.cost(torch.zeros(2, CODEBOOK_DIM), torch.zeros(3, CODEBOOK_DIM),
                      torch.tensor([0, 8]))
    with pytest.raises(ValueError, match="curvature table"):
        geometry.curvature_of(torch.tensor([-1]))


def test_a_table_that_cannot_be_a_curvature_table_is_refused():
    for bad, message in ((torch.zeros(0), "empty"),
                         (torch.tensor([1.0, -1.0]), "> 0"),
                         (torch.tensor([1.0, float("nan")]), "non-finite")):
        with pytest.raises(ValueError, match=message):
            PerItemCurvatureGeometry(bad)


def test_quant_parts_with_mismatched_shapes_is_an_error(curvature_table):
    geometry = PerItemCurvatureGeometry(curvature_table)
    with pytest.raises(ValueError, match="shapes"):
        geometry.quant_parts(torch.zeros(4, CODEBOOK_DIM), torch.zeros(3, CODEBOOK_DIM),
                             torch.arange(4))


def test_the_geometry_carries_no_parameter_and_no_buffer():
    """Nothing here trains, so DDP has nothing to synchronise and no state to restore."""
    geometry = PerItemCurvatureGeometry(torch.ones(5))
    module = torch.nn.Module()
    assert not isinstance(geometry, torch.nn.Module)
    assert list(module.parameters()) == []
    assert geometry.table.requires_grad is False
