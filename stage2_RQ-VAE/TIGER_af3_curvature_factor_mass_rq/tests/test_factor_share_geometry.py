"""Mathematics of `factor_share_geometry`: the identities, the domain, the two switches.

Every assertion here is a statement of `ideas/AF3_curvature_factor_mass_idea/math.json`,
so a wrong factor, a wrong side (product vs codeword), a wrong curvature or a lost share
fails loudly instead of merely running.

The tests are written against the frozen geometry contract
(`tools/research/geometry/poincare_contract.py`), not against a re-derivation, so an
operator that quietly drifts from the contract fails too.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

import factor_share_geometry as geometry


def _fixture(batch: int = 3, codes: int = 5, seed: int = 0):
    torch.manual_seed(seed)
    latent = torch.randn(batch, geometry.D_TOTAL, dtype=torch.float64)
    codebook = torch.randn(codes, geometry.D_TOTAL, dtype=torch.float64) * 0.3
    share = torch.tensor([0.12, 0.5, 0.87][:batch], dtype=torch.float64)
    return latent, codebook, share


def _sq(x, y, c):
    return geometry.contract().squared_distance(x, y, c)


# --------------------------------------------------------------------------------------
# constants: the fixed curvature ladder and the two fixed scales
# --------------------------------------------------------------------------------------

def test_constants_match_math_json():
    assert (geometry.D_E, geometry.D_H1, geometry.D_H2) == (16, 8, 8)
    assert geometry.D_E + geometry.D_H1 + geometry.D_H2 == 32  # = baseline codebook_dim
    assert geometry.C_HI == 4.0 and geometry.C_LO == 0.25      # fixed ladder, never per item
    assert geometry.SCALE_HYP == 0.25                          # pinned by the c -> 0 limit
    assert geometry.SHARE_REF == 0.5


def test_geometry_has_no_parameters():
    """No trainable weight, no buffer: the module is functions only."""
    for name, value in vars(geometry).items():
        if name.startswith("_"):
            continue
        assert not isinstance(value, torch.nn.Module), name
        assert not isinstance(value, torch.nn.Parameter), name


# --------------------------------------------------------------------------------------
# share_map
# --------------------------------------------------------------------------------------

def test_share_struct_is_the_identity_and_domain_is_open_interval():
    u = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
    p = geometry.share(u, "struct")
    assert torch.equal(p, u)  # p_i = u_i, nothing else
    assert float(p.min()) > 0.0 and float(p.max()) < 1.0


def test_share_ref_is_the_constant_even_for_other_u():
    u = torch.tensor([0.01, 0.4, 0.99], dtype=torch.float64)
    p = geometry.share(u, "ref")
    assert torch.equal(p, torch.full_like(u, geometry.SHARE_REF))
    # Same signature on both branches: behaviour_off changes the value, not the call.
    assert list(inspect.signature(geometry.share).parameters) == ["u", "mode"]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5, float("nan")])
def test_share_rejects_values_outside_the_open_interval(bad):
    u = torch.tensor([0.5, bad], dtype=torch.float64)
    with pytest.raises(ValueError):
        geometry.share(u, "struct")
    with pytest.raises(ValueError):
        geometry.share(u, "ref")


def test_share_rejects_unknown_mode():
    with pytest.raises(ValueError):
        geometry.share(torch.tensor([0.5], dtype=torch.float64), "both_on")


# --------------------------------------------------------------------------------------
# chart_head: the encoder output IS a point of M, and the product side carries no p
# --------------------------------------------------------------------------------------

def test_chart_head_is_expmap0_of_each_hyperbolic_block_only():
    latent, _, _ = _fixture()
    contract = geometry.contract()
    x_hi, x_lo = geometry.item_chart_head(latent)
    assert torch.equal(x_hi, contract.expmap0(latent[:, 16:24], geometry.C_HI))
    assert torch.equal(x_lo, contract.expmap0(latent[:, 24:], geometry.C_LO))
    # r^{E,0} = h_E: the Euclidean block is carried through unscaled and unmapped.
    rE, h1, h2 = geometry.split_tangent(latent)
    assert torch.equal(rE, latent[:, :16])
    assert torch.equal(h1, latent[:, 16:24]) and torch.equal(h2, latent[:, 24:])


def test_chart_head_takes_no_share_at_all():
    """`energy_split`: scaling only ever touches the codeword side."""
    parameters = list(inspect.signature(geometry.item_chart_head).parameters)
    assert parameters == ["h"]
    parameters = list(inspect.signature(geometry.codeword_chart).parameters)
    assert parameters == ["tau", "p"]


def test_log_at_origin_inverts_the_chart_head():
    """`logmap0(expmap0(h)) == h` while the chart does not clip.

    A `randn(3, 32)` tangent (|h| ~ 2.8) is NOT a valid fixture here: with c_hi = 4 the
    ball radius is 0.5, so `project` saturates the point and `logmap0` returns the
    CLIPPED point's tangent, not the input. The round trip is an identity exactly while
    the tangent stays inside the ball, so the fixture is scaled down and the ball
    membership is asserted as part of the test.
    """
    torch.manual_seed(0)
    latent = torch.randn(3, geometry.D_TOTAL, dtype=torch.float64) * 0.01
    x_hi, x_lo = geometry.item_chart_head(latent)
    contract = geometry.contract()
    assert torch.equal(x_hi, contract.expmap0(latent[:, 16:24], geometry.C_HI))
    assert torch.equal(x_lo, contract.expmap0(latent[:, 24:], geometry.C_LO))
    # inside the ball of each curvature: c |x|^2 < 1 (a saturated point sits at
    # (1 - BALL_EPS)/sqrt(c) and is NOT invertible by logmap0)
    assert bool((geometry.C_HI * x_hi.pow(2).sum(-1) < 1.0).all())
    assert bool((geometry.C_LO * x_lo.pow(2).sum(-1) < 1.0).all())
    back = geometry.log_at_origin(x_hi, x_lo)
    assert back.shape == (latent.shape[0], geometry.D_H)
    assert torch.allclose(back, torch.cat([latent[:, 16:24], latent[:, 24:]], dim=-1),
                          rtol=0, atol=1e-8)


# --------------------------------------------------------------------------------------
# energy_split: the share multiplies the CODEWORD blocks, with sqrt(p) / sqrt(1-p)
# --------------------------------------------------------------------------------------

def _codeword_fixture(batch: int = 3, codes: int = 5, seed: int = 0):
    """A codebook small enough to stay inside both balls (c_hi -> radius 0.5), so the
    expmap0/logmap0 round trip below is exact rather than clipped by `project`."""
    torch.manual_seed(seed)
    codebook = torch.randn(codes, geometry.D_TOTAL, dtype=torch.float64) * 0.05
    p = torch.tensor([0.12, 0.5, 0.87][:batch], dtype=torch.float64)
    return codebook, p


def test_codeword_chart_scales_the_codeword_side_by_sqrt_of_the_share():
    codebook, p = _codeword_fixture()
    contract = geometry.contract()
    v_hi, v_lo = geometry.codeword_chart(codebook, p)
    assert v_hi.shape == (p.shape[0], codebook.shape[0], geometry.D_H1)
    assert v_lo.shape == (p.shape[0], codebook.shape[0], geometry.D_H2)
    for row in range(p.shape[0]):
        back_hi = contract.logmap0(v_hi[row], geometry.C_HI)
        back_lo = contract.logmap0(v_lo[row], geometry.C_LO)
        expected_hi = math.sqrt(float(p[row])) * codebook[:, 16:24]
        expected_lo = math.sqrt(1.0 - float(p[row])) * codebook[:, 24:]
        assert torch.allclose(back_hi, expected_hi, rtol=0, atol=1e-8)
        assert torch.allclose(back_lo, expected_lo, rtol=0, atol=1e-8)


def test_codeword_chart_depends_on_the_share_while_the_product_point_does_not():
    latent, _, _ = _fixture()
    codebook, _ = _codeword_fixture()
    low = torch.full((latent.shape[0],), 0.1, dtype=torch.float64)
    high = torch.full((latent.shape[0],), 0.9, dtype=torch.float64)
    v_hi_low, _ = geometry.codeword_chart(codebook, low)
    v_hi_high, _ = geometry.codeword_chart(codebook, high)
    assert not torch.allclose(v_hi_low, v_hi_high)
    x_hi, x_lo = geometry.item_chart_head(latent)
    contract = geometry.contract()
    # The product point is the UNSCALED expmap0 of h1/h2: a sqrt(p) factor would move
    # it, so equality against the raw chart is the evidence that the share is not there.
    assert torch.equal(x_hi, contract.expmap0(
        latent[:, geometry.D_E:geometry.D_E + geometry.D_H1], geometry.C_HI))
    assert torch.equal(x_lo, contract.expmap0(
        latent[:, geometry.D_E + geometry.D_H1:], geometry.C_LO))
    # Computing the codeword side at either share cannot move the product point.
    geometry.codeword_chart(codebook, low)
    geometry.codeword_chart(codebook, high)
    again_hi, again_lo = geometry.item_chart_head(latent)
    assert torch.equal(x_hi, again_hi)
    assert torch.equal(x_lo, again_lo)
    # the effective codeword radius is monotone in sqrt(p)
    assert (contract.logmap0(v_hi_high, geometry.C_HI).norm(dim=-1)
            > contract.logmap0(v_hi_low, geometry.C_HI).norm(dim=-1)).all()


def test_codeword_chart_needs_the_share():
    codebook, _ = _codeword_fixture()
    with pytest.raises(ValueError):
        geometry.codeword_chart(codebook, None)


# --------------------------------------------------------------------------------------
# product_cost: the formula, term by term
# --------------------------------------------------------------------------------------

def test_product_cost_is_the_sum_of_the_three_terms_with_the_stated_weights():
    latent, codebook, p = _fixture()
    contract = geometry.contract()
    rE, h1, h2 = geometry.split_tangent(latent)
    x_hi, x_lo = geometry.item_chart_head(latent)
    v_hi, v_lo = geometry.codeword_chart(codebook, p)
    tau_E = codebook[:, :geometry.D_E]

    euclidean = (rE.unsqueeze(1) - tau_E.unsqueeze(0)).pow(2).sum(-1)
    hyperbolic = (_sq(x_hi.unsqueeze(1), v_hi, geometry.C_HI)
                  + _sq(x_lo.unsqueeze(1), v_lo, geometry.C_LO))
    expected = euclidean + geometry.SCALE_HYP * hyperbolic

    got = geometry.product_cost(latent, codebook, p)
    assert got.shape == (latent.shape[0], codebook.shape[0])
    assert torch.allclose(got, expected, rtol=0, atol=1e-12)
    # The Euclidean block enters with weight 1, the hyperbolic blocks with SCALE_HYP.
    assert torch.allclose(got - euclidean, geometry.SCALE_HYP * hyperbolic,
                          rtol=0, atol=1e-12)


def test_product_cost_is_row_separable_no_item_pair_signal():
    """D_ik depends on p_i only: moving p_1 must not move row 0 (must_hold#1)."""
    latent, codebook, p = _fixture()
    moved = p.clone()
    moved[1] = 0.9 if float(p[1]) < 0.8 else 0.1  # a different share for item 1 only
    base = geometry.product_cost(latent, codebook, p)
    other = geometry.product_cost(latent, codebook, moved)
    assert torch.equal(base[0], other[0])
    assert not torch.allclose(base[1], other[1])


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_product_cost_rejects_a_degenerate_share(bad):
    latent, codebook, p = _fixture()
    p = p.clone()
    p[0] = bad
    with pytest.raises(ValueError):
        geometry.product_cost(latent, codebook, p)


def test_product_cost_rejects_missing_or_misshaped_p():
    latent, codebook, _ = _fixture()
    with pytest.raises(ValueError):
        geometry.product_cost(latent, codebook, None)
    with pytest.raises(ValueError):
        geometry.product_cost(latent, codebook, torch.tensor([0.5, 0.5], dtype=torch.float64))
    with pytest.raises(ValueError):
        geometry.product_cost(latent[:, :16], codebook, torch.tensor([0.5], dtype=torch.float64))
    with pytest.raises(ValueError):
        geometry.product_cost(latent, codebook[:, :16],
                              torch.full((latent.shape[0],), 0.5, dtype=torch.float64))


def test_product_cost_rejects_unknown_mode():
    latent, codebook, p = _fixture()
    with pytest.raises(ValueError):
        geometry.product_cost(latent, codebook, p, mode="curvature_off")


def test_cost_rejects_blocks_that_do_not_agree():
    latent, codebook, p = _fixture()
    rE, _, _ = geometry.split_tangent(latent)
    x_hi, x_lo = geometry.item_chart_head(latent)
    v_hi, v_lo = geometry.codeword_chart(codebook, p)
    tau_E = codebook[:, :geometry.D_E]
    with pytest.raises(ValueError):
        geometry.cost(rE[:-1], x_hi, x_lo, v_hi, v_lo, tau_E)
    with pytest.raises(ValueError):
        geometry.cost(rE, x_hi, x_lo, v_hi[:, :3], v_lo, tau_E)


# --------------------------------------------------------------------------------------
# euclidean_degeneracy / geometry_off
# --------------------------------------------------------------------------------------

def test_geometry_off_is_the_baseline_expansion_elementwise():
    latent, codebook, p = _fixture()
    baseline = (latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t())
    assert torch.equal(geometry.euclidean_expansion(latent, codebook), baseline)
    assert torch.equal(geometry.product_cost(latent, codebook, p, mode="geometry_off"),
                       baseline)
    # p is ignored, but not optional: a missing share is still a wiring error.
    with pytest.raises(ValueError):
        geometry.product_cost(latent, codebook, None, mode="geometry_off")


def test_c_to_zero_limit_keeps_the_scaling_and_drops_only_the_curvature(monkeypatch):
    """`euclidean_degeneracy`: c -> 0 gives 32-dim WEIGHTED squared Euclidean."""
    latent, codebook, p = _fixture()
    monkeypatch.setattr(geometry, "C_HI", 0.0)
    monkeypatch.setattr(geometry, "C_LO", 0.0)
    rE, h1, h2 = geometry.split_tangent(latent)
    tau_E = codebook[:, :geometry.D_E]
    s_hi = torch.sqrt(p).view(-1, 1, 1)
    s_lo = torch.sqrt(1.0 - p).view(-1, 1, 1)
    expected = (
        (rE.unsqueeze(1) - tau_E.unsqueeze(0)).pow(2).sum(-1)
        + (h1.unsqueeze(1) - s_hi * codebook.unsqueeze(0)[:, :, 16:24]).pow(2).sum(-1)
        + (h2.unsqueeze(1) - s_lo * codebook.unsqueeze(0)[:, :, 24:]).pow(2).sum(-1)
    )
    got = geometry.product_cost(latent, codebook, p)
    assert torch.allclose(got, expected, rtol=0, atol=1e-9)
    # ... and it is NOT the baseline cost: the codeword side is still scaled.
    baseline = (latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t())
    assert not torch.allclose(got, baseline)


# --------------------------------------------------------------------------------------
# the two switches, one side each
# --------------------------------------------------------------------------------------

def test_behaviour_off_is_independent_of_u_and_equals_the_share_ref_arm():
    latent, codebook, p = _fixture()
    half = torch.full_like(p, geometry.SHARE_REF)
    other = torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64)
    off_a = geometry.product_cost(latent, codebook, p, mode="behaviour_off")
    off_b = geometry.product_cost(latent, codebook, other, mode="behaviour_off")
    assert torch.equal(off_a, off_b)
    assert torch.equal(off_a, geometry.product_cost(latent, codebook, half))
    # the main arm differs as soon as some share is not SHARE_REF
    assert not torch.allclose(geometry.product_cost(latent, codebook, p), off_a)


def test_geometry_off_and_behaviour_off_are_different_switches():
    latent, codebook, p = _fixture()
    off_euclidean = geometry.product_cost(latent, codebook, p, mode="geometry_off")
    off_share = geometry.product_cost(latent, codebook, p, mode="behaviour_off")
    assert not torch.allclose(off_euclidean, off_share)  # curvature is still there


# --------------------------------------------------------------------------------------
# geometric_quantization: (C, Q), denominators, stop-grad sides
# --------------------------------------------------------------------------------------

def _parts_inputs(seed: int = 1):
    torch.manual_seed(seed)
    batch, codes = 4, 3
    latent = torch.randn(batch, geometry.D_TOTAL, dtype=torch.float64) * 0.05
    codebook = torch.randn(codes, geometry.D_TOTAL, dtype=torch.float64) * 0.05
    p = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)
    rE, _, _ = geometry.split_tangent(latent)
    x_hi, x_lo = geometry.item_chart_head(latent)
    v_hi, v_lo = geometry.codeword_chart(codebook, p)
    tau_E = codebook[:, :geometry.D_E]
    index = torch.tensor([0, 2, 1, 0])
    return latent, codebook, p, rE, x_hi, x_lo, v_hi, v_lo, tau_E, index


def test_quant_parts_matches_the_gathered_cost_over_32_dimensions():
    _, codebook, p, rE, x_hi, x_lo, v_hi, v_lo, tau_E, index = _parts_inputs()
    commitment, codebook_term = geometry.quant_parts(rE, x_hi, x_lo, v_hi, v_lo, tau_E, index)
    assert commitment.dim() == 0 and codebook_term.dim() == 0

    rows = torch.arange(rE.shape[0])
    chosen_hi = v_hi[rows, index]
    chosen_lo = v_lo[rows, index]
    chosen_E = tau_E[index]
    per_row = ((rE - chosen_E).pow(2).sum(-1)
               + geometry.SCALE_HYP * (_sq(x_hi, chosen_hi, geometry.C_HI)
                                       + _sq(x_lo, chosen_lo, geometry.C_LO)))
    expected = per_row.mean() / float(geometry.D_TOTAL)  # 32, like F.mse_loss
    assert torch.allclose(commitment, expected, rtol=0, atol=1e-12)
    assert torch.allclose(codebook_term, expected, rtol=0, atol=1e-12)
    # denominator is D_TOTAL, not the number of blocks and not the batch
    wrong = per_row.mean() / 3.0
    assert not torch.allclose(commitment, wrong)


def test_commitment_reaches_the_product_side_only():
    _, codebook, p, rE, x_hi, x_lo, v_hi, v_lo, tau_E, index = _parts_inputs()
    rE = rE.clone().requires_grad_(True)
    tau = codebook.clone().requires_grad_(True)
    v_hi, v_lo = geometry.codeword_chart(tau, p)
    commitment, _ = geometry.quant_parts(rE, x_hi, x_lo, v_hi, v_lo, tau[:, :geometry.D_E],
                                        index)
    commitment.backward()
    assert rE.grad is not None and float(rE.grad.abs().sum()) > 0.0
    assert tau.grad is None  # codeword side is stopped, like losses/commitment.py


def test_codebook_term_reaches_the_codeword_side_only():
    _, codebook, p, rE, x_hi, x_lo, _, _, _, index = _parts_inputs()
    rE = rE.clone().requires_grad_(True)
    tau = codebook.clone().requires_grad_(True)
    v_hi, v_lo = geometry.codeword_chart(tau, p)
    _, codebook_term = geometry.quant_parts(rE, x_hi, x_lo, v_hi, v_lo, tau[:, :geometry.D_E],
                                           index)
    codebook_term.backward()
    assert tau.grad is not None and float(tau.grad.abs().sum()) > 0.0
    assert rE.grad is None  # product side is stopped, like losses/codebook.py


def test_quant_parts_domain_and_dtype_of_index():
    _, _, _, rE, x_hi, x_lo, v_hi, v_lo, tau_E, index = _parts_inputs()
    with pytest.raises(ValueError):
        geometry.quant_parts(rE, x_hi, x_lo, v_hi, v_lo, tau_E, None)
    with pytest.raises(ValueError):
        geometry.quant_parts(rE, x_hi, x_lo, v_hi, v_lo, tau_E, index[:-1])
    with pytest.raises(ValueError):
        geometry.quant_parts(rE, x_hi, x_lo, v_hi, v_lo, tau_E,
                             torch.tensor([0, 1, 2, 9]))


def test_quant_parts_gradient_reaches_the_encoder_like_the_baseline():
    """Both terms are differentiable in the product side (straight-through upstream)."""
    _, codebook, p, rE, x_hi, x_lo, _, _, _, index = _parts_inputs()
    latent = torch.cat([rE, geometry.log_at_origin(x_hi, x_lo)], dim=-1).clone()
    latent.requires_grad_(True)
    rE2, _, _ = geometry.split_tangent(latent)
    x_hi2, x_lo2 = geometry.item_chart_head(latent)
    v_hi2, v_lo2 = geometry.codeword_chart(codebook, p)
    commitment, codebook_term = geometry.quant_parts(
        rE2, x_hi2, x_lo2, v_hi2, v_lo2, codebook[:, :geometry.D_E], index)
    (commitment + codebook_term).backward()
    assert latent.grad is not None
    assert float(latent.grad.abs().sum()) > 0.0
