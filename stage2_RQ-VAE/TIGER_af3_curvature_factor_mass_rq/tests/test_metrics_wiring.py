"""The read-only `metrics` of `math.json`: each one measures, none of them assumes.

These readings are what the arm uses to prove the mechanism is wired
(`switches.behaviour_off`: "SID 与主臂相同即未接线") and that the fixed curvature is
not saturated (`constants.c_hi.why`). They are diagnostics only: nothing here enters
the loss, the gradient or the assignment.

Groups:

* `codeword_report`: the two scaling factors, the codeword-side norms per level, the
  `project` boundary saturation count, and the c -> 0 expansion exponents;
* `assignment_report`: per-level argmin flips between the main arm and
  `behaviour_off`, plus the SID difference rate (zero only when the shares coincide);
* `geometry_readout`: the c -> 0 limit against the baseline expansion, the batch
  permutation equivariance, the dimension control and the effective beta scale;
* the trainer actually calls all three.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import factor_share_geometry as geometry
import train_rqvae
from distances import factor_share as adapter
from model.model import RQVAE


def tokenizer_config(**overrides):
    values = dict(
        hidden_sizes=[8, 4],
        codebook_num=2,
        codebook_size=(6, 6),
        codebook_dim=32,
        dropout=0.0,
        beta=0.25,
        vq_type="vq",
        ema_decay=0.99,
        fix_code_embs=False,
        # A positive Sinkhorn setting so the LAST level really carries the
        # Sinkhorn feed (`RQLayer` gives it to `level == codebook_num - 1` only,
        # `model/layers.py:358`); earlier levels stay argmin.
        sk_epsilon=1.0,
        sk_iters=20,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def share_table(values) -> dict:
    u = np.asarray(values, dtype=np.float64)
    return {"u": u, "item_ids": np.arange(u.size, dtype=np.int64)}


def hidden_fixture(rows: int = 5, seed: int = 0, scale: float = 0.3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, geometry.D_TOTAL, generator=generator) * scale


def codebook_fixture(codes: int = 5, seed: int = 1, scale: float = 0.05) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(codes, geometry.D_TOTAL, generator=generator) * scale


# --------------------------------------------------------------------------------------
# codeword_scale_energy + clip_nan_count + the `4_method` scaling log
# --------------------------------------------------------------------------------------

def test_codeword_report_measures_the_two_scalings_and_the_expansion_powers():
    table = share_table([0.1, 0.3, 0.5, 0.7, 0.9])
    # Keep the item-side point and the codeword-side tangent fixed across rows. The
    # fitted exponent then isolates the p powers: `cross` and `tau_term` are
    # coefficient * p**power. With random per-row dots the coefficients h1_i·tau_i
    # and ||tau_i||^2 add their own slope, so an exact 1e-6 assertion would fail even
    # for a correct scaling (the report is a fitted residual, not an identity).
    hidden = hidden_fixture(rows=1).repeat(5, 1)
    codebook = codebook_fixture(codes=1).repeat(5, 1)
    readings = train_rqvae.codeword_report(table, hidden, [codebook, codebook])
    assert readings["items"] == 5
    # s_hi^2 + s_lo^2 = p + (1 - p) = 1: the `4_method` log item, measured not assumed
    assert readings["scale_sq_sum_rel_error_max"] < 1e-6
    # the effective codeword radius grows with p, so the ratio is monotone in p
    assert readings["scale_ratio_p_spearman"] is not None
    assert readings["scale_ratio_p_spearman"] > 0.99
    # c -> 0 expansion: ||x||^2 ~ p^0, cross term ~ p^{1/2}, ||tau||^2 ~ p^1
    assert abs(readings["power_slope_cross"] - 0.5) < 1e-6
    assert abs(readings["power_slope_tau"] - 1.0) < 1e-6
    assert readings["power_slope_residual"] < 1e-6
    assert set(readings["levels"]) == {"level0", "level1"}
    for level in readings["levels"].values():
        assert set(level["g_hi_over_radius_percentiles"]) == {"0", "1", "25", "50",
                                                             "75", "99", "100"}
        assert level["nan_count"] == 0
    # a tiny codebook stays far inside the balls: no `project` saturation
    assert sum(level["clip_count"] for level in readings["levels"].values()) == 0


def test_codeword_report_counts_project_saturation_instead_of_assuming_it():
    table = share_table([0.1, 0.5, 0.9])
    hidden = hidden_fixture(rows=3)
    # |g| = sqrt(p) * |tau| with |tau| ~ 20: well past the ball radius 1/sqrt(4) = 0.5
    codebook = codebook_fixture(codes=4, scale=20.0)
    readings = train_rqvae.codeword_report(table, hidden, codebook)
    clips = sum(level["clip_count"] for level in readings["levels"].values())
    assert clips > 0
    # saturated points sit exactly at the project limit, never outside the ball
    limit = (1.0 - float(geometry.contract().BALL_EPS)) / float(geometry.C_HI) ** 0.5
    for level in readings["levels"].values():
        assert level["v_hi_norm_percentiles"]["100"] <= limit + 1e-9


def test_codeword_report_refuses_a_share_that_cannot_be_indexed():
    table = share_table([0.5, 0.5])
    with np.testing.assert_raises(RuntimeError):
        train_rqvae.codeword_report(table, hidden_fixture(rows=4), codebook_fixture(), sample=4)


# --------------------------------------------------------------------------------------
# argmin_flip_vs_behaviour_off + sid_diff_rate
# --------------------------------------------------------------------------------------

def _model(shares) -> RQVAE:
    torch.manual_seed(0)
    model = RQVAE(tokenizer_config(), in_dim=8, distance=adapter.distances,
                  shares=torch.as_tensor(shares, dtype=torch.float32),
                  share_mode="both_on")
    model.eval()
    return model


def test_assignment_report_counts_per_level_flips_and_the_sid_difference_rate():
    shares = torch.linspace(0.1, 0.9, 16)
    model = _model(shares)
    embeddings = torch.randn(16, 8)
    ids = torch.arange(16, dtype=torch.long)
    readings = train_rqvae.assignment_report(model, embeddings, ids, probe=16)
    assert readings["items"] == 16
    flips = readings["argmin_flip_vs_behaviour_off"]
    # codebook_num=2: level 0 is the argmin level, level 1 carries the Sinkhorn feed
    assert [layer.use_sk for layer in model.rq.vq_layers] == [False, True]
    assert set(flips) == {"level0_argmin", "level1_sinkhorn"}
    assert all(0 <= count <= 16 for count in flips.values())
    expected_rate = float(np.mean([count / 16.0 for count in flips.values()]))
    assert abs(readings["argmin_flip_rate"] - expected_rate) < 1e-6
    assert 0.0 <= readings["sid_diff_rate"] <= 1.0
    # the diagnostic must not leave the model on the off branch
    assert model.share_mode == "both_on"


def test_sid_diff_rate_is_zero_when_the_share_is_already_the_reference():
    # p_i == SHARE_REF for every item: the off branch is the same forward, so a
    # non-zero count would mean the switch (not the share) is driving the assignment.
    shares = torch.full((16,), float(geometry.SHARE_REF))
    model = _model(shares)
    readings = train_rqvae.assignment_report(model, torch.randn(16, 8),
                                             torch.arange(16, dtype=torch.long), probe=16)
    assert readings["sid_diff_rate"] == 0.0
    assert sum(readings["argmin_flip_vs_behaviour_off"].values()) == 0


# --------------------------------------------------------------------------------------
# the remaining read-only statistics
# --------------------------------------------------------------------------------------

def test_geometry_readout_measures_the_limit_equivariance_and_dimension_control():
    table = share_table([0.1, 0.3, 0.5, 0.7, 0.9])
    hidden = hidden_fixture()
    codebook = codebook_fixture()
    readings = train_rqvae.geometry_readout(table, hidden, codebook)
    # the c -> 0 limit KEEPS the scalings, so it cannot equal the baseline expansion
    assert readings["euclid_limit_rel_error_max"] > 1e-3
    # the diagnostic restores the fixed curvature ladder after the temporary c -> 0
    assert float(geometry.C_HI) == 4.0
    assert float(geometry.C_LO) == 0.25
    # a permutation of the batch permutes D and nothing else
    assert readings["batch_equivariance_max_abs_diff"] < 1e-9
    assert readings["dimension_control_rel_change"] > 0.0
    scale = readings["beta_effective_scale"]
    assert scale["commitment_over_baseline_mse"] > 0.0
    assert scale["codebook_over_baseline_mse"] > 0.0
    assert "factor_cost_vs_share_spearman" in readings


def test_geometry_readout_is_row_separable_and_leaves_the_constants_alone():
    table = share_table([0.2, 0.8])
    hidden = hidden_fixture(rows=2)
    codebook = codebook_fixture()
    train_rqvae.geometry_readout(table, hidden, codebook)
    assert float(geometry.C_HI) == 4.0 and float(geometry.C_LO) == 0.25
    # no (i, j) term: moving row 0's share cannot change row 1's D
    baseline = geometry.product_cost(hidden, codebook, torch.tensor([0.2, 0.8]))
    moved = geometry.product_cost(hidden, codebook, torch.tensor([0.7, 0.8]))
    assert torch.allclose(baseline[1], moved[1], rtol=0, atol=1e-12)
    assert not torch.allclose(baseline[0], moved[0])


# --------------------------------------------------------------------------------------
# the trainer must actually call them (a metric nobody calls is not a reading)
# --------------------------------------------------------------------------------------

def test_the_trainer_calls_the_three_reports():
    source = inspect.getsource(train_rqvae.main)
    for name in ("codeword_report(", "assignment_report(", "geometry_readout("):
        assert name in source
    assert "scale_sq_sum_rel_error" in source   # the `4_method` log line
    assert "codeword_scale" in source and "assignment_wiring" in source


# --------------------------------------------------------------------------------------
# the p of each probed row is ITS OWN p_i: the share is indexed by item id, not by row
# --------------------------------------------------------------------------------------

def test_codeword_report_pairs_each_probed_row_with_its_own_item_share():
    u = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    table = share_table(u)
    hidden = hidden_fixture(rows=3)
    codebook = codebook_fixture()
    ids = torch.tensor([7, 1, 4], dtype=torch.long)
    got = train_rqvae.codeword_report(table, hidden, [codebook], item_ids=ids)
    # the same three rows, described by a table whose first three entries ARE those
    # shares: reading `p_all[:count]` instead of `p_all[ids]` cannot reproduce this
    expected = train_rqvae.codeword_report(
        share_table([u[int(i)] for i in ids]), hidden, [codebook])
    for key in ("v_hi_norm_percentiles", "v_lo_norm_percentiles",
                "g_hi_over_radius_percentiles", "g_lo_over_radius_percentiles"):
        np.testing.assert_allclose(list(got["levels"]["level0"][key].values()),
                                   list(expected["levels"]["level0"][key].values()),
                                   rtol=0, atol=1e-12)
    for key in ("scale_ratio_p_spearman", "power_slope_cross", "power_slope_tau"):
        if expected[key] is None:
            assert got[key] is None
        else:
            assert got[key] == pytest.approx(expected[key], rel=0, abs=1e-9)
    assert (got["levels"]["level0"]["clip_count"]
            == expected["levels"]["level0"]["clip_count"])


def test_geometry_readout_pairs_each_probed_row_with_its_own_item_share():
    u = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    table = share_table(u)
    hidden = hidden_fixture(rows=3)
    codebook = codebook_fixture()
    ids = torch.tensor([7, 1, 4], dtype=torch.long)
    got = train_rqvae.geometry_readout(table, hidden, codebook, item_ids=ids)
    expected = train_rqvae.geometry_readout(
        share_table([u[int(i)] for i in ids]), hidden, codebook)
    assert got["euclid_limit_rel_error_max"] == pytest.approx(
        expected["euclid_limit_rel_error_max"], rel=0, abs=1e-9)
    for key in ("commitment_over_baseline_mse", "codebook_over_baseline_mse"):
        assert got["beta_effective_scale"][key] == pytest.approx(
            expected["beta_effective_scale"][key], rel=0, abs=1e-9)
    if expected["factor_cost_vs_share_spearman"] is not None:
        assert got["factor_cost_vs_share_spearman"] == pytest.approx(
            expected["factor_cost_vs_share_spearman"], rel=0, abs=1e-6)


def test_the_reports_refuse_item_ids_that_do_not_cover_the_probed_rows():
    table = share_table([0.2, 0.4, 0.6, 0.8])
    hidden = hidden_fixture(rows=3)
    codebook = codebook_fixture()
    with np.testing.assert_raises(RuntimeError):
        train_rqvae.codeword_report(table, hidden, [codebook],
                                    item_ids=torch.tensor([0, 1], dtype=torch.long))
    with np.testing.assert_raises(RuntimeError):
        train_rqvae.codeword_report(table, hidden, [codebook],
                                    item_ids=torch.tensor([0, 1, 9], dtype=torch.long))
    with np.testing.assert_raises(RuntimeError):
        train_rqvae.geometry_readout(table, hidden, codebook,
                                     item_ids=torch.tensor([0, 1], dtype=torch.long))
    with np.testing.assert_raises(RuntimeError):
        train_rqvae.geometry_readout(table, hidden, codebook,
                                     item_ids=torch.tensor([0, 1, 9], dtype=torch.long))


def test_the_trainer_hands_the_probe_ids_to_every_row_reading():
    source = inspect.getsource(train_rqvae.main)
    # `codeword_report` and `geometry_readout` measure the same probe rows whose shares
    # `energy_conservation_report` / `assignment_report` already resolve by item id.
    assert source.count("item_ids=probe_item_ids") >= 2
    assert "probe_item_ids = train_item_ids[probe_ids]" in source
