"""The AF3 component-share distance: the formulas of math.json and the two switches.

Every numeric check compares the tree's implementation against a reference written
here from the spec text (``math.json`` formulas ``energy_split`` / ``product_cost`` /
``euclidean_degeneracy``), not against the tree's own helpers. The mutant checks pin
*where* the per-item factor is applied: the square root on the CODEWORD side, none on
the item side.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

import runtime_context  # noqa: E402
from distances import DEFAULT, DISTANCES, build_distance  # noqa: E402
from distances import distance as D  # noqa: E402


# --------------------------------------------------------------------------------------
# reference ball operators, written from poincare_contract.py:85/:141/:153
# --------------------------------------------------------------------------------------

C_HI_REF = 4.0
C_LO_REF = 0.25
SCALE_REF = 0.25


def ref_expmap0(v: torch.Tensor, c: float) -> torch.Tensor:
    root = math.sqrt(c)
    norm = v.norm(dim=-1, keepdim=True)
    z = root * norm
    factor = torch.where(z < 1e-8, torch.ones_like(z),
                         torch.tanh(z) / z.clamp_min(1e-30))
    return factor * v


def ref_mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    return numerator / denominator.clamp_min(1e-30)


def ref_d2(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    root = math.sqrt(c)
    norm = ref_mobius_add(-x, y, c).norm(dim=-1)
    return ((2.0 / root) * torch.atanh((root * norm).clamp(max=1 - 1e-7))).pow(2)


def ref_product_cost(latent: torch.Tensor, codebook: torch.Tensor,
                     share: torch.Tensor) -> torch.Tensor:
    """`product_cost`: item side unscaled, codeword side scaled by sqrt(p)/sqrt(1-p)."""
    x_e, x_h1, x_h2 = latent[:, :16], latent[:, 16:24], latent[:, 24:]
    t_e, t_h1, t_h2 = codebook[:, :16], codebook[:, 16:24], codebook[:, 24:]
    cost = (x_e.unsqueeze(1) - t_e.unsqueeze(0)).pow(2).sum(-1)
    s_hi = torch.sqrt(share).view(-1, 1, 1)
    s_lo = torch.sqrt(1.0 - share).view(-1, 1, 1)
    x_hi = ref_expmap0(x_h1, C_HI_REF).unsqueeze(1)
    x_lo = ref_expmap0(x_h2, C_LO_REF).unsqueeze(1)
    v_hi = ref_expmap0(s_hi * t_h1.unsqueeze(0), C_HI_REF)
    v_lo = ref_expmap0(s_lo * t_h2.unsqueeze(0), C_LO_REF)
    return cost + SCALE_REF * (ref_d2(x_hi, v_hi, C_HI_REF) + ref_d2(x_lo, v_lo, C_LO_REF))


def ref_item_scaled_cost(latent: torch.Tensor, codebook: torch.Tensor,
                         share: torch.Tensor) -> torch.Tensor:
    """A mutant that also scales the ITEM side -- must not be what the tree computes."""
    x_e, x_h1, x_h2 = latent[:, :16], latent[:, 16:24], latent[:, 24:]
    t_e, t_h1, t_h2 = codebook[:, :16], codebook[:, 16:24], codebook[:, 24:]
    cost = (x_e.unsqueeze(1) - t_e.unsqueeze(0)).pow(2).sum(-1)
    s_hi = torch.sqrt(share).view(-1, 1, 1)
    s_lo = torch.sqrt(1.0 - share).view(-1, 1, 1)
    x_hi = ref_expmap0(s_hi * x_h1.unsqueeze(1), C_HI_REF).squeeze(1).unsqueeze(1)
    x_lo = ref_expmap0(s_lo * x_h2.unsqueeze(1), C_LO_REF).squeeze(1).unsqueeze(1)
    v_hi = ref_expmap0(s_hi * t_h1.unsqueeze(0), C_HI_REF)
    v_lo = ref_expmap0(s_lo * t_h2.unsqueeze(0), C_LO_REF)
    return cost + SCALE_REF * (ref_d2(x_hi, v_hi, C_HI_REF) + ref_d2(x_lo, v_lo, C_LO_REF))


def ref_p_scaled_cost(latent: torch.Tensor, codebook: torch.Tensor,
                      share: torch.Tensor) -> torch.Tensor:
    """A mutant that scales the codeword by p instead of sqrt(p) -- the old bug."""
    x_e, x_h1, x_h2 = latent[:, :16], latent[:, 16:24], latent[:, 24:]
    t_e, t_h1, t_h2 = codebook[:, :16], codebook[:, 16:24], codebook[:, 24:]
    cost = (x_e.unsqueeze(1) - t_e.unsqueeze(0)).pow(2).sum(-1)
    s_hi = share.view(-1, 1, 1)
    s_lo = (1.0 - share).view(-1, 1, 1)
    x_hi = ref_expmap0(x_h1, C_HI_REF).unsqueeze(1)
    x_lo = ref_expmap0(x_h2, C_LO_REF).unsqueeze(1)
    v_hi = ref_expmap0(s_hi * t_h1.unsqueeze(0), C_HI_REF)
    v_lo = ref_expmap0(s_lo * t_h2.unsqueeze(0), C_LO_REF)
    return cost + SCALE_REF * (ref_d2(x_hi, v_hi, C_HI_REF) + ref_d2(x_lo, v_lo, C_LO_REF))


def ref_euclidean(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """The frozen baseline expansion of stage2_RQ-VAE/TIGER_RQ-VAE/distances/distance.py."""
    return (latent.pow(2).sum(1, keepdim=True)
            - 2 * latent @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t())


def make_pair(rows: int = 3, codebook: int = 5, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    latent = 0.2 * torch.randn(rows, 32, generator=generator, dtype=torch.float64)
    words = 0.2 * torch.randn(codebook, 32, generator=generator, dtype=torch.float64)
    return latent, words


@pytest.fixture
def small_table(monkeypatch):
    """A three-item frozen table; the real split is never read."""
    shares = np.array([0.1, 0.9, 0.5], dtype=np.float64)
    frozen = {
        "item_ids": np.arange(3, dtype=np.int64),
        "p": shares,
        "kappa_bar": np.array([2.0, -1.0, 0.5], dtype=np.float64),
        "degrees": np.array([3.0, 3.0, 3.0], dtype=np.float64),
        "num_items": 3,
        "digest": "unit-test",
    }
    monkeypatch.setattr(D, "table", lambda path=None: frozen)
    monkeypatch.setattr(D, "_TABLE", None)
    return frozen


@pytest.fixture(autouse=True)
def restore_wiring():
    runtime_context.clear()
    D.set_switches(behaviour_off=False, geometry_off=False)
    yield
    D.set_switches(behaviour_off=False, geometry_off=False)
    runtime_context.clear()


# --------------------------------------------------------------------------------------
# constants and registry (math.json constants, idea 4_method)
# --------------------------------------------------------------------------------------

def test_frozen_constants_match_math_json():
    assert (D.D_E, D.D_H1, D.D_H2) == (16, 8, 8)
    assert D.D_E + D.D_H1 + D.D_H2 == 32
    assert D.C_HI == 4.0 and D.C_LO == 0.25
    assert D.SCALE_HYP == 0.25
    assert D.SHARE_REF == 0.5
    assert D.SK_EPSILON == 0.003 and D.SK_ITERS == 50
    assert D.AF3_FACTOR_MASS is True
    assert D.NAME == "euclidean"  # spec: the module keeps its name


def test_registry_keeps_euclidean_and_trains_the_mechanism():
    assert DEFAULT == "af3_factor_mass"
    assert set(DISTANCES) >= {"euclidean", "af3_factor_mass"}
    # the `euclidean` entry is the baseline 32-d expansion (geometry_off reference)
    assert DISTANCES["euclidean"]["function"] is D.euclidean
    latent, words = make_pair()
    assert torch.allclose(DISTANCES["euclidean"]["function"](latent, words),
                          ref_euclidean(latent, words), rtol=0, atol=0)
    built = build_distance()
    assert built is D.distances
    with pytest.raises(Exception):
        build_distance(name="no_such_metric")


def test_distance_module_does_not_own_the_allocation():
    # formula `assignment`: Sinkhorn/argmin stay in model/layers.py
    for name in ("sinkhorn", "argmin", "assignment", "center_distance"):
        assert not hasattr(D, name)


# --------------------------------------------------------------------------------------
# formula `af3_potential` / `share_map`
# --------------------------------------------------------------------------------------

def test_share_map_is_the_rank_share():
    potentials = np.array([5.0, 1.0, 3.0, 3.0])
    ranks = np.array([4.0, 1.0, 2.5, 2.5])  # ties take the average rank
    expected = (ranks - 0.5) / 4.0
    assert np.allclose(D.shares_from_potentials(potentials), expected)
    # monotone in the potential, inside (0, 1)
    shares = D.shares_from_potentials(potentials)
    order = np.argsort(potentials)
    assert np.all(np.diff(shares[order]) >= 0.0)
    assert shares.min() > 0.0 and shares.max() < 1.0


def test_curvature_and_potential_formula():
    # kappa_ij = 4 - d_i - d_j + 3 T_ij, on a 4-node path 0-1-2-3
    rows = np.array([0, 1, 2])
    cols = np.array([1, 2, 3])
    matrix = D.adjacency(rows, cols, 4)
    curvature = D.edge_curvatures(matrix, rows, cols)
    # degrees 1,2,2,1; no triangles -> kappa = 4 - d_i - d_j
    assert np.allclose(curvature, np.array([4 - 1 - 2, 4 - 2 - 2, 4 - 2 - 1], dtype=float))
    potentials = D.structural_potentials(matrix, rows, cols, curvature)
    assert np.allclose(potentials, np.array([1.0, 0.5, 0.5, 1.0]))


# --------------------------------------------------------------------------------------
# ball operators and the c -> 0 degeneracy
# --------------------------------------------------------------------------------------

def test_expmap_and_squared_distance_degenerate_to_euclidean_as_c_to_zero():
    latent, _ = make_pair(rows=4)
    v = latent[:, :8].clone()
    y = latent[:, 8:16].clone()

    assert torch.allclose(D.expmap0(v, 0.0), v)
    assert torch.allclose(D.squared_distance(v, y, 0.0), 4.0 * (v - y).pow(2).sum(-1))

    c = 1e-6
    lifted = D.SCALE_HYP * D.squared_distance(D.expmap0(v, c), D.expmap0(y, c), c)
    assert torch.allclose(lifted, (v - y).pow(2).sum(-1), rtol=0, atol=1e-5)


def test_ball_operators_match_the_contract_reference():
    latent, _ = make_pair(rows=6)
    v = 0.3 * latent[:, :8]
    y = 0.3 * latent[:, 8:16]
    for c in (C_HI_REF, C_LO_REF):
        assert torch.allclose(D.expmap0(v, c), ref_expmap0(v, c), rtol=0, atol=1e-12)
        assert torch.allclose(D.squared_distance(v, y, c), ref_d2(v, y, c),
                              rtol=0, atol=1e-10)
    # expmap0 stays inside the ball of radius (1 - eps)/sqrt(c)
    limit = (1.0 - D.BALL_EPS) / math.sqrt(C_HI_REF)
    assert float(D.expmap0(50.0 * v, C_HI_REF).norm(dim=-1).max()) <= limit + 1e-12


# --------------------------------------------------------------------------------------
# formula `latent_split` / `energy_split` / `product_cost`
# --------------------------------------------------------------------------------------

def test_product_cost_matches_the_reference_formula():
    latent, words = make_pair(seed=1)
    share = torch.tensor([0.1, 0.35, 0.8], dtype=torch.float64)
    cost = D._product_cost(latent, words, share.view(-1, 1, 1))
    reference = ref_product_cost(latent, words, share)
    assert cost.shape == (3, 5)
    assert torch.allclose(cost, reference, rtol=0, atol=1e-10)


def test_code_word_side_scales_by_sqrt_p_and_item_side_is_unscaled():
    latent, words = make_pair(seed=2)
    share = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    cost = D._product_cost(latent, words, share.view(-1, 1, 1))
    # the item side carries no factor (energy_split: item side unscaled)
    assert not torch.allclose(cost, ref_item_scaled_cost(latent, words, share),
                              rtol=0, atol=1e-8)
    # the codeword side carries sqrt(p), not p (s_hi = sqrt(p_i))
    assert not torch.allclose(cost, ref_p_scaled_cost(latent, words, share),
                              rtol=0, atol=1e-8)


def test_cost_depends_on_the_share_and_on_the_latent_gradient():
    latent, words = make_pair(seed=3)
    moved = D.distance_depends_on_p(latent, words, 0.2, 0.8)
    assert moved["changed"] is True and moved["max_abs_diff"] > 1e-6

    probe = latent.clone().requires_grad_(True)
    D._product_cost(probe, words,
                    torch.full((probe.shape[0], 1, 1), 0.4, dtype=torch.float64)).sum().backward()
    assert probe.grad is not None and float(probe.grad.abs().sum()) > 0.0


def test_cost_has_no_pair_term():
    # formula `no_pair_signal`: row i depends only on row i of the latent and its p
    latent, words = make_pair(seed=4)
    share = torch.tensor([0.2, 0.6, 0.9], dtype=torch.float64)
    full = D._product_cost(latent, words, share.view(-1, 1, 1))
    for row in range(latent.shape[0]):
        single = D._product_cost(latent[row:row + 1], words, share[row].view(1, 1, 1))
        assert torch.allclose(single[0], full[row], rtol=0, atol=1e-12)


# --------------------------------------------------------------------------------------
# switch `geometry_off`
# --------------------------------------------------------------------------------------

def test_geometry_off_returns_the_baseline_cost_element_for_element(small_table):
    latent, words = make_pair(rows=4, seed=5)
    runtime_context.set_fields({"item_id": torch.tensor([0, 1, 2, 0])})
    D.set_switches(geometry_off=True)
    branch = D.distances(latent, words)
    reference = ref_euclidean(latent, words)
    assert torch.allclose(branch, reference, rtol=0, atol=1e-6)
    report = D.geometry_off_matches_baseline(latent, words)
    assert report["ok"] is True and report["max_abs_diff"] <= 1e-6
    # geometry_off must be the baseline, not the product cost
    D.set_switches(geometry_off=False)
    assert not torch.allclose(D.distances(latent, words), reference, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------------------
# switch `behaviour_off`
# --------------------------------------------------------------------------------------

def test_behaviour_off_pins_the_share_to_share_ref(small_table):
    latent, words = make_pair(rows=2, seed=6)
    runtime_context.set_fields({"item_id": torch.tensor([0, 1])})
    shares = torch.as_tensor(small_table["p"], dtype=torch.float64)

    on = D.distances(latent, words)
    assert torch.allclose(on, ref_product_cost(latent, words, shares[[0, 1]]),
                          rtol=0, atol=1e-10)

    D.set_switches(behaviour_off=True)
    off = D.distances(latent, words)
    assert torch.allclose(off, ref_product_cost(latent, words,
                                                torch.full((2,), D.SHARE_REF,
                                                           dtype=torch.float64)),
                          rtol=0, atol=1e-10)
    # the geometry stays (behaviour_off is not the baseline cost)
    assert not torch.allclose(off, ref_euclidean(latent, words), rtol=0, atol=1e-6)
    # the readout reports the constant share
    readout = D.readouts()
    assert readout["share_source"] == "SHARE_REF"
    assert readout["switches"]["behaviour_off"] is True
    assert all(abs(value - D.SHARE_REF) < 1e-12
               for value in readout["p_percentiles"].values())


def test_behaviour_off_does_not_need_the_frozen_table(monkeypatch):
    def explode(path=None):
        raise AssertionError("behaviour_off must not read the frozen share table")

    monkeypatch.setattr(D, "table", explode)
    monkeypatch.setattr(D, "_TABLE", None)
    latent, words = make_pair(rows=3, seed=7)
    D.set_switches(behaviour_off=True)
    cost = D.distances(latent, words)
    assert cost.shape == (3, 5)
    assert torch.allclose(cost, ref_product_cost(latent, words,
                                                 torch.full((3,), D.SHARE_REF,
                                                            dtype=torch.float64)),
                          rtol=0, atol=1e-10)


def test_switches_default_to_off():
    assert D.switches() == {"behaviour_off": False, "geometry_off": False}
    # behaviour_off alone keeps the product geometry (only p is pinned)
    # geometry_off alone returns the baseline expansion
    assert D.NAME == "euclidean" and D.MECHANISM == "af3_factor_mass"


def test_row_chunking_changes_no_value(small_table, monkeypatch):
    latent, words = make_pair(rows=4, seed=8)
    runtime_context.set_fields({"item_id": torch.tensor([0, 1, 2, 0])})
    whole = D.distances(latent, words)
    monkeypatch.setattr(D, "ROW_CHUNK", 1)
    chunked = D.distances(latent, words)
    assert torch.allclose(whole, chunked, rtol=0, atol=1e-12)


def test_share_rows_follow_the_item_id_order(small_table):
    # formula `share_map`: row i of D uses p_i of item_id[i], never the sorted order
    latent, words = make_pair(rows=3, seed=11)
    frozen = torch.as_tensor(small_table["p"], dtype=torch.float64)
    runtime_context.set_fields({"item_id": torch.tensor([2, 0, 1])})
    cost = D.distances(latent, words)
    assert torch.allclose(cost, ref_product_cost(latent, words, frozen[[2, 0, 1]]),
                          rtol=0, atol=1e-10)
    # the identity order is a different cost here, so the check is not vacuous
    assert not torch.allclose(cost, ref_product_cost(latent, words, frozen),
                              rtol=0, atol=1e-10)


# --------------------------------------------------------------------------------------
# boundary readout of the two factors (uses sqrt(p) on the codeword side)
# --------------------------------------------------------------------------------------

def ref_boundary_ratio(values: torch.Tensor, c: float) -> np.ndarray:
    """`||expmap0(v, c)|| / ((1 - BALL_EPS) / sqrt(c))`, from the contract formula."""
    limit = (1.0 - D.BALL_EPS) / math.sqrt(c)
    mapped = ref_expmap0(values, c)
    return (mapped.norm(dim=-1) / limit).flatten().numpy()


def test_ball_boundary_readout_uses_sqrt_share_on_the_codeword_side():
    latent, words = make_pair(rows=4, codebook=6, seed=12)
    share = 0.7
    report = D.saturation_readout(latent, words, share=share)

    # codeword side: sqrt(p) on the high factor, sqrt(1-p) on the low one
    hi_ref = ref_boundary_ratio(math.sqrt(share) * words[:, 16:24], C_HI_REF)
    lo_ref = ref_boundary_ratio(math.sqrt(1.0 - share) * words[:, 24:], C_LO_REF)
    for name, q in (("p05", 5.0), ("p50", 50.0), ("p95", 95.0)):
        assert abs(report["codeword_hi"]["quantiles"][name]
                   - float(np.percentile(hi_ref, q))) < 1e-12
        assert abs(report["codeword_lo"]["quantiles"][name]
                   - float(np.percentile(lo_ref, q))) < 1e-12

    # item side carries no factor at all
    item_hi = ref_boundary_ratio(latent[:, 16:24], C_HI_REF)
    item_lo = ref_boundary_ratio(latent[:, 24:], C_LO_REF)
    assert abs(report["item_hi"]["quantiles"]["p50"]
               - float(np.percentile(item_hi, 50.0))) < 1e-12
    assert abs(report["item_lo"]["quantiles"]["p50"]
               - float(np.percentile(item_lo, 50.0))) < 1e-12

    # the readout pins sqrt(p), not p (the old bug) -- the two differ materially
    hi_p = ref_boundary_ratio(share * words[:, 16:24], C_HI_REF)
    lo_p = ref_boundary_ratio((1.0 - share) * words[:, 24:], C_LO_REF)
    assert abs(report["codeword_hi"]["quantiles"]["p50"]
               - float(np.percentile(hi_p, 50.0))) > 1e-6
    assert abs(report["codeword_lo"]["quantiles"]["p50"]
               - float(np.percentile(lo_p, 50.0))) > 1e-6


def test_ball_boundary_readout_flags_a_saturated_point():
    latent, words = make_pair(rows=3, seed=13)
    report = D.ball_boundary_readout(50.0 * latent[:, :8], C_HI_REF)
    limit = (1.0 - D.BALL_EPS) / math.sqrt(C_HI_REF)
    assert abs(report["limit"] - limit) < 1e-12
    assert report["clipped_fraction"] == 1.0
    for value in report["quantiles"].values():
        assert abs(value - 1.0) < 1e-6
    # a point well inside the ball is not reported as clipped
    assert D.ball_boundary_readout(0.01 * latent[:, :8], C_HI_REF)["clipped_fraction"] == 0.0


def test_readouts_scaled_factors_follow_the_share_and_the_switch(small_table):
    shares = np.asarray(small_table["p"], dtype=np.float64)
    plain = D.readouts()
    assert abs(plain["scaled_hi_min"] - float(np.sqrt(shares).min())) < 1e-12
    assert abs(plain["scaled_lo_min"] - float(np.sqrt(1.0 - shares).min())) < 1e-12
    assert abs(plain["share_min"] - float(shares.min())) < 1e-12
    assert plain["share_source"] == "af3_potential_rank"

    D.set_switches(behaviour_off=True)
    off = D.readouts()
    assert abs(off["scaled_hi_min"] - math.sqrt(D.SHARE_REF)) < 1e-12
    assert abs(off["scaled_lo_min"] - math.sqrt(D.SHARE_REF)) < 1e-12
    assert abs(off["share_min"] - D.SHARE_REF) < 1e-12


# --------------------------------------------------------------------------------------
# fail-loud contract of `shares_for`
# --------------------------------------------------------------------------------------

def test_missing_or_out_of_range_item_id_fails_loud(small_table):
    with pytest.raises(ValueError):
        D.shares_for(None)
    with pytest.raises(ValueError):
        D.shares_for([0, 3])
    with pytest.raises(ValueError):
        D.shares_for([-1])
    latent, words = make_pair(rows=2, seed=9)
    runtime_context.clear()  # the host published no item_id
    with pytest.raises(ValueError):
        D.distances(latent, words)


# --------------------------------------------------------------------------------------
# frozen table: build once from the train split, atomic cache, digest check
# --------------------------------------------------------------------------------------

def test_table_is_built_from_train_only_and_cached_atomically(tmp_path, monkeypatch):
    split = tmp_path / "train.parquet"
    split.write_bytes(b"unit-test-train-split")
    monkeypatch.setattr(D, "train_parquet", lambda: split)

    frame = pd.DataFrame({
        "history": [[0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 5], [5, 6, 7]],
        "target": [5, 6, 7, 8, 9],
    })
    built = D.build_p_table(frame)
    shares = built["p"]
    assert shares.size == int(max(max(seq) for seq in D.sequences(frame))) + 1
    assert np.all(shares > 0.0) and np.all(shares < 1.0)
    assert np.allclose(shares, (D.average_ranks(built["kappa_bar"]) - 0.5) / shares.size)
    assert built["digest"] == D._split_digest(split)

    target = tmp_path / "af3_factor_mass_p.npz"
    D.write_p_table(built, target)
    assert target.is_file() and not target.with_name(target.name + ".tmp").exists()
    loaded = D.read_p_table(target)
    assert np.allclose(loaded["p"], shares)
    assert np.allclose(loaded["kappa_bar"], built["kappa_bar"])
    assert loaded["num_items"] == shares.size

    # a cache built from another split is refused, never silently rebuilt
    split.write_bytes(b"another-split")
    with pytest.raises(ValueError):
        D.read_p_table(target)
