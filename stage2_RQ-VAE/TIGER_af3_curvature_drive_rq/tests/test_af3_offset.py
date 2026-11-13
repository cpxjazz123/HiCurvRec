"""`distances/af3_offset.py`: `C_ik = d(z_i, c_k) + lambda_l H_delta(s_i - g_k^(l))`.

Three things are checked here, in this order:

1. the formula, against `d` from the baseline module plus a hand-computed offset;
2. the two switches -- `geometry_off` (`lambda_l = 0`) and `behaviour_off`
   (`s_i = g_k = 0`) must each leave the baseline cost bit-for-bit, which is why the
   comparison is `torch.equal` and not an approximate one;
3. the interface the quantiser relies on: `needs_rows`, one view per level via
   `at_level`, and a registry entry that takes its runtime arguments as keywords.
"""

from __future__ import annotations

import pytest
import torch

from distances import DISTANCES, DistanceError, build_distance
from distances.af3_offset import NAME, Af3Offset, Af3OffsetLevel
from distances.distance import distances as euclidean
from model.codeword_potential import CodewordPotential

LATENT = torch.tensor([[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]])
CODEBOOK = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [0.25, 0.75]])


def _potential(s_values, lambdas=(2.0, 0.5), sizes=(4, 4), g_values=None):
    module = CodewordPotential(sizes, num_items=len(s_values))
    module.set_potentials(torch.tensor(s_values, dtype=torch.float32))
    for level, value in enumerate(lambdas):
        module.lambdas[level] = value
    with torch.no_grad():
        if g_values is not None:
            for level, values in enumerate(g_values):
                module.g[level].copy_(torch.tensor(values, dtype=torch.float32))
    return module


def test_cost_is_the_baseline_distance_plus_the_offset():
    potential = _potential([0.0, 1.0, -1.0], g_values=[[0.0, 0.5, -0.5, 1.0], [0.0] * 4])
    metric = Af3Offset(base=euclidean, potential=potential)
    level = metric.at_level(0)
    rows = torch.tensor([0, 1, 2])

    base = euclidean(LATENT, CODEBOOK)
    assert torch.allclose(level.base_cost(LATENT, CODEBOOK), base, atol=0, rtol=0)

    expected_offset = 2.0 * torch.nn.functional.huber_loss(
        torch.tensor([0.0, 1.0, -1.0]).unsqueeze(1)
        - torch.tensor([0.0, 0.5, -0.5, 1.0]).unsqueeze(0),
        torch.zeros(3, 4), reduction="none", delta=1.0)
    assert torch.allclose(level(LATENT, CODEBOOK, rows), base + expected_offset, atol=1e-6)


def test_default_rows_are_the_item_ids_of_the_rows():
    potential = _potential([0.0, 1.0, -1.0], g_values=[[0.0, 0.5, -0.5, 1.0], [0.0] * 4])
    level = Af3Offset(base=euclidean, potential=potential).at_level(0)
    assert torch.equal(level(LATENT, CODEBOOK),
                       level(LATENT, CODEBOOK, torch.arange(LATENT.shape[0])))


def test_geometry_off_leaves_the_baseline_cost_untouched():
    # lambda_l = 0 -> the AF3 term does not enter the cost, whatever s and g are
    potential = _potential([3.0, -2.0, 1.0], lambdas=(0.0, 0.0),
                           g_values=[[1.0, 2.0, 3.0, 4.0], [0.0] * 4])
    assert potential.geometry_off() is True
    for level in range(2):
        cost = Af3Offset(base=euclidean, potential=potential).at_level(level)(
            LATENT, CODEBOOK, torch.arange(3))
        assert torch.equal(cost, euclidean(LATENT, CODEBOOK))


def test_behaviour_off_leaves_the_baseline_cost_untouched():
    # s_i = 0 and g_k = 0 -> H_delta(0) = 0, so the offset vanishes even with lambda > 0
    potential = _potential([0.0, 0.0, 0.0], lambdas=(5.0, 5.0))
    assert potential.geometry_off() is False
    for level in range(2):
        cost = Af3Offset(base=euclidean, potential=potential).at_level(level)(
            LATENT, CODEBOOK, torch.arange(3))
        assert torch.equal(cost, euclidean(LATENT, CODEBOOK))


def test_a_positive_lambda_change_does_change_the_cost():
    potential = _potential([2.0, -2.0, 0.5], lambdas=(1.0, 1.0),
                           g_values=[[0.0, 0.0, 0.0, 0.0], [0.0] * 4])
    cost = Af3Offset(base=euclidean, potential=potential).at_level(0)(
        LATENT, CODEBOOK, torch.arange(3))
    assert not torch.equal(cost, euclidean(LATENT, CODEBOOK))


def test_levels_are_independent():
    potential = _potential([1.0, 2.0, 3.0], lambdas=(1.0, 1.0),
                           g_values=[[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]])
    metric = Af3Offset(base=euclidean, potential=potential)
    rows = torch.arange(3)
    first = metric.at_level(0)(LATENT, CODEBOOK, rows)
    second = metric.at_level(1)(LATENT, CODEBOOK, rows)
    assert not torch.equal(first, second)
    # changing level 1's codeword potential leaves level 0 alone
    with torch.no_grad():
        potential.g[1].fill_(0.0)
    assert torch.equal(metric.at_level(0)(LATENT, CODEBOOK, rows), first)
    assert not torch.equal(metric.at_level(1)(LATENT, CODEBOOK, rows), second)


def test_at_level_rejects_an_index_outside_the_stack():
    potential = _potential([0.0, 1.0, -1.0], sizes=(4, 4))
    metric = Af3Offset(base=euclidean, potential=potential)
    with pytest.raises(ValueError):
        metric.at_level(2)
    with pytest.raises(ValueError):
        metric.at_level(-1)


def test_the_metric_declares_that_it_needs_the_batch_rows():
    potential = _potential([0.0, 1.0, -1.0])
    metric = Af3Offset(base=euclidean, potential=potential)
    assert metric.needs_rows is True
    assert metric.has_levels is True
    assert metric.at_level(0).needs_rows is True


def test_a_level_rejects_an_offset_that_does_not_fit_the_cost():
    potential = _potential([0.0, 1.0, -1.0])
    level = Af3Offset(base=euclidean, potential=potential).at_level(0)
    with pytest.raises(ValueError):
        level(LATENT, CODEBOOK, torch.tensor([0, 1]))  # 2 ids for 3 rows


def test_initialise_level_calibrates_from_the_base_distance():
    potential = _potential([0.0, 1.0, -1.0], lambdas=(0.0, 0.0), g_values=[[0.0] * 4, [0.0] * 4])
    level = Af3Offset(base=euclidean, potential=potential).at_level(0)
    rows = torch.tensor([0, 1, 2])
    tokens = torch.argmin(euclidean(LATENT, CODEBOOK), dim=-1)
    report = level.initialise_level(LATENT, CODEBOOK, rows, tokens)
    base = euclidean(LATENT, CODEBOOK)
    assigned = base.gather(1, tokens.reshape(-1, 1)).squeeze(1)
    s = potential.potentials_for(rows)
    g = torch.zeros(4)
    g.index_add_(0, tokens, s)
    counts = torch.zeros(4)
    counts.index_add_(0, tokens, torch.ones(3))
    g[counts > 0] /= counts[counts > 0]
    residual = (s - g.index_select(0, tokens)).abs()
    assert report["den"] == pytest.approx(float(residual.median()))
    expected_lambda = 0.0 if float(residual.median()) <= 1e-8 else float(assigned.median()) / float(residual.median())
    assert report["lambda"] == pytest.approx(expected_lambda)


def test_registry_builds_the_factory_with_its_runtime_arguments():
    potential = _potential([0.0, 1.0, -1.0])
    assert "af3_offset" in DISTANCES and "factory" in DISTANCES["af3_offset"]
    metric = build_distance(None, "af3_offset", base=euclidean, potential=potential)
    assert isinstance(metric, Af3Offset)
    assert isinstance(metric.at_level(0), Af3OffsetLevel)
    assert metric.at_level(0).potential is potential


def test_registry_reports_unknown_names_and_keeps_the_baseline_default():
    with pytest.raises(DistanceError):
        build_distance(None, "af3_curvature_v2")
    assert build_distance() is euclidean
    assert build_distance({"distance": "euclidean"}) is euclidean


def test_the_name_and_description_are_the_registry_entry():
    assert NAME == "af3_offset"
    assert DISTANCES[NAME]["description"].startswith("squared Euclidean")
