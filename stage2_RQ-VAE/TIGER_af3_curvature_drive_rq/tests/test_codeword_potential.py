"""`model/codeword_potential.py`: `g`, `den_l`, `lambda_l` and the cost offset.

Hand-computed cases for the three formulas this module owns

    g_k^(l)   = mean{ s_i : token_l(i) = k }        (empty code stays 0)
    den_l     = median_i |s_i - g_{token_l(i)}^(l)|
    lambda_l  = median_i d(z_i, c_{token}) / den_l   (0 when den_l <= eps)
    bias_ik   = lambda_l * H_delta(s_i - g_k^(l))    (the `assignment_cost` offset)

plus the two things the mechanism depends on but the formulas do not spell out: the
offset must be detached (it only decides argmin) and `g` must be a real parameter
(DDP synchronises it, the structural loss trains it).
"""

from __future__ import annotations

import pytest
import torch

from model.codeword_potential import CodewordPotential, huber


def _prepared(s, tokens, distances, sizes=(2, 3)):
    module = CodewordPotential(sizes, num_items=len(s))
    module.set_potentials(torch.tensor(s, dtype=torch.float32))
    report = module.initialise_level(
        0, list(range(len(s))), torch.tensor(tokens), torch.tensor(distances, dtype=torch.float32)
    )
    return module, report


def test_codeword_potential_init_is_the_per_code_mean():
    module, _ = _prepared([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1], [1.0, 1.0, 1.0, 1.0])
    # code 0 holds s = {0, 1} -> 0.5; code 1 holds s = {2, 3} -> 2.5
    assert module.g[0].tolist() == pytest.approx([0.5, 2.5])


def test_empty_codes_stay_at_zero():
    module, _ = _prepared([0.0, 1.0, 2.0, 3.0], [0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0])
    assert module.g[0].tolist() == pytest.approx([1.5, 0.0])


def test_den_is_the_median_residual_and_lambda_its_ratio():
    module, report = _prepared([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1], [1.0, 1.0, 3.0, 3.0])
    # residuals |s - g_token| = [0.5, 0.5, 0.5, 0.5]; distance medians 0.5 and 2.0
    assert report["den"] == pytest.approx(0.5)
    assert report["lambda"] == pytest.approx(4.0)
    assert float(module.dens[0]) == pytest.approx(0.5)
    assert float(module.lambdas[0]) == pytest.approx(4.0)
    assert bool(module.initialised[0])


def test_lambda_is_zero_when_den_vanishes_and_the_switch_reports_itself():
    module, report = _prepared([1.0, 1.0, 1.0], [0, 0, 0], [5.0, 5.0, 5.0], sizes=(2, 2))
    assert report["den"] == pytest.approx(0.0)
    assert report["lambda"] == 0.0
    # every level still holds lambda = 0, which is exactly the `geometry_off` state
    assert module.geometry_off() is True


def test_geometry_flag_is_false_once_one_level_carries_a_scale():
    module, _ = _prepared([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1], [1.0, 1.0, 3.0, 3.0])
    assert module.geometry_off() is False
    module.lambdas[0] = 0.0
    assert module.geometry_off() is True


def test_offset_equals_lambda_times_huber_grid():
    module, _ = _prepared([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1], [1.0, 1.0, 3.0, 3.0])
    rows = torch.arange(4)
    offset = module.bias(0, rows)
    s = torch.tensor([0.0, 1.0, 2.0, 3.0])
    g = module.g[0].detach()
    expected = 4.0 * huber(s.unsqueeze(1) - g.unsqueeze(0))
    assert offset.shape == (4, 2)
    assert torch.allclose(offset, expected, atol=1e-6)


def test_zero_offset_when_the_potential_matches_the_codeword():
    # switch `behaviour_off`: s_i = g_k = 0, so H_delta(0) = 0 even with lambda > 0
    module = CodewordPotential((2, 2), num_items=4)
    module.set_potentials(torch.zeros(4))
    for parameter in module.g:
        parameter.data.zero_()
    module.lambdas[0] = 3.0
    offset = module.bias(0, torch.arange(4))
    assert torch.equal(offset, torch.zeros(4, 2))


def test_offset_is_detached_and_g_is_trainable():
    module, _ = _prepared([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1], [1.0, 1.0, 3.0, 3.0])
    offset = module.bias(0, torch.arange(4))
    assert not offset.requires_grad
    assert isinstance(module.g, torch.nn.ParameterList)
    assert module.g[0].requires_grad
    assert not module.s.requires_grad  # `s` is frozen


def test_gather_keeps_the_gradient_into_g():
    module, _ = _prepared([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1], [1.0, 1.0, 3.0, 3.0])
    gathered = module.gather(0, torch.tensor([1, 0, 1]))
    assert torch.allclose(gathered, torch.tensor([2.5, 0.5, 2.5]))
    gathered.sum().backward()
    assert module.g[0].grad is not None
    assert module.g[0].grad.tolist() == pytest.approx([1.0, 2.0])
    assert module.s.grad is None


def test_potentials_for_indexes_the_table_and_checks_the_range():
    module = CodewordPotential((2,), num_items=4)
    module.set_potentials(torch.tensor([10.0, 11.0, 12.0, 13.0]))
    assert module.potentials_for([3, 1]).tolist() == pytest.approx([13.0, 11.0])
    with pytest.raises(ValueError):
        module.potentials_for([4])
    with pytest.raises(ValueError):
        module.potentials_for(None)


def test_set_potentials_rejects_a_wrong_shape_or_non_finite_table():
    module = CodewordPotential((2,), num_items=3)
    with pytest.raises(ValueError):
        module.set_potentials(torch.zeros(4))
    with pytest.raises(ValueError):
        module.set_potentials(torch.tensor([0.0, float("nan"), 1.0]))


def test_metrics_report_both_calibration_numbers_per_level():
    module, _ = _prepared([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1], [1.0, 1.0, 3.0, 3.0])
    metrics = module.metrics()
    assert metrics["den_level1"] == pytest.approx(0.5)
    assert metrics["lambda_level1"] == pytest.approx(4.0)
    assert metrics["lambda_level2"] == pytest.approx(0.0)
    assert metrics["structural_delta"] == pytest.approx(1.0)


def test_levels_do_not_share_state():
    module = CodewordPotential((2, 3), num_items=4)
    module.set_potentials(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    module.initialise_level(0, [0, 1, 2, 3], torch.tensor([0, 0, 1, 1]), torch.ones(4))
    module.initialise_level(1, [0, 1, 2, 3], torch.tensor([0, 1, 2, 2]), torch.ones(4))
    assert module.g[0].tolist() == pytest.approx([0.5, 2.5])
    assert float(module.g[1][2]) == pytest.approx(2.5)
    assert module.bias(0, torch.arange(4)).shape == (4, 2)
    with pytest.raises(IndexError):
        module.bias(2, torch.arange(4))
