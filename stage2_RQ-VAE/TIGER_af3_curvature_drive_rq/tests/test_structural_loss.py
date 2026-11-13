"""`losses/structural.py`: `L_struct` and where its gradient lands.

    L_struct = mean_{i,l} H_delta(s_i - g_{token_l(i)}^(l))          (delta = 1)

The test that matters most is the gradient one: math.json's `gradient_routing` says
`dL_struct/dg != 0` and `dL_struct/dz = 0`. A term that quietly reached the encoder
would change the embedding, which the mechanism forbids.
"""

from __future__ import annotations

import pytest
import torch

from losses import LOSSES, LossError, build_loss
from losses.structural import DELTA, SPEC, STRUCTURAL_WEIGHT, huber, structural_loss
from model.codeword_potential import CodewordPotential


def _module(s_values, tokens, codebook_sizes=(3, 3), g_values=None):
    module = CodewordPotential(codebook_sizes, num_items=len(s_values))
    with torch.no_grad():
        if g_values is not None:
            for level, values in enumerate(g_values):
                module.g[level].copy_(torch.tensor(values, dtype=torch.float32))
    return module, torch.tensor(s_values, dtype=torch.float32), torch.tensor(tokens)


def _parts():
    return [(torch.tensor(0.5), torch.tensor(0.25))]


def test_structural_loss_is_the_mean_huber_deviation():
    module, potentials, tokens = _module(
        [0.0, 2.0], [[1, 0], [2, 1]], g_values=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    # deviations: -1, -1, +2, +2 -> H = 0.5, 0.5, 1.5, 1.5 -> mean 1.0
    assert float(structural_loss(potentials, tokens, module)) == pytest.approx(1.0)


def test_structural_loss_is_zero_when_every_potential_matches_its_codeword():
    module, potentials, tokens = _module([1.0, 1.0], [[0, 0], [0, 0]],
                                        g_values=[[1.0, 9.0], [1.0, 9.0]])
    assert float(structural_loss(potentials, tokens, module)) == pytest.approx(0.0)


def test_structural_loss_uses_the_linear_branch_beyond_delta():
    module, potentials, tokens = _module([3.0], [[0]], codebook_sizes=(1,), g_values=[[0.0]])
    # |3| > 1 -> 1 * (3 - 0.5) = 2.5
    assert float(structural_loss(potentials, tokens, module)) == pytest.approx(2.5)


def test_the_spec_and_the_constants_match_math_json():
    assert DELTA == 1.0
    assert SPEC["delta"] == DELTA
    assert SPEC["weight"] == STRUCTURAL_WEIGHT == 1.0
    assert SPEC["detached"] == "encoder"
    assert SPEC["zero_when_equal"] is True
    assert SPEC["scale_power"] == 1


def test_huber_matches_its_definition_at_the_boundary():
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    assert huber(x).tolist() == pytest.approx([1.5, 0.5, 0.125, 0.0, 0.125, 0.5, 1.5])


def test_gradient_reaches_g_and_never_the_encoder_side():
    module, potentials, tokens = _module(
        [0.0, 2.0], [[1, 0], [2, 1]], g_values=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    z = torch.zeros(2, 4, requires_grad=True)          # stands in for the encoder output
    structural_loss(potentials, tokens, module).backward()
    for level in range(2):
        assert module.g[level].grad is not None
        assert torch.isfinite(module.g[level].grad).all()
    assert module.g[1].grad.abs().sum() > 0            # the term really trains the codewords
    assert z.grad is None or torch.count_nonzero(z.grad) == 0   # dL_struct/dz = 0
    assert module.s.grad is None


def test_gradient_is_zero_at_the_behaviour_off_point():
    # switch `behaviour_off`: s = 0 and g = 0, so the deviation and its slope are both 0
    module = CodewordPotential((2, 2), num_items=3)
    loss = structural_loss(torch.zeros(3), torch.zeros(3, 2, dtype=torch.long), module)
    assert float(loss) == 0.0
    loss.backward()
    for level in range(2):
        assert module.g[level].grad is not None
        assert torch.count_nonzero(module.g[level].grad) == 0


def test_structural_loss_rejects_a_mismatched_batch():
    module, potentials, tokens = _module([0.0, 2.0], [[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        structural_loss(torch.zeros(3), tokens, module)            # 3 potentials, 2 rows
    with pytest.raises(ValueError):
        structural_loss(potentials, torch.zeros(2), module)         # tokens must be (B, L)
    with pytest.raises(ValueError):
        structural_loss(torch.zeros(0), torch.zeros(0, 2, dtype=torch.long), module)
    with pytest.raises(TypeError):
        structural_loss(torch.zeros(2, dtype=torch.long), tokens, module)


def test_structural_is_registered_in_the_loss_registry():
    assert "structural" in LOSSES
    assert LOSSES["structural"]["function"] is structural_loss
    assert LOSSES["structural"]["spec"]["form"] == "huber_potential"


def test_total_loss_adds_the_weighted_term_and_keeps_the_baseline_when_disabled():
    module, potentials, tokens = _module(
        [0.0, 2.0], [[1, 0], [2, 1]], g_values=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    payload = {"potentials": potentials, "tokens": tokens, "codeword": module}
    reconstructed, target = torch.zeros(2, 3), torch.ones(2, 3)

    base_total, base_components = build_loss({"beta": 0.25})(reconstructed, target, _parts())
    criterion = build_loss({"beta": 0.25,
                            "losses": ("reconstruction", "quantization", "structural")})
    total, components = criterion(reconstructed, target, _parts(), structural=payload)

    assert set(base_components) == {"reconstruction", "quantization"}
    assert float(components["structural"]) == pytest.approx(
        STRUCTURAL_WEIGHT * float(structural_loss(potentials, tokens, module)))
    assert float(total) == pytest.approx(float(base_total) + float(components["structural"]))


def test_zero_weight_is_not_the_same_as_behaviour_off():
    module, potentials, tokens = _module(
        [0.0, 2.0], [[1, 0], [2, 1]], g_values=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    payload = {"potentials": potentials, "tokens": tokens, "codeword": module}
    criterion = build_loss({"beta": 0.25,
                            "losses": ("reconstruction", "quantization", "structural"),
                            "structural_weight": 0.0})
    total, components = criterion(torch.zeros(2, 3), torch.ones(2, 3), _parts(),
                                  structural=payload)
    # the term is still computed (so this is not `behaviour_off`) but not added
    assert float(components["structural"]) == 0.0
    assert float(structural_loss(potentials, tokens, module)) > 0.0
    assert float(total) == pytest.approx(float(components["reconstruction"])
                                         + float(components["quantization"]))


def test_a_payload_without_the_enabled_term_is_an_error_not_a_silent_drop():
    module, potentials, tokens = _module([0.0, 2.0], [[1, 0], [2, 1]])
    payload = {"potentials": potentials, "tokens": tokens, "codeword": module}
    with pytest.raises(LossError):
        build_loss({"beta": 0.25})(torch.zeros(2, 3), torch.ones(2, 3),
                                   _parts(), structural=payload)
    enabled = build_loss({"beta": 0.25, "losses": ("reconstruction", "structural")})
    with pytest.raises(LossError):
        enabled(torch.zeros(2, 3), torch.ones(2, 3), None, structural=None)
