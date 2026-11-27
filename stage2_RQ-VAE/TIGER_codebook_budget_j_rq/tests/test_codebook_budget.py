"""Known-answer and wiring tests for the issue #272 round-3 codebook budget."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from model.budget import scale_codebook_terms


class Tiny(torch.nn.Module):
    """A minimal model: one codebook vector and one encoder scalar."""

    def __init__(self, with_constraint=True):
        super().__init__()
        self.cb = torch.nn.Parameter(torch.tensor([1.0, 2.0, -1.0]))
        self.enc = torch.nn.Parameter(torch.tensor([0.5]))
        self.with_constraint = with_constraint

    def terms(self):
        local = self.cb.sum() + 3.0 * self.enc
        if self.with_constraint:
            constraint = (self.cb ** 2).sum() + 0.0 * self.enc
        else:
            constraint = self.cb.new_zeros(())
        recon = (0.5 * self.cb).sum() + 2.0 * self.enc
        return local, constraint, recon


def _run(beta, with_constraint=True):
    model = Tiny(with_constraint=with_constraint)
    model.zero_grad(set_to_none=True)
    local, constraint, recon = model.terms()
    total = recon + local + constraint
    total.backward(retain_graph=True)
    scale_codebook_terms(local, constraint, [model.cb], beta)
    return model, local, constraint, recon


def test_beta_one_keeps_the_total_gradient():
    model, *_ = _run(1.0)
    assert torch.allclose(model.cb.grad, torch.tensor([3.5, 5.5, -0.5]), atol=1e-7)
    assert torch.allclose(model.enc.grad, torch.tensor([5.0]), atol=1e-7)


def test_beta_zero_leaves_only_the_reconstruction_codebook_gradient():
    model, *_ = _run(0.0)
    assert torch.allclose(model.cb.grad, torch.tensor([0.5, 0.5, 0.5]), atol=1e-7)
    assert torch.allclose(model.enc.grad, torch.tensor([5.0]), atol=1e-7)


def test_beta_one_percent_matches_the_hand_computed_codebook_gradient():
    model, *_ = _run(0.01)
    expected = torch.tensor([0.5 + 0.01 * 3.0, 0.5 + 0.01 * 5.0, 0.5 + 0.01 * (-1.0)])
    assert torch.allclose(model.cb.grad, expected, atol=1e-7)
    # The encoder keeps the full local+constraint+reconstruction gradient.
    assert torch.allclose(model.enc.grad, torch.tensor([5.0]), atol=1e-7)


def test_inactive_constraint_is_a_noop():
    model, *_ = _run(0.01, with_constraint=False)
    expected = torch.tensor([0.5 + 0.01 * 1.0, 0.5 + 0.01 * 1.0, 0.5 + 0.01 * 1.0])
    assert torch.allclose(model.cb.grad, expected, atol=1e-7)


def test_missing_codebook_gradient_is_skipped():
    model = Tiny()
    model.zero_grad(set_to_none=True)
    local, constraint, recon = model.terms()
    total = recon + local + constraint
    total.backward(retain_graph=True)
    # A fresh parameter with no .grad and no graph connection to the terms.
    detached = torch.nn.Parameter(torch.zeros(2))
    record = scale_codebook_terms(local, constraint, [detached], 0.01)
    assert record["touched"] == 0
    # The real codebook is untouched because it is not in the parameter list.
    assert torch.allclose(model.cb.grad, torch.tensor([3.5, 5.5, -0.5]), atol=1e-7)


def test_invalid_beta_and_empty_parameters_raise():
    model = Tiny()
    model.zero_grad(set_to_none=True)
    local, constraint, recon = model.terms()
    (recon + local + constraint).backward(retain_graph=True)
    with pytest.raises(ValueError):
        scale_codebook_terms(local, constraint, [model.cb], -0.1)
    with pytest.raises(ValueError):
        scale_codebook_terms(local, constraint, [model.cb], 1.1)
    with pytest.raises(ValueError):
        scale_codebook_terms(local, constraint, [], 0.01)


def test_reference_composition_matches_the_helper():
    model = Tiny()
    model.zero_grad(set_to_none=True)
    local, constraint, recon = model.terms()
    (recon + local + constraint).backward(retain_graph=True)
    # Independent path: compose the three separately differentiated gradients.
    g_recon = torch.autograd.grad(recon, [model.cb], retain_graph=True)[0]
    g_local = torch.autograd.grad(local, [model.cb], retain_graph=True)[0]
    g_constraint = torch.autograd.grad(constraint, [model.cb], retain_graph=True)[0]
    expected = g_recon + 0.01 * (g_local + g_constraint)
    scale_codebook_terms(local, constraint, [model.cb], 0.01)
    assert torch.allclose(model.cb.grad, expected, atol=1e-7)
    assert torch.allclose(model.enc.grad, torch.tensor([5.0]), atol=1e-7)
