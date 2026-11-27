"""Mutation checks for the issue #272 round-3 codebook budget.

Each mutation is a deliberate wrong variant of ``scale_codebook_terms``; the
specified assertion is the known-answer check used by
``tests/test_codebook_budget.py``.  A mutation is caught when the known answer
rejects it (a crash or an import error is not a catch).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.budget import scale_codebook_terms


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cb = torch.nn.Parameter(torch.tensor([1.0, 2.0, -1.0]))
        self.enc = torch.nn.Parameter(torch.tensor([0.5]))

    def terms(self):
        local = self.cb.sum() + 3.0 * self.enc
        constraint = (self.cb ** 2).sum() + 0.0 * self.enc
        recon = (0.5 * self.cb).sum() + 2.0 * self.enc
        return local, constraint, recon


def _state():
    model = Tiny()
    model.zero_grad(set_to_none=True)
    local, constraint, recon = model.terms()
    (recon + local + constraint).backward(retain_graph=True)
    return model, local, constraint, recon


def _expected_cb(beta=0.01):
    return torch.tensor([0.5 + beta * 3.0, 0.5 + beta * 5.0, 0.5 + beta * (-1.0)])


def _mut_beta_factor(local, constraint, params, beta):
    combined = local + constraint
    grads = torch.autograd.grad(combined, params, allow_unused=True)
    for p, g in zip(params, grads):
        if g is not None:
            p.grad.sub_(g, alpha=float(beta))  # wrong: beta instead of 1-beta


def _mut_sign(local, constraint, params, beta):
    combined = local + constraint
    grads = torch.autograd.grad(combined, params, allow_unused=True)
    for p, g in zip(params, grads):
        if g is not None:
            p.grad.add_(g, alpha=1.0 - float(beta))  # wrong: adds


def _mut_encoder(local, constraint, params, beta):
    combined = local + constraint
    grads = torch.autograd.grad(combined, params, allow_unused=True)
    for p, g in zip(params, grads):
        if g is not None:
            p.grad.sub_(g, alpha=1.0 - float(beta))  # wrong: caller passes the encoder too


def _mut_local_only(local, constraint, params, beta):
    grads = torch.autograd.grad(local, params, allow_unused=True)  # wrong: drops the constraint
    for p, g in zip(params, grads):
        if g is not None:
            p.grad.sub_(g, alpha=1.0 - float(beta))


def _mut_constraint_only(local, constraint, params, beta):
    grads = torch.autograd.grad(constraint, params, allow_unused=True)  # wrong: drops the local
    for p, g in zip(params, grads):
        if g is not None:
            p.grad.sub_(g, alpha=1.0 - float(beta))


def test_mutation_beta_factor_is_caught_by_the_known_answer():
    model, local, constraint, recon = _state()
    _mut_beta_factor(local, constraint, [model.cb], 0.01)
    assert not torch.allclose(model.cb.grad, _expected_cb(), atol=1e-7)


def test_mutation_sign_is_caught_by_the_known_answer():
    model, local, constraint, recon = _state()
    _mut_sign(local, constraint, [model.cb], 0.01)
    assert not torch.allclose(model.cb.grad, _expected_cb(), atol=1e-7)


def test_mutation_encoder_scope_is_caught_by_the_encoder_assertion():
    model, local, constraint, recon = _state()
    _mut_encoder(local, constraint, [model.cb, model.enc], 0.01)
    assert not torch.allclose(model.enc.grad, torch.tensor([5.0]), atol=1e-7)


def test_mutation_local_only_is_caught_by_the_known_answer():
    model, local, constraint, recon = _state()
    _mut_local_only(local, constraint, [model.cb], 0.01)
    assert not torch.allclose(model.cb.grad, _expected_cb(), atol=1e-7)


def test_mutation_constraint_only_is_caught_by_the_known_answer():
    model, local, constraint, recon = _state()
    _mut_constraint_only(local, constraint, [model.cb], 0.01)
    assert not torch.allclose(model.cb.grad, _expected_cb(), atol=1e-7)


def test_mutation_no_budget_is_caught_by_the_known_answer():
    # The round-1 behaviour (no budget) is the red baseline: it must fail the
    # same known-answer assertion that the helper passes.
    model, local, constraint, recon = _state()
    # no-op: leave the total gradient untouched
    assert not torch.allclose(model.cb.grad, _expected_cb(), atol=1e-7)
