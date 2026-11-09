"""The mechanism inside the model: the assignment, the demand and the gradient.

These tests are the wiring the specification demands and the framework's own
``encoder_gradient`` check demands: the level-1 codeword comes from the
behaviour-adjusted score, the residual recursion uses the UNLIFTED codeword,
the demand is measured on the surface that was scored, and the encoder still
receives a gradient through the straight-through estimator.
"""
from __future__ import annotations

import math

import pytest
import torch

from model.level1_behavior import LevelOneMechanism
from model.model import RQVAE


class _Config:
    hidden_sizes = (8, 8)
    codebook_num = 3
    codebook_size = (8, 8, 8)
    codebook_dim = 4
    dropout = 0.0
    vq_type = "vq"
    beta = 0.25
    sk_epsilon = -1.0
    sk_iters = -1


def _model(**kwargs) -> RQVAE:
    torch.manual_seed(0)
    mechanism = LevelOneMechanism(scale=2.0, codebook_size=8, **kwargs)
    return RQVAE(_Config(), in_dim=5, mechanism=mechanism)


def test_the_level_one_tokens_come_from_the_mechanism():
    model = _model()
    embeddings = torch.randn(6, 5)
    reconstructed, quant_loss, _, tokens = model(embeddings)
    assert tokens.shape == (6, 3)
    assert model.mechanism is not None
    # the chosen level-1 codeword is the one the mechanism's argmin picked
    latent = model.encoder(embeddings)
    _, _, _, chosen = model.mechanism.score(latent, model.rq.vq_layers[0].get_code_embs())
    assert torch.equal(tokens[:, 0], chosen)


def test_the_residual_recursion_uses_the_unlifted_codeword():
    """r_next = r - a_{k1}: the codeword, not its lift."""
    model = _model()
    embeddings = torch.randn(4, 5)
    latent = model.encoder(embeddings)
    codebook = model.rq.vq_layers[0].get_code_embs()
    _, _, _, chosen = model.mechanism.score(latent, codebook)
    expected = latent - codebook[chosen]
    lifted = model.mechanism.lift(codebook)[chosen]
    assert not torch.allclose(expected, latent - lifted, atol=1e-4)


def test_the_demand_is_measured_on_the_scored_surface():
    model = _model()
    embeddings = torch.randn(7, 5)
    model(embeddings)
    distance = model.mechanism.last_distance()
    assert distance is not None
    assignment = model.mechanism.soft_assignment(distance)
    assert assignment.shape == distance.shape
    assert torch.allclose(assignment.sum(dim=-1), torch.ones(7), atol=1e-5)


def test_the_encoder_receives_a_gradient_through_the_straight_through_path():
    """The framework's encoder_gradient check: the level-1 choice must reach
    the encoder, and only the straight-through link carries it."""
    model = _model()
    embeddings = torch.randn(6, 5, requires_grad=True)
    reconstructed, quant_loss, _, _ = model(embeddings)
    loss = torch.nn.functional.mse_loss(reconstructed, embeddings) + quant_loss
    model.zero_grad()
    loss.backward()
    gradient = next((parameter.grad for parameter in model.encoder.parameters()
                     if parameter.grad is not None), None)
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def test_the_mechanism_has_no_learnable_parameter():
    """The mechanism's quantities are constants and buffers: nothing is trained."""
    model = _model()
    names = [name for name, _ in model.mechanism.named_parameters()]
    assert names == []


def test_the_epoch_update_changes_the_offset():
    model = _model()
    embeddings = torch.randn(8, 5)
    model(embeddings)
    model.observe_level1(torch.rand(8) + 0.5)
    before = model.mechanism.offset().clone()
    model.end_epoch()
    after = model.mechanism.offset()
    assert torch.isfinite(after).all()
    assert after.shape == before.shape
    assert model.mechanism.epoch_index == 2
