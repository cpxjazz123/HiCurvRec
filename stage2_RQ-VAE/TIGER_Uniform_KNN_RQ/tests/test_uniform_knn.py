"""Contract tests for the Uniform-KNN Stage2 arm.

These tests are written before the implementation: they pin the KNN target shape, the KL
normalisation, the tie-free finite behaviour, and the two gradient paths the mechanism needs
(encoder and codebook).  They use a real RQVAE instance, not only synthetic tensors.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from losses.knn import (
    KNN_NEGATIVES,
    KNN_NEIGHBOURS,
    QuantizedMemory,
    build_knn_targets,
    hard_quantized_from_embeddings,
    knn_neighbourhood_kl,
    soft_quantized_readout,
)
from model import RQVAE


def _model(seed: int = 7) -> RQVAE:
    torch.manual_seed(seed)
    config = SimpleNamespace(
        hidden_sizes=(16, 8), codebook_num=3, codebook_size=(8, 8, 8),
        codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=5,
    )
    model = RQVAE(config, in_dim=8)
    with torch.no_grad():
        for layer in model.rq.vq_layers:
            layer.embed.weight.normal_(0.0, 0.1)
    return model


def test_targets_have_fixed_support_and_normalised_probabilities():
    rng = np.random.default_rng(13)
    embeddings = rng.normal(size=(64, 12)).astype(np.float32)
    candidates, log_target, tau = build_knn_targets(
        embeddings, k=KNN_NEIGHBOURS, negatives=KNN_NEGATIVES, seed=5
    )

    assert candidates.shape == (64, KNN_NEIGHBOURS + KNN_NEGATIVES)
    assert log_target.shape == candidates.shape
    assert tau.shape == (64,)
    assert np.isfinite(log_target).all()
    assert np.allclose(np.exp(log_target).sum(axis=1), 1.0, atol=1e-5)

    positives = candidates[:, :KNN_NEIGHBOURS]
    negatives = candidates[:, KNN_NEIGHBOURS:]
    for item in range(64):
        assert item not in positives[item]
        assert item not in negatives[item]
        assert not (set(positives[item].tolist()) & set(negatives[item].tolist()))


def test_kl_is_zero_for_identical_distribution_and_finite_for_duplicate_q():
    target = np.log(np.full((4, 6), 1.0 / 6.0, dtype=np.float32))
    q_anchor = torch.zeros(4, 4, requires_grad=True)
    q_candidates = torch.zeros(4, 6, 4)
    loss = knn_neighbourhood_kl(q_anchor, q_candidates, target)
    assert torch.isfinite(loss)
    assert abs(float(loss)) < 1e-5
    loss.backward()
    assert q_anchor.grad is not None


def test_kl_backpropagates_to_the_anchor():
    target = np.log(np.array([[0.7, 0.2, 0.1, 0.0]] * 3, dtype=np.float32))
    q_anchor = torch.randn(3, 4, requires_grad=True)
    q_candidates = torch.randn(3, 4, 4)
    loss = knn_neighbourhood_kl(q_anchor, q_candidates, target)
    loss.backward()
    assert q_anchor.grad is not None
    assert torch.isfinite(q_anchor.grad).all()
    assert q_anchor.grad.abs().sum() > 0


def test_soft_readout_reaches_encoder_and_codebook():
    model = _model()
    x = torch.randn(6, 8)
    with torch.no_grad():
        encoded = model.encoder(x)
        _, _, _, tokens = model.rq(encoded)
    q_proxy, q_hard = soft_quantized_readout(model, x, tokens)
    assert q_proxy.shape == (6, 4)
    assert q_hard.shape == (6, 4)
    assert torch.isfinite(q_proxy).all()

    q_proxy.sum().backward()
    encoder_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.encoder.named_parameters()
        if parameter.grad is not None and name.endswith("weight")
    )
    codebook_grad = sum(
        float(layer.embed.weight.grad.abs().sum())
        for layer in model.rq.vq_layers
        if layer.embed.weight.grad is not None
    )
    assert encoder_grad > 0, "the encoder must receive a gradient through the soft readout"
    assert codebook_grad > 0, "the codebook must receive a gradient through the soft readout"
    assert torch.allclose(q_hard, q_hard.detach())


def test_soft_readout_uses_the_supplied_hard_tokens():
    model = _model(seed=11)
    x = torch.randn(4, 8)
    with torch.no_grad():
        encoded = model.encoder(x)
        _, _, _, tokens = model.rq(encoded)
        expected = sum(
            torch.nn.functional.embedding(tokens[:, level], layer.get_code_embs())
            for level, layer in enumerate(model.rq.vq_layers)
        )
    _, q_hard = soft_quantized_readout(model, x, tokens, assignment="model")
    assert torch.allclose(q_hard, expected, atol=1e-6)


def test_export_readout_matches_argmin_path():
    model = _model(seed=23)
    x = torch.randn(5, 8)
    expected_q, expected_tokens = hard_quantized_from_embeddings(model, x)
    q_proxy, q_hard = soft_quantized_readout(model, x, assignment="export")
    assert torch.allclose(q_hard, expected_q, atol=1e-6)
    # The export path must not depend on the model's training-time Sinkhorn tokens.
    with torch.no_grad():
        encoded = model.encoder(x)
        _, _, _, train_tokens = model.rq(encoded)
    assert not torch.equal(expected_tokens, train_tokens), "the fixture must exercise the mismatch"


def test_memory_refresh_and_post_step_update_are_export_consistent():
    model = _model(seed=31)
    x = torch.randn(12, 8)
    memory = QuantizedMemory.from_model(model, x, step=0)
    assert memory.stats()["max_age"] == 0
    with torch.no_grad():
        for layer in model.rq.vq_layers:
            layer.embed.weight.add_(0.05)
    memory.refresh(model, x, step=4)
    expected, _ = hard_quantized_from_embeddings(model, x)
    assert torch.allclose(memory.vectors, expected, atol=1e-6)
    assert memory.stats()["step"] == 4
    # Updating a subset after a parameter change keeps the rest stale but reports it.
    ids = torch.tensor([1, 4, 7])
    memory.update_batch(model, x[ids], ids, step=5)
    stats = memory.stats()
    assert stats["max_age"] >= 0
    assert stats["stale_items"] == len(x) - len(ids)


def test_hard_readout_matches_selected_codebooks():
    model = _model(seed=19)
    x = torch.randn(3, 8)
    with torch.no_grad():
        encoded = model.encoder(x)
        _, _, _, tokens = model.rq(encoded)
        expected = sum(
            torch.nn.functional.embedding(tokens[:, level], layer.get_code_embs())
            for level, layer in enumerate(model.rq.vq_layers)
        )
    from losses.knn import hard_quantized_from_tokens
    actual = hard_quantized_from_tokens(model, tokens)
    assert torch.allclose(actual, expected, atol=1e-6)
