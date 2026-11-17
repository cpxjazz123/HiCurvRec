"""The codebook-level geometric bias: one alpha per codeword.

The mechanism this tests: a codeword carries its own Euclidean/hyperbolic
weight, so the *partition* is where the graphlet bias acts, not the item.  Every
test here fails before the mechanism exists:

* a constant per-codeword table must reproduce the per-item cost exactly;
* two codewords that are close in different halves must swap winners when their
  weights swap;
* the encoder and the codebook must receive gradients through the table;
* the export must use the same table the training used;
* the density -> alpha schedule must be monotone, bounded and deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.mixed_rq import (EUCLIDEAN_DIM, TOTAL_DIM, AlphaTable, MixedRQ,
                            encode_tokens, mixed_cost)
from model.model import MixedRQVAE
from model.poincare_contract import expmap0


def config():
    return SimpleNamespace(
        hidden_sizes=(16,), codebook_num=2, codebook_size=5, codebook_dim=TOTAL_DIM,
        dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
        sk_epsilon=-1.0, sk_iters=-1,
    )


def probe(batch: int = 4, codes: int = 5, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    residual_e = torch.randn(batch, EUCLIDEAN_DIM, generator=generator, dtype=torch.float64)
    tangent = torch.randn(batch, EUCLIDEAN_DIM, generator=generator, dtype=torch.float64) * 0.3
    codes_e = torch.randn(codes, EUCLIDEAN_DIM, generator=generator, dtype=torch.float64)
    codes_h = torch.randn(codes, EUCLIDEAN_DIM, generator=generator, dtype=torch.float64) * 0.3
    return residual_e, expmap0(tangent, 1.0), codes_e, codes_h


def test_a_constant_code_table_reproduces_the_per_item_cost() -> None:
    residual_e, residual_h, codes_e, codes_h = probe()
    alpha = torch.full((4,), 0.5, dtype=torch.float64)
    table = torch.full((codes_e.shape[0],), 0.5, dtype=torch.float64)
    per_item = mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha)
    per_code = mixed_cost(residual_e, residual_h, codes_e, codes_h, None, code_alpha=table)
    assert torch.allclose(per_item, per_code, atol=0.0, rtol=0.0)


def test_swapping_two_code_weights_swaps_the_winner() -> None:
    batch = 1
    residual_e = torch.zeros(batch, EUCLIDEAN_DIM, dtype=torch.float64)
    residual_h = torch.zeros(batch, EUCLIDEAN_DIM, dtype=torch.float64)
    codes_e = torch.stack([torch.full((EUCLIDEAN_DIM,), 10.0, dtype=torch.float64),
                           torch.zeros(EUCLIDEAN_DIM, dtype=torch.float64)])
    codes_h = torch.stack([torch.zeros(EUCLIDEAN_DIM, dtype=torch.float64),
                           torch.full((EUCLIDEAN_DIM,), 2.0, dtype=torch.float64)])
    # code 0 is exact in the hyperbolic half and far in Euclidean; code 1 the reverse.
    hyperbolic_wins = mixed_cost(residual_e, residual_h, codes_e, codes_h, None,
                                 code_alpha=torch.tensor([0.0, 1.0], dtype=torch.float64))
    euclidean_wins = mixed_cost(residual_e, residual_h, codes_e, codes_h, None,
                                code_alpha=torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert int(torch.argmin(hyperbolic_wins, dim=-1)) == 0
    assert int(torch.argmin(euclidean_wins, dim=-1)) == 1


def test_gradients_reach_the_encoder_and_the_codebook_through_the_table() -> None:
    torch.manual_seed(0)
    model = MixedRQVAE(config(), in_dim=32, curvature=1.0, code_alpha_mode=True)
    model.train()
    embeddings = torch.randn(7, 32)
    tokens = model(embeddings)[3]
    reconstructed, quant_loss, _, _ = model(embeddings)
    loss = reconstructed.pow(2).mean() + quant_loss
    loss.backward()
    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    table_grads = [p.grad for level in model.rq.levels
                   for p in level.embed.parameters() if p.grad is not None]
    assert encoder_grads and any(bool(g.abs().sum() > 0) for g in encoder_grads)
    assert table_grads and any(bool(g.abs().sum() > 0) for g in table_grads)
    assert tokens.shape == (7, 2)


def test_the_export_uses_the_same_table_as_the_forward_pass() -> None:
    torch.manual_seed(1)
    model = MixedRQVAE(config(), in_dim=32, curvature=1.0, code_alpha_mode=True)
    model.eval()
    with torch.no_grad():
        model.rq.code_alpha.copy_(torch.tensor(
            [[0.3, 0.7, 0.3, 0.7, 0.5], [0.7, 0.3, 0.7, 0.3, 0.5]]))
    embeddings = torch.randn(9, 32)
    with torch.no_grad():
        tokens_forward = model(embeddings)[3]
        tokens_export = encode_tokens(model.encoder(embeddings), model.rq,
                                      code_alpha=model.rq.code_alpha)
    assert torch.equal(tokens_forward, tokens_export)


def test_the_density_schedule_is_monotone_bounded_and_deterministic() -> None:
    from model.graphlet_allocation import ALPHA_MAX, ALPHA_MIN, POPULATION_ALPHA, allocate

    density = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, True, True, True, True])
    alpha = allocate(density, mask)
    assert alpha.shape == density.shape
    assert np.all(np.diff(alpha) > 0)
    assert alpha.min() >= ALPHA_MIN and alpha.max() <= ALPHA_MAX
    assert np.allclose(alpha, np.linspace(ALPHA_MIN, ALPHA_MAX, density.size))

    unused = np.array([0.0, 1.0, 99.0])
    mask = np.array([True, True, False])
    alpha = allocate(unused, mask)
    assert alpha[2] == POPULATION_ALPHA
    assert np.array_equal(alpha, allocate(unused, mask))


def test_a_constant_table_reproduces_the_f0_training_path() -> None:
    """The control arm must be bit-identical to F0: same cost, same tokens."""
    torch.manual_seed(2)
    f0 = MixedRQVAE(config(), in_dim=32, curvature=1.0, code_alpha_mode=False)
    control = MixedRQVAE(config(), in_dim=32, curvature=1.0, code_alpha_mode=True)
    control.load_state_dict(f0.state_dict(), strict=False)
    control.rq.code_alpha.fill_(0.5)
    embeddings = torch.randn(11, 32)
    with torch.no_grad():
        f0_out = f0(embeddings)
        control_out = control(embeddings)
    assert torch.allclose(f0_out[0], control_out[0], atol=1e-6)
    assert torch.allclose(f0_out[1], control_out[1], atol=1e-6)
    assert torch.equal(f0_out[3], control_out[3])
