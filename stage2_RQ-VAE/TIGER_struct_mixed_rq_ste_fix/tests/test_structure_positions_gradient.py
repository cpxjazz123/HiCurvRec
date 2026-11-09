"""The structure term must carry exactly one codebook gradient path.

levels[-1] is already the native readback, which is differentiable in the codebook.
Adding codes_to_positions on top of it doubled that gradient silently.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import structure as S
import training_hooks as H
from model.model import RQVAE


def _model(seed=0):
    torch.manual_seed(seed)
    cfg = SimpleNamespace(codebook_num=3, codebook_dim=32, codebook_size=8,
                          vq_type="vq", beta=0.25, sk_epsilon=-1.0, sk_iters=-1,
                          hidden_sizes=(48, 32), dropout=0.0)
    return RQVAE(cfg, in_dim=32)


def _codebook(model):
    return [layer.get_code_embs() for layer in model.rq.vq_layers]


def _norm(grads):
    return sum(float(g.pow(2).sum()) for g in grads if g is not None) ** 0.5


def test_the_positions_carry_one_codebook_gradient():
    model = _model()
    # a freshly initialised codebook maps every item to one code; spread both the
    # codebook and the items so the fixture has more than one triple
    with torch.no_grad():
        for layer in model.rq.vq_layers:
            layer.get_code_embs().normal_(0, 0.5)
    embeddings = torch.empty(40, 32).normal_(0, 2.0)
    ids = list(range(12))
    positions = H._positions(model, embeddings, ids)[-1]
    _, _, _, tokens, levels = model.forward_with_codes(embeddings[ids])
    assert len({tuple(row.tolist()) for row in tokens}) > 1, (
        "the fixture collapsed onto one code; the gradient comparison is void")
    kinds = ["near", "far", "far", "near"]
    h = S.triple_targets(kinds)
    loss_positions = S.structure_loss(positions.reshape(4, 3, -1), h, scale=0.3)
    loss_levels = S.structure_loss(levels[-1].reshape(4, 3, -1), h, scale=0.3)
    assert float(loss_levels) > 0, "degenerate fixture: every triple sits at one point"
    params = _codebook(model)
    g_positions = torch.autograd.grad(loss_positions, params, retain_graph=True,
                                      allow_unused=True)
    g_levels = torch.autograd.grad(loss_levels, params, retain_graph=True,
                                   allow_unused=True)
    assert _norm(g_positions) > 0, "no codebook gradient at all"
    for a, b in zip(g_positions, g_levels):
        assert a is not None and b is not None
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7), (
            "the structure term adds a second codebook path: "
            f"{float(a.abs().sum())} vs {float(b.abs().sum())}")


if __name__ == "__main__":
    test_the_positions_carry_one_codebook_gradient()
    print("ok")
