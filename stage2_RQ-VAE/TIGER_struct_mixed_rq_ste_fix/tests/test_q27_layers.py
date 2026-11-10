"""The structure term must average the three cumulative levels, not the last."""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structure as S
import train_rqvae as T
import training_hooks as H
import test_real_batch_evidence as E


def _term(model, items, triples):
    return H.structure_term(model, items, S.build_graph(T.TRAIN_FILE, seed=T.SEED),
                            random.Random(T.SEED), scale=1.0, size=len(triples),
                            triples=triples)


def test_three_levels_differ_from_the_last_level_only():
    model, items, _ = E._setup()
    graph = S.build_graph(T.TRAIN_FILE, seed=T.SEED)
    triples = S.sample_triples(graph, 64, random.Random(T.SEED))
    term = _term(model, items, triples)

    ids = [item for centre, a, b, _ in triples for item in (centre, a, b)]
    stack = H._positions(model, items, ids)
    positions = (stack if stack.dim() == 2 else stack[-1]).reshape(len(triples), 3, -1)
    h = S.triple_targets([kind for *_, kind in triples], device=positions.device)
    last = S.structure_loss(positions, h, scale=1.0)
    assert not torch.allclose(term, last, rtol=0.0, atol=1e-6)


def _codeword_params(model):
    return [layer.get_code_embs() for layer in model.rq.vq_layers]


def test_the_prefix_gradient_differs_from_the_last_layer_gradient():
    model, items, _ = E._setup()
    graph = S.build_graph(T.TRAIN_FILE, seed=T.SEED)
    triples = S.sample_triples(graph, 32, random.Random(T.SEED))
    captured = []
    original = model.forward_with_codes

    def wrapped(embeddings):
        out = original(embeddings)
        captured.extend(out[4])
        return out

    model.forward_with_codes = wrapped
    term = _term(model, items, triples)
    assert len(captured) == 3
    ids = [item for centre, a, b, _ in triples for item in (centre, a, b)]
    stack = H._positions(model, items, ids)
    positions = (stack if stack.dim() == 2 else stack[-1]).reshape(len(triples), 3, -1)
    h = S.triple_targets([kind for *_, kind in triples], device=positions.device)
    last = S.structure_loss(positions, h, scale=1.0)
    params = _codeword_params(model)
    g_term = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    g_last = torch.autograd.grad(last, params, retain_graph=True, allow_unused=True)
    differs = any(
        (a is None) != (b is None) or (a is not None and b is not None
                                        and not torch.allclose(a, b, rtol=0.0, atol=0.0))
        for a, b in zip(g_term, g_last))
    assert differs


if __name__ == "__main__":
    test_three_levels_differ_from_the_last_level_only()
    test_the_prefix_gradient_differs_from_the_last_layer_gradient()
    print("ok")
