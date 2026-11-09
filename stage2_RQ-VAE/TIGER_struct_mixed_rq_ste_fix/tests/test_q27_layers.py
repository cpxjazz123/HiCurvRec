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
    positions = H._positions(model, items, ids)[-1].reshape(len(triples), 3, -1)
    h = S.triple_targets([kind for *_, kind in triples], device=positions.device)
    last = S.structure_loss(positions, h, scale=1.0)
    assert not torch.allclose(term, last, rtol=0.0, atol=1e-6)


def test_the_second_level_receives_a_gradient():
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
    params = list(model.encoder.parameters()) + [l.get_code_embs() for l in model.rq.vq_layers]
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    assert any(g is not None and float(g.abs().max()) > 0 for g in grads)
    assert term.requires_grad


if __name__ == "__main__":
    test_three_levels_differ_from_the_last_level_only()
    test_the_second_level_receives_a_gradient()
    print("ok")
