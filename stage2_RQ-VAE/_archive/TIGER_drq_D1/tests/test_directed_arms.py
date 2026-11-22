"""Issue #221 CGV 2-3: label provenance and the per-arm gradient path.

Each arm is measured on the real quantizer: D1 must not touch the codebook, D2's
legacy STE must leave it at zero, D3 must reach it, D4's frozen head must stay
bit-identical, and the labels must equal a recount from the directed graph.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

import drq  # noqa: E402
from directed_graphlets import classify, load_adjacency  # noqa: E402
from directed_role import DirectedTypeRoleHead, instance_targets, load_pool, load_vocabulary  # noqa: E402
from model import RQVAE  # noqa: E402


def small_model(seed: int = 0) -> RQVAE:
    torch.manual_seed(seed)
    config = SimpleNamespace(codebook_num=3, codebook_dim=8, codebook_size=[32, 32, 32],
                             vq_type="vq", beta=0.25, sk_epsilon=0.003, sk_iters=5,
                             hidden_sizes=[16, 16], dropout=0.0,
                             ema_decay=0.99, fix_code_embs=False)
    return RQVAE(config, in_dim=24)


def sample_draw(family: str = "P3", count: int = 3):
    vocab = load_vocabulary()
    families = load_pool("heldout", vocab)
    nodes, orbits, subtypes = families[family]
    return vocab, (nodes[:count], orbits[:count], subtypes[:count]), family


def arm_loss(arm: str, model: RQVAE, head, family: str, count: int = 3):
    vocab, (nodes, orbits, subtypes), name = sample_draw(family, count)
    items = torch.rand(24587, 24)
    width = nodes.shape[1]
    flat = torch.from_numpy(nodes.reshape(-1)).long()
    if arm == "D1":
        encoded = model.encoder(items[flat])
    else:
        _, _, _, tokens = model(items[flat])
        books = [layer.get_code_embs() for layer in model.rq.vq_layers]
        if arm == "D2":
            encoded = model.encoder(items[flat])
            hard = drq.hard_cumulative(tokens, [book.detach() for book in books])
            encoded = drq.legacy_ste(encoded, hard)
        else:
            probs = [drq.level_softmax(model.rq.vq_layers[level],
                                       model.rq.last_residuals[level], 0.5)
                     for level in range(3)]
            encoded = drq.q_st(probs, tokens, books)
    features = {name: encoded.view(count, width, -1)}
    draw = {name: (torch.from_numpy(nodes), torch.from_numpy(orbits),
                   torch.from_numpy(subtypes))}
    return drq.joint_loss(head, features, draw, vocab, drq.targets_for(draw, vocab))[0]


def head_for(vocab) -> DirectedTypeRoleHead:
    torch.manual_seed(0)
    return DirectedTypeRoleHead(dim=8, hidden=16, n_subtypes=vocab["n_subtypes"],
                                n_orbits=vocab["n_orbits"])


def test_pool_labels_match_a_recount_from_the_directed_graph():
    vocab = load_vocabulary()
    families = load_pool("heldout", vocab)
    adjacency = load_adjacency()
    for family, (nodes, orbits, subtypes) in families.items():
        raw = vocab["raw"][family]
        keys = {index: key for key, index in raw.items() if key != "__other__"}
        for row in range(min(40, len(nodes))):
            instance = nodes[row]
            matrix = np.asarray(adjacency[instance][:, instance].todense(), dtype=bool)
            skeleton, key, orbit_ids = classify(matrix)
            assert skeleton == family
            assert keys[int(subtypes[row])] == key, (family, row)
            assert tuple(orbits[row].tolist()) == orbit_ids or int(subtypes[row]) == raw["__other__"]


def test_d1_reaches_the_encoder_only():
    vocab, _draw, family = sample_draw()
    model, head = small_model(), head_for(vocab)
    loss = arm_loss("D1", model, head, family)
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    books = [layer.get_code_embs() for layer in model.rq.vq_layers]
    assert drq.grad_norm(torch.autograd.grad(loss, encoder, retain_graph=True,
                                             allow_unused=True)) > 0
    assert drq.grad_norm(torch.autograd.grad(loss, books, retain_graph=True,
                                             allow_unused=True)) == 0.0


def test_d2_legacy_ste_leaves_the_codebook_alone():
    vocab, _draw, family = sample_draw()
    model, head = small_model(), head_for(vocab)
    loss = arm_loss("D2", model, head, family)
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    books = [layer.get_code_embs() for layer in model.rq.vq_layers]
    assert drq.grad_norm(torch.autograd.grad(loss, encoder, retain_graph=True,
                                             allow_unused=True)) > 0
    assert drq.grad_norm(torch.autograd.grad(loss, books, retain_graph=True,
                                             allow_unused=True)) == 0.0


def test_d3_reaches_the_encoder_and_the_codebook():
    vocab, _draw, family = sample_draw()
    model, head = small_model(), head_for(vocab)
    loss = arm_loss("D3", model, head, "P4")
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    books = [layer.get_code_embs() for layer in model.rq.vq_layers]
    encoder_norm = drq.grad_norm(torch.autograd.grad(loss, encoder, retain_graph=True,
                                                    allow_unused=True))
    book_norm = drq.grad_norm(torch.autograd.grad(loss, books, retain_graph=True,
                                                 allow_unused=True))
    assert encoder_norm > 0 and book_norm > 0
    assert np.isfinite(encoder_norm) and np.isfinite(book_norm)


def test_d4_frozen_head_does_not_move():
    vocab, _draw, family = sample_draw()
    model, head = small_model(), head_for(vocab)
    head.requires_grad_(False)
    loss = arm_loss("D3", model, head, "P4")
    before = [p.detach().clone() for p in head.parameters()]
    optimizer = torch.optim.AdamW([p for p in list(model.parameters()) + list(head.parameters())
                                   if p.requires_grad], lr=0.02)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(before, head.parameters()))


def test_one_step_reduces_the_joint_loss():
    vocab, _draw, family = sample_draw("P4", count=8)
    model, head = small_model(), head_for(vocab)
    optimizer = torch.optim.AdamW([p for p in list(model.parameters()) + list(head.parameters())
                                   if p.requires_grad], lr=0.02)
    history = []
    for _ in range(6):
        loss = arm_loss("D1", model, head, "P4", count=8)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    assert history[-1] < history[0], history


def test_orbit_mask_admits_only_the_instance_subtype_orbits():
    vocab = load_vocabulary()
    families = load_pool("heldout", vocab)
    nodes, orbits, subtypes = families["K3"]
    targets = instance_targets("K3", subtypes[:5], orbits[:5], vocab)
    for subtype_id in targets["subtype"]:
        assert int(subtype_id) >= 0
        allowed = set(vocab["orbits_of"][int(subtype_id)])
        assert allowed
        assert all(value in allowed for value in np.ravel(targets["orbit"]).tolist())
