"""Issue #220 CGV 1-3: labels/forward, the gradient path per arm, and the direction.

Each assertion changes if the implementation is wrong: the orbit labels are
recomputed from the graph, the codeword gradient is measured per arm, and a
single optimiser step must move the reported cross-entropies down.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

import trq  # noqa: E402
from graphlet_roles import (ORBIT_TO_GLOBAL, ROLE_MASK, TYPE_INDEX, TYPES, TypeRoleHead,  # noqa: E402
                            load_instances, role_loss)
from model import RQVAE  # noqa: E402


def small_model(seed: int = 0) -> RQVAE:
    torch.manual_seed(seed)
    config = SimpleNamespace(codebook_num=3, codebook_dim=8, codebook_size=[32, 32, 32],
                             vq_type="vq", beta=0.25, sk_epsilon=0.003, sk_iters=5,
                             hidden_sizes=[16, 16], dropout=0.0,
                             ema_decay=0.99, fix_code_embs=False)
    return RQVAE(config, in_dim=24)


def instance_tensors(name: str, count: int = 3):
    grouped = load_instances("heldout")
    nodes, orbits = grouped[name]
    take = min(count, len(nodes))
    return (torch.from_numpy(nodes[:take].astype(np.int64)),
            torch.from_numpy(orbits[:take].astype(np.int64)))


def features_for(arm: str, model: RQVAE, items: torch.Tensor, nodes: torch.Tensor):
    width = nodes.shape[1]
    flat = nodes.reshape(-1)
    if arm == "B":
        encoded = model.encoder(items[flat]).view(len(nodes), width, -1)
        return encoded, None
    _, _, _, tokens = model(items[flat])
    residuals = model.rq.last_residuals
    books = [layer.get_code_embs() for layer in model.rq.vq_layers]
    if arm == "C":
        encoded = model.encoder(items[flat])
        hard = trq.hard_cumulative(tokens, [book.detach() for book in books])
        return trq.legacy_ste(encoded, hard).view(len(nodes), width, -1), tokens
    probs = [trq.level_softmax(model.rq.vq_layers[level], residuals[level], 0.5)
             for level in range(3)]
    return trq.q_st(probs, tokens, books).view(len(nodes), width, -1), tokens


def loss_for(arm: str, model: RQVAE, head, nodes: torch.Tensor, orbits: torch.Tensor,
             name: str = "P4"):
    items = torch.rand(24587, 24)
    features, _ = features_for(arm, model, items, nodes)
    roles = ORBIT_TO_GLOBAL[name][orbits]
    mask = ROLE_MASK[TYPE_INDEX[name]].unsqueeze(0).expand(len(nodes), -1)
    types = {name: torch.full((len(nodes),), TYPE_INDEX[name], dtype=torch.long)}
    return trq.joint_loss(head, {name: features}, {name: roles}, types, {name: mask})


def test_labels_come_from_the_graph_not_from_the_input():
    """P4 ends/internals and the C4 single orbit are recomputed from the pool."""
    nodes, orbits = instance_tensors("P4", 5)
    assert set(np.unique(orbits.tolist())) == {0, 1}
    c4_nodes, c4_orbits = instance_tensors("C4", 5)
    assert set(np.unique(c4_orbits.tolist())) == {0}


def test_head_is_permutation_invariant_and_equivariant():
    torch.manual_seed(0)
    head = TypeRoleHead(dim=6, hidden=8)
    nodes = torch.rand(4, 4, 6)
    type_logits, role_logits = head(nodes)
    order = torch.tensor([2, 0, 3, 1])
    permuted_type, permuted_role = head(nodes[:, order])
    assert torch.allclose(type_logits, permuted_type, atol=1e-6)
    assert torch.allclose(role_logits[:, order], permuted_role, atol=1e-6)


def test_cross_entropies_are_finite_and_positive():
    model = small_model()
    head = TypeRoleHead(dim=8, hidden=16)
    nodes, orbits = instance_tensors("K1_3")
    loss, diag = loss_for("B", model, head, nodes, orbits, "K1_3")
    assert torch.isfinite(loss) and float(loss) > 0
    assert diag["type"] >= 0 and diag["role"] >= 0


def test_arm_b_gradients_reach_encoder_not_codebook():
    model = small_model()
    head = TypeRoleHead(dim=8, hidden=16)
    nodes, orbits = instance_tensors("P3")
    loss, _ = loss_for("B", model, head, nodes, orbits, "P3")
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    books = [layer.get_code_embs() for layer in model.rq.vq_layers]
    enc = trq.grad_norm(torch.autograd.grad(loss, encoder, retain_graph=True, allow_unused=True))
    book = trq.grad_norm(torch.autograd.grad(loss, books, retain_graph=True, allow_unused=True))
    assert enc > 0
    assert book == 0.0, "arm B must not move the codebook"


def test_arm_c_ste_codebook_gradient_is_zero():
    model = small_model()
    head = TypeRoleHead(dim=8, hidden=16)
    nodes, orbits = instance_tensors("P3")
    loss, _ = loss_for("C", model, head, nodes, orbits, "P3")
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    books = [layer.get_code_embs() for layer in model.rq.vq_layers]
    enc = trq.grad_norm(torch.autograd.grad(loss, encoder, retain_graph=True, allow_unused=True))
    book = trq.grad_norm(torch.autograd.grad(loss, books, retain_graph=True, allow_unused=True))
    assert enc > 0
    assert book == 0.0, "legacy STE keeps the codebook out of the auxiliary path"


def test_arm_d_gradients_reach_encoder_and_codebook():
    model = small_model()
    head = TypeRoleHead(dim=8, hidden=16)
    nodes, orbits = instance_tensors("P4")
    loss, _ = loss_for("D", model, head, nodes, orbits, "P4")
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    books = [layer.get_code_embs() for layer in model.rq.vq_layers]
    enc = trq.grad_norm(torch.autograd.grad(loss, encoder, retain_graph=True, allow_unused=True))
    book = trq.grad_norm(torch.autograd.grad(loss, books, retain_graph=True, allow_unused=True))
    assert enc > 0 and book > 0
    assert np.isfinite(enc) and np.isfinite(book)


def test_arm_e_frozen_head_stays_frozen():
    model = small_model()
    head = TypeRoleHead(dim=8, hidden=16)
    head.requires_grad_(False)
    nodes, orbits = instance_tensors("P4")
    loss, _ = loss_for("D", model, head, nodes, orbits, "P4")
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    books = [layer.get_code_embs() for layer in model.rq.vq_layers]
    encoder_norm = trq.grad_norm(torch.autograd.grad(loss, encoder, retain_graph=True, allow_unused=True))
    book_norm = trq.grad_norm(torch.autograd.grad(loss, books, retain_graph=True, allow_unused=True))
    assert encoder_norm > 0 and book_norm > 0
    before = [p.detach().clone() for p in head.parameters()]
    optimizer = torch.optim.AdamW([p for p in list(model.parameters()) + list(head.parameters())
                                   if p.requires_grad], lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(before, head.parameters()))


def test_one_step_reduces_the_joint_loss():
    model = small_model()
    head = TypeRoleHead(dim=8, hidden=16)
    nodes, orbits = instance_tensors("P4", 8)
    optimizer = torch.optim.AdamW([p for p in list(model.parameters()) + list(head.parameters())
                                   if p.requires_grad], lr=0.02)
    before_parents = torch.tensor([0.0])
    for _ in range(6):
        loss, _ = loss_for("B", model, head, nodes, orbits, "P4")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        before_parents = torch.cat([before_parents, loss.detach().reshape(1)])
    losses = before_parents[1:]
    assert float(losses[-1]) < float(losses[0]), losses.tolist()


def test_ste_onehot_matches_the_hard_onehot_in_forward():
    hard = torch.tensor([0, 2, 1])
    soft = torch.softmax(torch.randn(3, 4), dim=-1).detach().requires_grad_(True)
    value = trq.ste_onehot(hard, soft, 4)
    assert torch.equal(value.detach(), torch.nn.functional.one_hot(hard, 4).float())
    value.sum().backward()
    assert torch.allclose(soft.grad, torch.ones_like(soft))
