"""Issue #217: the instance pools, the orbit labels and the joint head."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from graphlet_roles import (N_ROLES, ORBIT_TO_GLOBAL, ROLE_MASK, TYPES, InstanceSampler,
                            TypeRoleHead, load_instances, quantized_ste, relation_margin,
                            role_loss, soft_quantized, type_loss)


def test_pools_have_both_splits_and_valid_orbits():
    coverage = json.loads((TREE.parents[1] / "results" / "research" / "graphlet_roles" / "coverage.json").read_text())
    train, heldout = load_instances("train"), load_instances("heldout")
    for name in TYPES:
        assert coverage["types"][name]["train"] > 0, name
        assert coverage["types"][name]["heldout"] > 0, name
        orbits = np.unique(np.concatenate([train[name][1].ravel(), heldout[name][1].ravel()]))
        assert set(orbits.tolist()) <= set(range(len(ORBIT_TO_GLOBAL[name]))), name


def test_instances_never_cross_the_split():
    train, heldout = load_instances("train"), load_instances("heldout")
    for name in TYPES:
        train_sets = {tuple(row) for row in train[name][0].tolist()}
        held_sets = {tuple(row) for row in heldout[name][0].tolist()}
        assert not (train_sets & held_sets), name


def test_type_readout_is_permutation_invariant_and_role_readout_equivariant():
    torch.manual_seed(0)
    head = TypeRoleHead(dim=8, hidden=16)
    nodes = torch.randn(2, 4, 8)
    permutation = torch.tensor([2, 0, 3, 1])
    type_a, role_a = head(nodes)
    type_b, role_b = head(nodes[:, permutation])
    assert torch.allclose(type_a, type_b, atol=1e-6)
    assert torch.allclose(role_a[:, permutation], role_b, atol=1e-6)


def test_role_loss_only_allows_the_instance_orbits():
    logits = torch.zeros(1, 3, N_ROLES)
    mask = ROLE_MASK[TYPES.index("K3")].unsqueeze(0)
    legal = ORBIT_TO_GLOBAL["K3"][torch.zeros(1, 3, dtype=torch.long)]
    value = role_loss(logits, legal, mask)
    assert torch.isfinite(value)
    logits[:, :, :] = 0.0
    logits[:, :, 1] = 100.0  # P3 leaf: a role that a K3 instance cannot have
    masked = role_loss(logits, legal, mask)
    assert float(masked) <= 1e-6, "an impossible orbit must be masked out of the denominator"


def test_relation_margin_rewards_same_role_pairs_and_pushes_back():
    good = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, -0.1]]])
    roles = torch.tensor([[0, 1, 0, 1]])
    same_types = torch.ones(1, dtype=torch.bool)
    satisfied = relation_margin(good, roles, same_types)
    bad = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]])
    violated = relation_margin(bad, roles, same_types)
    assert float(satisfied) <= 1e-6
    assert float(violated) > 0.1
    value = relation_margin(bad.clone().requires_grad_(True), roles, same_types)
    value.backward()


def test_quantized_ste_value_is_the_hard_sum_and_the_gradient_is_the_identity():
    torch.manual_seed(4)
    encoder_out = torch.randn(6, 4, requires_grad=True)
    codebooks = [torch.randn(5, 4, requires_grad=True) for _ in range(3)]
    value = quantized_ste(encoder_out, codebooks)
    hard = 0.0
    residual = encoder_out.detach()
    for book in codebooks:
        picked = book.detach()[torch.cdist(residual, book.detach()).argmin(-1)]
        hard = hard + picked
        residual = residual - picked
    assert torch.allclose(value.detach(), hard, atol=1e-6)
    value.sum().backward()
    assert torch.allclose(encoder_out.grad, torch.ones_like(encoder_out))
    assert all(book.grad is None for book in codebooks)


def test_soft_quantized_reaches_the_encoder_and_leaves_the_codebook_alone():
    torch.manual_seed(1)
    encoder_out = torch.randn(6, 4, requires_grad=True)
    codebooks = [torch.randn(5, 4, requires_grad=True) for _ in range(3)]
    quantized = soft_quantized(encoder_out, codebooks, temperature=0.05)
    loss = quantized.pow(2).mean()
    loss.backward()
    assert encoder_out.grad is not None and torch.isfinite(encoder_out.grad).all()
    # Stop-gradient: the quantiser is owned by the reconstruction loss, so the auxiliary
    # term must not move it (the Q arm collapsed its codebook spacing with that gradient).
    assert all(book.grad is None for book in codebooks)
    hard = 0.0
    residual = encoder_out.detach()
    for book in codebooks:
        index = torch.cdist(residual, book.detach()).argmin(dim=-1)
        picked = book.detach()[index]
        hard = hard + picked
        residual = residual - picked
    assert torch.allclose(quantized.detach(), hard, atol=0.5)


def test_sampler_is_deterministic_for_a_seed():
    left = InstanceSampler("train", per_type=8)
    right = InstanceSampler("train", per_type=8)
    generator = torch.Generator().manual_seed(7)
    first = left.sample(generator)
    second = right.sample(torch.Generator().manual_seed(7))
    for name in TYPES:
        assert np.array_equal(first[name][0], second[name][0])
        assert np.array_equal(first[name][1], second[name][1])


def test_type_loss_is_finite_and_differentiable():
    logits = torch.randn(4, len(TYPES), requires_grad=True)
    value = type_loss(logits, torch.tensor([0, 1, 2, 3]))
    value.backward()
    assert torch.isfinite(value) and logits.grad is not None


def test_graphlet_aux_covers_every_arm_and_reaches_the_encoder():
    from graphlet_roles import (ARM_QUANT, ARM_RELATION, ARM_ROLE, ARM_TYPE, graphlet_aux)

    torch.manual_seed(3)
    dim = 6
    encoder = torch.nn.Linear(dim, dim)
    head = TypeRoleHead(dim=dim, hidden=16)
    codebooks = [torch.randn(7, dim, requires_grad=True) for _ in range(3)]
    sampler = InstanceSampler("train", per_type=4)
    largest = max(int(nodes.max()) for nodes, _ in load_instances("train").values())
    embeddings = torch.randn(largest + 1, dim)
    for arm in (ARM_TYPE, ARM_ROLE, ARM_QUANT, ARM_RELATION):
        encoder.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        value, diagnostics = graphlet_aux(encoder, head, sampler,
                                          torch.Generator().manual_seed(5), arm,
                                          embeddings, codebooks, torch.device("cpu"))
        assert torch.isfinite(value) and float(value) > 0, arm
        value.backward()
        assert encoder.weight.grad is not None and float(encoder.weight.grad.abs().sum()) > 0, arm
        if arm in (ARM_QUANT, ARM_RELATION):
            assert all(book.grad is None for book in codebooks), arm
        assert diagnostics["instances"] == 4 * len(TYPES)


def test_type_loss_is_computed_over_the_mixed_batch():
    import math
    from graphlet_roles import ARM_TYPE, graphlet_aux

    torch.manual_seed(5)
    dim = 6
    encoder = torch.nn.Linear(dim, dim)
    head = TypeRoleHead(dim=dim, hidden=16)
    sampler = InstanceSampler("train", per_type=8)
    largest = max(int(nodes.max()) for nodes, _ in load_instances("train").values())
    embeddings = torch.randn(largest + 1, dim)
    value, diagnostics = graphlet_aux(encoder, head, sampler, torch.Generator().manual_seed(9),
                                      ARM_TYPE, embeddings, [torch.randn(4, dim) for _ in range(3)],
                                      torch.device("cpu"))
    # Five balanced classes at chance cost ln 5 = 1.609; a single-class group would cost ~0.
    assert 1.2 < diagnostics["type"] < 2.2, diagnostics
