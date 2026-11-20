"""Issue #224: the relational loss is a real, correct, wired objective.

Each test pins one property that would silently turn the experiment into a
non-test: a distribution that does not sum to one, a KL with the arguments
reversed (identical numbers for the wrong direction on small batches), a target
that still carries gradient into the Teacher, a loss that never reaches the
codebooks, or a permutation dependence that would make the metric a function of
node order rather than of structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import relational_distill as rd  # noqa: E402
import trainable_student as ts  # noqa: E402


def _representations(seed: int = 3, instances: int = 6, nodes: int = 4, dim: int = 8):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(instances, nodes, dim, generator=generator)


def test_relation_rows_are_distributions_over_the_other_nodes():
    representations = _representations()
    distribution = rd.relation_distribution(representations, temperature=0.2)
    assert torch.allclose(distribution.sum(-1), torch.ones(distribution.shape[:-1]), atol=1e-6)
    diagonal = torch.diagonal(distribution, dim1=-2, dim2=-1)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-12), \
        "a node must not be its own relation target"


def test_the_loss_is_zero_when_the_student_relation_equals_the_target():
    representations = _representations()
    target = rd.relation_distribution(representations, temperature=0.2)
    result = rd.relational_loss(representations, target, temperature=0.2)
    assert result["kl"] < 1e-6
    assert result["top1_agreement"] == 1.0


def test_the_loss_falls_when_the_student_moves_toward_the_target():
    target_representation = _representations(seed=5)
    target = rd.relation_distribution(target_representation, temperature=0.2)
    student = (target_representation + 0.5 * torch.randn(
        target_representation.shape, generator=torch.Generator().manual_seed(9))).requires_grad_(True)
    before = rd.relational_loss(student, target, temperature=0.2)["kl"]
    optimizer = torch.optim.Adam([student], lr=5e-2)
    for _ in range(200):
        optimizer.zero_grad()
        loss = rd.relational_loss(student, target, temperature=0.2)["loss"]
        loss.backward()
        optimizer.step()
    after = rd.relational_loss(student.detach(), target, temperature=0.2)["kl"]
    assert after < before * 0.5, f"the objective did not optimise: {before:.4f} -> {after:.4f}"


def test_kl_argument_order_is_pinned():
    """KL(P_T||P_Q) is not KL(P_Q||P_T); the reverse is a different number here."""
    target_representation = _representations(seed=11)
    target = rd.relation_distribution(target_representation, temperature=0.05)
    student_representation = target_representation + 0.7 * torch.randn(
        target_representation.shape, generator=torch.Generator().manual_seed(12))
    forward = rd.relational_loss(student_representation, target, temperature=0.05)["kl"]
    student = rd.relation_distribution(student_representation, temperature=0.05)
    reverse = float((student * (torch.log(student.clamp_min(1e-12))
                                - torch.log(target.clamp_min(1e-12)))).sum(-1).mean())
    assert abs(forward - reverse) > 1e-3, "the loss looks symmetric; the arguments may be swapped"


def test_a_common_permutation_leaves_the_loss_unchanged():
    target_representation = _representations(seed=13)
    target = rd.relation_distribution(target_representation, temperature=0.2)
    student_representation = target_representation + 0.3
    permutation = torch.tensor([2, 0, 3, 1])
    plain = rd.relational_loss(student_representation, target, temperature=0.2)["kl"]
    permuted = rd.relational_loss(student_representation[:, permutation],
                                  target[:, permutation][:, :, permutation], temperature=0.2)["kl"]
    assert abs(plain - permuted) < 1e-6, "the metric depends on node order"


def test_the_target_carries_no_gradient_and_the_teacher_does_not_move():
    model = ts.build_small_model(in_dim=24)
    teacher = ts.build_small_teacher(dim=24, hidden=16, levels=2, n_subtypes=5, n_orbits=7)
    before = ts.parameter_digest(teacher)
    generator = torch.Generator().manual_seed(17)
    items = torch.randn(20, 24, generator=generator)
    adjacency = ts.random_adjacency(5, 4, seed=18)
    target = rd.teacher_relation(teacher, items.view(5, 4, 24), adjacency, temperature=0.2)
    assert not target.requires_grad
    path = ts.quantised_student_input(model, items, instances=5, nodes=4)
    result = rd.relational_loss(path.q, target, temperature=0.2)
    result["loss"].backward()
    encoder_grad = sum(float(parameter.grad.abs().sum()) for parameter in model.encoder.parameters()
                       if parameter.grad is not None)
    assert encoder_grad > 0, "the relational loss gives the encoder no gradient"
    after = ts.parameter_digest(teacher)
    assert all(before[name] == after[name] for name in before)
    for parameter in teacher.parameters():
        assert parameter.grad is None


def test_the_relational_loss_reaches_every_codebook_through_the_real_quantiser():
    model = ts.build_small_model(in_dim=24, narrow=16, levels=3, codes=16, dim=8)
    generator = torch.Generator().manual_seed(19)
    items = torch.randn(48, 24, generator=generator)
    target_representation = torch.randn(12, 4, 8, generator=generator)
    target = rd.relation_distribution(target_representation, temperature=0.1)
    path = ts.quantised_student_input(model, items, instances=12, nodes=4)
    result = rd.relational_loss(path.q, target, temperature=0.1)
    norms = {"encoder": 0.0, "codebooks": []}
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    norms["encoder"] = float(torch.sqrt(sum((g.detach() ** 2).sum() for g in
                                            torch.autograd.grad(result["loss"], encoder,
                                                                retain_graph=True,
                                                                allow_unused=True) if g is not None)))
    for level, parameter in enumerate(ts.codebook_parameters(model).values()):
        grad = torch.autograd.grad(result["loss"], [parameter], retain_graph=True,
                                   allow_unused=True)[0]
        norms["codebooks"].append(0.0 if grad is None else float(grad.detach().norm()))
    assert norms["encoder"] > 0
    assert all(norm > 0 for norm in norms["codebooks"]), \
        f"a codebook level received nothing: {norms['codebooks']}"


def test_the_adjacency_control_is_uniform_over_neighbours():
    adjacency = torch.zeros(2, 3, 3)
    adjacency[0, 0, 1] = 1.0
    adjacency[0, 1, 0] = 1.0
    distribution = rd.adjacency_distribution(adjacency)
    assert torch.allclose(distribution.sum(-1), torch.ones(2, 3), atol=1e-6)
    assert pytest.approx(float(distribution[0, 0, 1]), abs=1e-6) == 1.0
    assert pytest.approx(float(distribution[1, 0, 1]), abs=1e-6) == 0.5, \
        "an isolated node must fall back to uniform"


def test_a_single_node_relation_is_refused():
    with pytest.raises(ValueError):
        rd.relation_distribution(torch.randn(3, 1, 8))
