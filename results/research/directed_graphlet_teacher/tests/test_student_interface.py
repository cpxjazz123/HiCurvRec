"""Student-side contracts for issue #222 B/C.

Three properties, each of which would silently fake a transfer result if broken:

  1. the Student never reads an adjacency: the function's signature has none and
     a bundle that carries one is refused, not ignored;
  2. the KD loss points the right way: `kl_div(log_softmax(student),
     softmax(teacher))`, so it falls when the student's logits move toward the
     teacher and a uniform teacher leaves a uniform student unpunished;
  3. the strict split treats `(i, j)` and `(j, i)` as one pair, so a held-out
     instance whose pair appeared in training is dropped rather than scored.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import gate  # noqa: E402
import kdistill  # noqa: E402
import student_head  # noqa: E402


def test_student_forward_signature_has_no_adjacency():
    signature = inspect.signature(gate.student_forward)
    names = {name.lower() for name in signature.parameters}
    for banned in ("adjacency", "adj", "edge", "edges", "graph", "subtype", "orbit"):
        assert banned not in names, f"student_forward exposes {banned!r}"
    source = inspect.getsource(student_head)
    for token in ("adjacency", "directed_adj", "edge_index"):
        assert token not in source, f"student_head.py mentions {token!r}"


def test_a_bundle_carrying_an_adjacency_is_refused():
    clean = {"q": torch.zeros(2, 3, 4), "mask": torch.ones(2, 3, dtype=torch.bool)}
    gate.assert_no_leak(clean)
    leaked = dict(clean, adjacency=torch.zeros(2, 3, 3))
    with pytest.raises(gate.LeakError):
        gate.assert_no_leak(leaked)
    with pytest.raises(gate.LeakError):
        gate.assert_no_leak(dict(clean, teacher_logits=torch.zeros(2, 5)))


def test_kd_loss_falls_as_the_student_approaches_the_teacher():
    generator = torch.Generator().manual_seed(2)
    teacher_logits = torch.randn(16, 6, generator=generator) * 2.0
    student = teacher_logits.clone().requires_grad_(True)
    loss_at_teacher = kdistill.kd_loss(student, teacher_logits, temperature=2.0)
    assert float(loss_at_teacher.detach()) < 1e-6, "teacher and student identical must cost zero"
    temperature = 2.0
    student = (teacher_logits + torch.randn(16, 6, generator=torch.Generator().manual_seed(9))).requires_grad_(True)
    loss = kdistill.kd_loss(student, teacher_logits, temperature=temperature)
    loss.backward()
    # d/dz of T^2 * KL(p_t || p_s) is exactly T * (p_s - p_t), and `batchmean`
    # divides by the row count: a sign test alone would pass for other monotone
    # surrogates, this fixes the function itself.
    expected = temperature / student.shape[0] * (
        torch.softmax(student.detach() / temperature, -1)
        - torch.softmax(teacher_logits / temperature, -1))
    assert torch.allclose(student.grad, expected, atol=1e-6), "KD gradient is not the KL gradient"


def test_kd_loss_points_at_the_teacher_from_a_uniform_student():
    generator = torch.Generator().manual_seed(6)
    teacher_logits = torch.randn(12, 5, generator=generator) * 2.0
    student = torch.zeros(12, 5, requires_grad=True)
    kdistill.kd_loss(student, teacher_logits, temperature=2.0).backward()
    target = teacher_logits.argmax(-1)
    gradient = student.grad.gather(1, target.unsqueeze(1)).squeeze(1)
    assert bool((gradient < 0).all()), "the teacher's own class must be pushed up"


def test_a_uniform_teacher_costs_nothing_where_the_student_is_uniform():
    teacher_logits = torch.zeros(8, 4)
    student = torch.zeros(8, 4, requires_grad=True)
    assert abs(float(kdistill.kd_loss(student, teacher_logits).detach())) < 1e-8


def test_kd_loss_is_not_symmetric():
    generator = torch.Generator().manual_seed(4)
    a = torch.randn(5, 3, generator=generator)
    b = torch.randn(5, 3, generator=generator) * 3
    forward = float(kdistill.kd_loss(a, b))
    backward = float(kdistill.kd_loss(b, a))
    assert abs(forward - backward) > 1e-4, "a symmetric loss would hide a reversed argument"


def test_pair_disjoint_split_drops_a_reversed_pair():
    train_nodes = torch.tensor([[1, 2, 3], [4, 5, 6]])
    heldout_nodes = torch.tensor([[2, 1, 9], [7, 8, 10]])
    keep = gate.pair_disjoint_keep(train_nodes, heldout_nodes)
    assert keep.tolist() == [False, True], \
        "the held-out instance shares the unordered pair (1,2) with training"


def test_pair_disjoint_split_keeps_a_clean_instance():
    train_nodes = torch.tensor([[1, 2, 3]])
    heldout_nodes = torch.tensor([[4, 5, 6], [6, 4, 7]])
    keep = gate.pair_disjoint_keep(train_nodes, heldout_nodes)
    assert keep.tolist() == [True, True]


def test_relational_head_fits_a_pairwise_label():
    """Positive control: the head CAN fit a relation defined by pairwise geometry.

    Without this, a negative result on the real codes would only say that the head
    was too weak to express a pair relation.
    """
    generator = torch.Generator().manual_seed(5)
    torch.manual_seed(0)
    head = student_head.StudentRelationalHead(8, 32, 3, 4)
    x = torch.randn(512, 3, 8, generator=generator)
    labels = torch.stack([(x[:, 0] - x[:, 1]).pow(2).sum(-1),
                          (x[:, 0] - x[:, 2]).pow(2).sum(-1),
                          (x[:, 1] - x[:, 2]).pow(2).sum(-1)], -1).argmax(-1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3)
    for _ in range(600):
        logits, _ = head(x)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    accuracy = float((head(x)[0].argmax(-1) == labels).float().mean())
    assert accuracy > 0.9, f"relational head stalled at {accuracy:.3f} on a synthetic relation"
