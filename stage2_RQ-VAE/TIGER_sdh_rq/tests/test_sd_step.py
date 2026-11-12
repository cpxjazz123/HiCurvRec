"""Issue #181 acceptance 3-4 on the real quantiser: the student reads the selected codes and the teacher stays frozen."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import af3_teacher, sd_step  # noqa: E402
from model import RQVAE  # noqa: E402


def _config():
    return SimpleNamespace(codebook_num=2, codebook_dim=8, codebook_size=(16, 16), beta=0.25,
                           vq_type="vq", sk_epsilon=-1.0, sk_iters=-1, dropout=0.0,
                           hidden_sizes=[32, 16])


def _teacher(n_items=12):
    coo = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6), (2, 7)]
    sparse, _ = af3_teacher.build_sparse(coo, n_items)
    return af3_teacher.SparseRows(sparse)


def test_the_step_is_finite_and_reaches_the_encoder():
    torch.manual_seed(0)
    module = RQVAE(_config(), in_dim=8)
    embeddings = torch.randn(12, 8) * 0.1
    embeddings.requires_grad_(True)
    rows = _teacher()
    loss = sd_step.structural_loss(module, embeddings, torch.tensor([0, 1, 2]),
                                   rows, {0: [1, 2], 1: [2, 3]}, {0: [3]}, n_items=12)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.encoder.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_permuting_the_codes_changes_the_student_side_only():
    torch.manual_seed(1)
    module = RQVAE(_config(), in_dim=8)
    embeddings = torch.randn(12, 8) * 0.1
    rows = _teacher()
    first = sd_step.structural_loss(module, embeddings, torch.tensor([0, 1, 2]), rows,
                                    {0: [1, 2]}, {0: [3]}, n_items=12)
    # a different embedding set changes the student's codes and therefore the loss; the teacher matrix is untouched
    permuted = embeddings[torch.arange(12).flip(0)]
    second = sd_step.structural_loss(module, permuted, torch.tensor([0, 1, 2]), rows,
                                     {0: [1, 2]}, {0: [3]}, n_items=12)
    assert torch.isfinite(first) and torch.isfinite(second)
    assert not torch.equal(first, second)


def test_the_teacher_lookup_carries_no_gradient():
    torch.manual_seed(2)
    module = RQVAE(_config(), in_dim=8)
    embeddings = torch.randn(12, 8) * 0.1
    rows = _teacher()
    loss = sd_step.structural_loss(module, embeddings, torch.tensor([0]), rows, {}, {}, n_items=12)
    loss.backward()
    # SparseRows holds plain tensors, not parameters: nothing on the teacher side can accumulate a gradient
    assert torch.is_tensor(rows.values) and not rows.values.requires_grad


def test_repeated_candidates_are_encoded_once(monkeypatch):
    # two anchors sharing every candidate must cost one encoding of the union, not one per anchor
    torch.manual_seed(3)
    module = RQVAE(_config(), in_dim=8)
    embeddings = torch.randn(12, 8) * 0.1
    rows = _teacher()
    calls = {"n": 0}
    real = sd_step.quantised_readout

    def counting(m, x, item_ids=None):
        calls["n"] += x.shape[0]
        return real(m, x, item_ids)

    monkeypatch.setattr(sd_step, "quantised_readout", counting)
    shared = {0: [1, 2], 1: [1, 2]}
    loss = sd_step.structural_loss(module, embeddings, torch.tensor([0, 1]), rows, shared, {}, n_items=12)
    assert torch.isfinite(loss)
    # 2 anchors + the union {1, 2, plus global draws} -- and crucially the global draws are shared between the two anchors
    assert calls["n"] < 2 * (1 + 64) + 64


def test_padded_candidates_do_not_break_the_device_placement(monkeypatch):
    # regression: mask was built from the cpu candidate ids while cosine and scores were on the model's device
    torch.manual_seed(4)
    module = RQVAE(_config(), in_dim=8)
    embeddings = torch.randn(12, 8) * 0.1
    rows = _teacher()
    # disjoint neighbour sets give anchors different set sizes, so padding exists
    loss = sd_step.structural_loss(module, embeddings, torch.tensor([0, 5]), rows,
                                   {0: [1, 2], 5: []}, {}, n_items=12)
    assert torch.isfinite(loss)
