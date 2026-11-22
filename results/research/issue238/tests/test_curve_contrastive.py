"""The curve contrastive loss must be a genuine two-sided coupling.

CLAUDE.md 19h rules out either side alone, so the three switch states have to be distinguishable:
curve off must reproduce the archived cosine InfoNCE exactly, behaviour off must destroy the edge
signal, and both on must differ from both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _armloader import load_arm  # noqa: E402

load_arm("TIGER_curve_contrastive_rq")
from losses import behavior_contrastive as bc  # noqa: E402


def _tensors(b=6, k=4, d=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, d), torch.randn(b, d), torch.randn(b, k, d)


def _archived_cosine(anchor, positive, negatives, temperature=0.07):
    a, p, n = (F.normalize(t, dim=-1) for t in (anchor, positive, negatives))

    def direction(first, second, other):
        pos = (first * second).sum(-1, keepdim=True)
        neg = torch.einsum("bd,bkd->bk", first, other)
        logits = torch.cat([pos, neg], dim=1) / temperature
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)

    return 0.5 * (direction(a, p, n) + direction(p, a, n))


def test_curve_off_reproduces_the_archived_info_nce():
    a, p, n = _tensors()
    got = bc.behaviour_contrastive_loss(a, p, n, access_behavior=True, access_curve=False)
    assert torch.allclose(got, _archived_cosine(a, p, n), rtol=1e-6, atol=1e-6)


def test_curve_on_differs_from_curve_off():
    a, p, n = _tensors()
    flat = bc.behaviour_contrastive_loss(a, p, n, access_curve=False)
    curved = bc.behaviour_contrastive_loss(a, p, n, access_curve=True)
    assert not torch.allclose(flat, curved, rtol=1e-4, atol=1e-4)


def test_behaviour_off_differs_from_behaviour_on():
    a, p, n = _tensors()
    on = bc.behaviour_contrastive_loss(a, p, n, access_behavior=True)
    off = bc.behaviour_contrastive_loss(a, p, n, access_behavior=False)
    assert not torch.allclose(on, off, rtol=1e-4, atol=1e-4)


def test_the_both_on_state_is_not_the_same_as_either_single_sided_state():
    a, p, n = _tensors()
    both = bc.behaviour_contrastive_loss(a, p, n)
    no_curve = bc.behaviour_contrastive_loss(a, p, n, access_curve=False)
    no_behavior = bc.behaviour_contrastive_loss(a, p, n, access_behavior=False)
    assert not torch.allclose(both, no_curve, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(both, no_behavior, rtol=1e-4, atol=1e-4)


def test_the_loss_is_finite_and_differentiable():
    a, p, n = _tensors()
    a = a.requires_grad_(True)
    value = bc.behaviour_contrastive_loss(a, p, n)
    assert torch.isfinite(value)
    value.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_a_degenerate_batch_does_not_produce_nan():
    a, p, n = _tensors(seed=7)
    same = a.clone()
    value = bc.behaviour_contrastive_loss(a, same, n)
    assert torch.isfinite(value)
