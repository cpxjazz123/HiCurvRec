"""The uniqueness loss of issue #175: pair set, gradient path, switches, degenerate batch."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import unique_loss as ul  # noqa: E402


def _tokens(rows: list[tuple[int, int, int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def test_pair_set_is_exactly_the_shared_code_triples():
    tokens = _tokens([(1, 2, 3), (1, 2, 3), (4, 5, 6), (1, 2, 3), (7, 8, 9)])
    groups = ul.collision_pairs(tokens)
    sizes = sorted(int(group.numel()) for group in groups)
    assert sizes == [3]                     # one group of three identical triples, no singleton group


def test_same_codes_but_same_item_is_still_a_pair_of_rows():
    # the issue defines P by identical codes with distinct items; rows are distinct items by construction here
    tokens = _tokens([(2, 2, 2), (2, 2, 2)])
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = ul.uniqueness_loss(z, tokens)
    assert float(loss) > 0.0                 # cos = 1 > m = 0.9


def test_loss_is_zero_and_finite_when_a_batch_has_no_collision():
    tokens = _tokens([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    z = torch.randn(3, 4)
    loss = ul.uniqueness_loss(z, tokens)
    assert float(loss) == 0.0 and torch.isfinite(loss)


def test_gradient_reaches_z_only_through_the_cosine():
    # the pairs have to be *above* the margin, or relu saturates and the loss is a constant zero with no gradient
    base = torch.tensor([[1.0] * 5, [1.0] * 4 + [0.95], [-1.0] * 5, [-1.0] * 4 + [-0.95]])
    z = (base + 0.01 * torch.randn(4, 5)).requires_grad_(True)
    tokens = _tokens([(1, 1, 1), (1, 1, 1), (2, 2, 2), (2, 2, 2)])
    loss = ul.uniqueness_loss(z, tokens)
    loss.backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0.0


def test_margin_direction_is_as_written():
    # a larger m is *more permissive* in this formula, the opposite of the paper's prose; the code follows the formula
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    tokens = _tokens([(1, 1, 1), (1, 1, 1)])
    low = float(ul.uniqueness_loss(z, tokens, margin=0.5))
    high = float(ul.uniqueness_loss(z, tokens, margin=0.99))
    assert low > high


def test_off_switch_zeroes_the_term(monkeypatch):
    monkeypatch.setattr(ul, "UNIQUE_OFF", True)
    z = torch.randn(4, 3, requires_grad=True)
    tokens = _tokens([(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)])
    loss = ul.uniqueness_loss(z, tokens)
    assert float(loss) == 0.0


def test_random_pair_source_still_produces_a_loss(monkeypatch):
    monkeypatch.setattr(ul, "PAIR_SOURCE_OFF", True)
    generator = torch.Generator().manual_seed(0)
    z = torch.randn(8, 3)
    tokens = _tokens([(i, i, i) for i in range(8)])
    loss = ul.uniqueness_loss(z, tokens, generator=generator)
    assert torch.isfinite(loss)
