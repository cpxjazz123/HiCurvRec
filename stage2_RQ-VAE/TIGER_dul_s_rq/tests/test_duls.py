"""DUL-S (issue #176): the batch normalisation and the measured, frozen beta."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import unique_loss as ul  # noqa: E402


def test_normalisation_is_the_batch_size_not_the_pair_count():
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    tokens = torch.tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], dtype=torch.long)
    loss = ul.uniqueness_loss(z, tokens)
    # two pairs of cos 1 each, relu(1 - 0.9) = 0.1 -> total 0.2; divided by B = 4 (not |P| = 2)
    assert abs(float(loss) - 0.05) < 1e-6   # float32 accumulation, not a bound on the definition


def test_zero_pairs_is_a_finite_zero():
    z = torch.randn(3, 4)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.long)
    loss = ul.uniqueness_loss(z, tokens)
    assert float(loss) == 0.0 and torch.isfinite(loss)


def test_beta_starts_at_the_paper_value_and_freezes(monkeypatch):
    monkeypatch.setattr(ul, "BETA_MEASURED", None)
    assert ul.beta() == ul.BETA_UNIQUE
    ul.set_beta(3.5)
    assert ul.beta() == 3.5
    ul.set_beta(1.0)
    assert ul.beta() == 1.0


def test_gradient_path_survives_the_new_normalisation():
    base = torch.tensor([[1.0] * 4, [1.0, 1.0, 1.0, 0.9], [0.0] * 4, [0.0, 0.0, 0.0, 0.1]])
    z = base.clone().requires_grad_(True)
    tokens = torch.tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], dtype=torch.long)
    loss = ul.uniqueness_loss(z, tokens)
    loss.backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0.0
