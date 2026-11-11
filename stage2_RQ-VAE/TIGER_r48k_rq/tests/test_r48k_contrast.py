"""Issue #178's acceptance for the contrast itself: the switch-off identity and the gradient path."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import r48k_geometry as g  # noqa: E402


def _load_trainer():
    spec = importlib.util.spec_from_file_location("r48k_trainer", TREE / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["r48k_trainer"] = module
    spec.loader.exec_module(module)
    return module


def r48_reference_loss(residuals, source_ids, target_ids, temperature):
    """A verbatim transcription of TIGER_BEHAVIOR_R48's body, used as the identity reference."""
    batch_size = int(source_ids.shape[0])
    losses = []
    for level in range(residuals.shape[0]):
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        distances = torch.cdist(source, candidates, p=2)
        logits = -distances / temperature
        duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
        duplicate_targets.fill_diagonal_(False)
        logits = logits.masked_fill(duplicate_targets, -torch.inf)
        valid = source_ids.ne(target_ids)
        if bool(valid.any()):
            labels = torch.arange(batch_size, device=source_ids.device)
            losses.append(F.cross_entropy(logits[valid], labels[valid]))
    if not losses:
        return residuals.sum() * 0.0
    return torch.stack(losses).mean()


def _batch(seed=0, n=8, levels=3, dim=16):
    generator = torch.Generator().manual_seed(seed)
    residuals = torch.randn(levels, 2 * n, dim, generator=generator, dtype=torch.float64, requires_grad=True)
    source = torch.arange(n, dtype=torch.long)
    target = (torch.arange(n, dtype=torch.long) + 3) % n
    return residuals, source, target


def test_geometry_off_is_bitwise_the_r48_loss(monkeypatch):
    trainer = _load_trainer()
    residuals, source, target = _batch()
    monkeypatch.setattr(g, "GEOMETRY_OFF", True)
    mine = trainer.behavior_contrastive_loss(residuals, source, target)
    ref = r48_reference_loss(residuals, source, target, trainer.BEHAVIOR_TEMPERATURE)
    assert torch.equal(mine, ref)


def test_the_curvature_actually_changes_the_loss(monkeypatch):
    trainer = _load_trainer()
    residuals, source, target = _batch(seed=1)
    monkeypatch.setattr(g, "GEOMETRY_OFF", False)
    curved = trainer.behavior_contrastive_loss(residuals, source, target)
    monkeypatch.setattr(g, "GEOMETRY_OFF", True)
    flat = trainer.behavior_contrastive_loss(residuals, source, target)
    assert torch.isfinite(curved) and torch.isfinite(flat)
    assert not torch.equal(curved, flat)


def test_only_the_behaviour_loss_sends_gradient_to_the_residuals(monkeypatch):
    trainer = _load_trainer()
    residuals, source, target = _batch(seed=2)
    monkeypatch.setattr(g, "GEOMETRY_OFF", False)
    loss = trainer.behavior_contrastive_loss(residuals, source, target)
    loss.backward()
    assert residuals.grad is not None
    assert float(residuals.grad.abs().sum()) > 0.0
    assert bool(torch.isfinite(residuals.grad).all())


def test_kappa_lookup_uses_the_frozen_table_for_the_pair_ids():
    table = {"kappa": {"1": 4.0, "2": 0.25}, "mean_frozen": 0.5, "mean_kappa": 1.0}
    values = g.kappa_of(torch.tensor([1, 2, 999]), table=table)
    assert [round(float(x), 6) for x in values] == [4.0, 0.25, 0.5]


def test_kappa_from_the_lookup_is_moved_to_the_points_device_and_dtype(monkeypatch):
    # regression for the smoke crash: kappa_of returns CPU float32 while the residuals are float64 (and cuda in the real run)
    trainer = _load_trainer()
    residuals, source, target = _batch(seed=3)
    monkeypatch.setattr(g, "GEOMETRY_OFF", False)
    loss = trainer.behavior_contrastive_loss(residuals, source, target)
    assert loss.dtype == residuals.dtype
    assert torch.isfinite(loss)


def test_float32_points_still_produce_a_non_constant_matrix(monkeypatch):
    # regression for the second smoke: in float32 the acosh argument rounded to exactly 1 for every pair, so the matrix was
    # constant zeros and Sinkhorn's centring raised "Cannot center a constant distance matrix"
    monkeypatch.setattr(g, "GEOMETRY_OFF", False)
    generator = torch.Generator().manual_seed(5)
    source = (torch.randn(6, 16, generator=generator) * 0.05).float()
    candidates = (torch.randn(6, 16, generator=generator) * 0.05).float()
    kappa = torch.full((6,), 4.0)
    matrix = g.behaviour_distances(source, candidates, kappa, kappa)
    assert matrix.dtype == source.dtype
    assert float(matrix.std()) > 0.0
    assert float(matrix.max()) > float(matrix.min())
