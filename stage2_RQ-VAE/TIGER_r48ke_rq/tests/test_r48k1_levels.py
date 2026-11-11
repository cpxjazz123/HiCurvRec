"""Issue #182: the three states of the contrast -- all Euclidean (R48), level 1 only (this arm), all curved (#179)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import r48k_geometry as g  # noqa: E402


def _trainer():
    spec = importlib.util.spec_from_file_location("r48k1_trainer", TREE / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["r48k1_trainer"] = module
    spec.loader.exec_module(module)
    return module


def _batch(seed=0, n=8, levels=3, dim=16):
    generator = torch.Generator().manual_seed(seed)
    residuals = torch.randn(levels, 2 * n, dim, generator=generator, dtype=torch.float64)
    source = torch.arange(n, dtype=torch.long)
    target = (torch.arange(n, dtype=torch.long) + 3) % n
    return residuals, source, target


def _level_logits(trainer, level, residuals, source, target):
    """The per-level logits the trainer builds, recomputed here from the same pieces."""
    n = int(source.shape[0])
    src, cand = residuals[level, :n], residuals[level, n:]
    if g.GEOMETRY_OFF or (level >= 1 and g.LEVEL1_ONLY):
        distances = torch.cdist(src, cand, p=2)
    else:
        distances = g.behaviour_distances(src, cand, g.kappa_of(source), g.kappa_of(target))
    return -distances / trainer.BEHAVIOR_TEMPERATURE


def test_level_one_only_curves_the_first_level(monkeypatch):
    trainer = _trainer()
    residuals, source, target = _batch()
    monkeypatch.setattr(g, "GEOMETRY_OFF", False)
    monkeypatch.setattr(g, "LEVEL1_ONLY", True)
    level1 = _level_logits(trainer, 0, residuals, source, target)
    level2 = _level_logits(trainer, 1, residuals, source, target)
    # level 1 is curved, levels 2/3 are the R48 Euclidean logits
    assert not torch.equal(level1, -torch.cdist(residuals[0, :8], residuals[0, 8:]) / trainer.BEHAVIOR_TEMPERATURE)
    assert torch.equal(level2, -torch.cdist(residuals[1, :8], residuals[1, 8:]) / trainer.BEHAVIOR_TEMPERATURE)


def test_all_levels_keeps_every_level_curved(monkeypatch):
    trainer = _trainer()
    residuals, source, target = _batch(seed=1)
    monkeypatch.setattr(g, "GEOMETRY_OFF", False)
    monkeypatch.setattr(g, "LEVEL1_ONLY", False)
    for level in (0, 1, 2):
        got = _level_logits(trainer, level, residuals, source, target)
        euclid = -torch.cdist(residuals[level, :8], residuals[level, 8:]) / trainer.BEHAVIOR_TEMPERATURE
        assert not torch.equal(got, euclid), level


def test_geometry_off_is_euclidean_on_every_level(monkeypatch):
    trainer = _trainer()
    residuals, source, target = _batch(seed=2)
    monkeypatch.setattr(g, "GEOMETRY_OFF", True)
    for level in (0, 1, 2):
        got = _level_logits(trainer, level, residuals, source, target)
        euclid = -torch.cdist(residuals[level, :8], residuals[level, 8:]) / trainer.BEHAVIOR_TEMPERATURE
        assert torch.equal(got, euclid), level
