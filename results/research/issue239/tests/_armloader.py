"""Isolate one Stage 2 arm's modules per test module.

Every test file here inserts a different candidate tree on sys.path and imports `distances`,
`model` and `losses` from it.  pytest collects them in one process, so without this the second
file would silently import the *first* tree's package -- which is exactly what happened: the two
graphlet tests resolved `distances` to TIGER_kappa_cost_rq, whose registry has no `graphlet`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_PACKAGES = ("distances", "model", "losses", "train_config", "train_rqvae")


def load_arm(arm: str) -> Path:
    """Drop every cached arm module, then put this arm first on sys.path."""
    for name in list(sys.modules):
        head = name.split(".")[0]
        if head in _PACKAGES:
            del sys.modules[name]
    path = ROOT / "stage2_RQ-VAE" / arm
    while str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))
    return path
