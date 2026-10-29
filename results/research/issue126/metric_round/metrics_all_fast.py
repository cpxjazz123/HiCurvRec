"""The cheap constructions of the metric rounds, in one bench run."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_all = _load("metrics_all")
SLOW = _all.SLOW

METRICS = {name: fn for name, fn in _all.METRICS.items() if name not in SLOW}
FREE_CONSTANTS = {name: value for name, value in _all.FREE_CONSTANTS.items()
                  if name not in SLOW}
