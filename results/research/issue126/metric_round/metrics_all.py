"""Every metric the rounds produced, in one bench run.

The individual modules stay where the round that produced them put them; this
one imports them so that a single measurement covers the whole construction set
on the corrected collection.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_round1 = _load("metrics_cheap")          # voronoi margin, adjusted dispersion
_round2 = _load("metrics_round2")         # tree-conditional path likelihood
_round3 = _load("metrics_round3")         # size-matched reach lift
_round4b = _load("metrics_round4b")       # geodesic covering radius
_round6 = _load("metrics_round6")         # expected-rank and mass lifts
_hbaa = _load("metrics_hbaa")             # HBAA and its controls
_qh = _load("metrics_qh")                 # the Issue #126 incumbent

METRICS = {}
METRICS.update(_round1.METRICS)
METRICS.update(_round2.METRICS)
METRICS.update(_round3.METRICS)
METRICS.update(_round4b.METRICS)
METRICS.update(_round6.METRICS)
METRICS.update(_qh.METRICS)
METRICS["hbaa"] = _hbaa.METRICS["hbaa"]
METRICS["hbaa_validation"] = _hbaa.METRICS["hbaa_validation"]

#: The three constructions that cost a minute or more per table, kept apart so
#: that a benchmark of the cheap ones is not held hostage by them.
SLOW = ("hyp_hsoftmax_path", "hyp_hsoftmax_path_null", "hyp_bucket_reach_lift_l1")

FREE_CONSTANTS = {}
for module in (_round1, _round2, _round3, _round4b, _round6, _qh, _hbaa):
    FREE_CONSTANTS.update(getattr(module, "FREE_CONSTANTS", {}))
