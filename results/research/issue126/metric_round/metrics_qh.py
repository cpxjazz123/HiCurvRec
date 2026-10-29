"""Q_H (Issue #126) as a bench metric, so the incumbent is measured the same way."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from tools.research.science import hyperbolic_sid_quality as frozen


def q_h_level1(table: dict, context: dict) -> float:
    """Q_H at level 1, computed by the frozen Issue #126 routine itself."""
    rows = np.asarray(table["rows"], dtype=np.int64)
    payload = {}
    for position, item in enumerate(table["item_ids"].tolist()):
        payload[str(item)] = rows[position].tolist()
    import os
    temporary = (REPO / "results" / "research" / "issue126" / "metric_round" /
                 ("_qh_tmp_%d.json" % os.getpid()))
    temporary.write_text(__import__("json").dumps(payload), encoding="utf-8")
    try:
        report = frozen.score(temporary, reference=context["reference"])
    finally:
        temporary.unlink(missing_ok=True)
    return float(report["q_h"])


q_h_level1.note = "the frozen Issue #126 score at level 1 (the incumbent)"

METRICS = {"q_h_level1": q_h_level1}
FREE_CONSTANTS = {"q_h_level1": 0}
