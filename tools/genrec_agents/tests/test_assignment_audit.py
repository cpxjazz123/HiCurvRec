"""The per-example assignment audit's tests (issue #115's JOIN R blocker)."""
from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.genrec_agents import assignment_audit as A  # noqa: E402


def _matrices(flip: bool, n=400, codes=8):
    rng = np.random.default_rng(0)
    full = rng.random((n, codes))
    ablated = full.copy()
    if flip:
        ablated[:, 0] -= 1.0        # a different code now wins
    return full, ablated


def test_a_no_op_term_is_refused():
    full, ablated = _matrices(flip=False)
    out = A.audit(full, ablated)
    assert not out["ok"] and out["flip_rate"] == 0.0
    assert "no-op" in out["failures"][0] or "flips" in out["failures"][0]


def test_a_term_that_changes_the_decision_passes():
    full, ablated = _matrices(flip=True)
    out = A.audit(full, ablated)
    assert out["ok"], out["failures"]
    assert out["flip_rate"] > A.GATE["min_flip_rate"]
    assert out["changed_codes_per_stage"] > 0


def test_the_criterion_is_the_flip_rate_not_a_variance():
    """A term can move every score and change no decision: the audit must still refuse it."""
    rng = np.random.default_rng(1)
    full = rng.random((200, 6))
    ablated = full + 0.5      # every score moves, the ordering does not
    out = A.audit(full, ablated)
    assert out["flip_rate"] == 0.0 and not out["ok"]


def test_types_are_checked():
    with pytest.raises(ValueError):
        A.audit(np.zeros((4, 3)), np.zeros((4, 4)))


def test_emit_writes_the_record_a_reviewer_can_check(tmp_path):
    full, ablated = _matrices(flip=True)
    out = A.emit(tmp_path / "audit.json", full, ablated, stage="level0",
                 components={"area": full[:, :1], "distance": full[:, 1:2]},
                 clamp_count=0, error_count=0)
    assert Path(out["path"]).is_file()
    doc = json.loads(Path(out["path"]).read_text())
    assert doc["stage"] == "level0"
    assert len(doc["per_example"]["winner_full"]) == doc["summary"]["examples"]
    assert set(doc["components"]) == {"area", "distance"}
    assert doc["clamp_count"] == 0 and doc["thresholds_frozen_before_metrics"] is True
    assert doc["summary"]["ok"] is True
