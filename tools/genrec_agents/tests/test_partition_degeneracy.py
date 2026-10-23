"""The partition-degeneracy gate's tests (issue #112).

The gate asks the question the campaign's verification suite had been missing: does a level's
assignment distinguish the catalogue at all? The arm that collapsed is the fixture: its exported
SIDs are in the repository, so the test is hermetic and does not need a GPU or a checkpoint.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_partition_degeneracy.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import partition_degeneracy as P  # noqa: E402

ARM_SIDS = REPO_ROOT / "results/stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser/sids_for_hgrec.npy"
CONTROL_SIDS = REPO_ROOT / "results/stage2_RQ-VAE/r8_control/item_sids.json"


def _load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.zeros((max(int(key) for key in raw) + 1, 4), dtype=np.int64)
    for key, value in raw.items():
        matrix[int(key)] = value
    return matrix


def test_the_gate_refuses_the_arm_whose_level0_collapsed():
    """The arm that passed every other gate: its level 0 held one code for all 24587 items."""
    sids = _load(ARM_SIDS)
    verdict = P.assess(sids, level=0)
    assert not verdict["ok"], "the collapsed level must be refused"
    assert verdict["distinct_codes"] == 1
    assert verdict["largest_cell"] == len(sids)
    assert verdict["failures"], "a refusal must say which condition failed"


def test_the_gate_accepts_the_archived_control():
    """The control's level 0 has 245 distinct codes and a largest cell of 237 of 24587."""
    sids = _load(CONTROL_SIDS)
    verdict = P.assess(sids, level=0)
    assert verdict["ok"], verdict["failures"]
    assert verdict["distinct_codes"] > P.GATE["min_distinct"]
    assert verdict["largest_cell_share"] < P.GATE["max_cell_share"]


def test_the_gate_refuses_a_synthetic_constant_partition():
    sids = np.stack([np.full(500, 7, dtype=np.int64),
                     np.arange(500, dtype=np.int64) % 256], axis=1)
    verdict = P.assess(sids, level=0)
    assert not verdict["ok"]
    assert any("distinct codes" in failure for failure in verdict["failures"])


def test_the_gate_refuses_a_partition_with_one_dominant_cell():
    """Distinct codes are not enough: one cell holding 60% of the catalogue is not a partition."""
    column = np.concatenate([np.zeros(300, dtype=np.int64),
                              np.arange(1, 200, dtype=np.int64)])
    sids = np.stack([column, np.arange(499, dtype=np.int64) % 256], axis=1)
    verdict = P.assess(sids, level=0)
    assert not verdict["ok"]
    assert any("largest cell" in failure for failure in verdict["failures"])


def test_the_gate_reports_the_level_it_was_asked_about():
    sids = _load(CONTROL_SIDS)
    for level in range(3):
        verdict = P.assess(sids, level=level)
        assert verdict["level"] == level
        assert verdict["items"] == len(sids)
