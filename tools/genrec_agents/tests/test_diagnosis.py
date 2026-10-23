"""The per-arm diagnosis's tests (issue #109, section 5).

The controller has to say which hypothesis failed for an arm, and say it with the
measurement that produced it. The tests therefore check that each mode is raised by its own
evidence and not by anything else, and that a healthy arm is not diagnosed as broken.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_diagnosis.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import diagnosis as D  # noqa: E402


def _modes(**kwargs):
    row = D.diagnose(mechanism_id="candidate_one", **kwargs)
    return [mode["mode"] for mode in row["modes"]]


def test_a_measured_gradient_mismatch_is_diagnosed():
    assert "invalid_gradient" in _modes(
        screening={"judges": {"failed": ["gradient_source"], "detail": "1 of 8 checks failed"}}
    )


def test_an_inactive_export_is_diagnosed():
    assert "inactive_hard_sid_path" in _modes(
        export_sensitivity={"changed_fraction": {"levels_3": 0.0}}
    )


def test_a_ratio_moved_through_its_denominator_is_diagnosed():
    """The ER rose while the positive share did not move: the denominator did it."""
    assert "pair_share_ratio_artifact" in _modes(
        evaluation={
            "hard_l1_er": 17.1,
            "pos_share": 0.0791,
            "neg_share": 0.0031,
        }
    )


def test_a_ratio_that_moved_through_its_numerator_is_not_that_failure():
    assert "pair_share_ratio_artifact" not in _modes(
        evaluation={
            "hard_l1_er": 17.1,
            "pos_share": 0.0931,
            "neg_share": 0.0046,
        }
    )


def test_a_g_h1_below_the_historical_range_is_diagnosed():
    assert "lost_history_predictability" in _modes(probe={"g_h1": 0.001})
    assert "lost_history_predictability" not in _modes(probe={"g_h1": 0.12})


def test_an_unsupported_conditional_l2_is_diagnosed():
    assert "conditional_l2_contamination" in _modes(
        evaluation={"d2_flags": ["low_support"]}
    )
    assert "conditional_l2_contamination" not in _modes(
        evaluation={"d2_flags": []}
    )


def test_a_grown_tail_is_diagnosed():
    assert "concentration_or_collisions" in _modes(
        structure={"collisions": {"level_1": {"largest_cell": 374}}}
    )


def test_a_narrowed_tree_is_diagnosed():
    assert "concentration_or_collisions" in _modes(
        structure={"prefix_structure": {"l1_to_l2": {"mean_branches": 60.0}}}
    )


def test_a_healthy_arm_is_not_diagnosed_as_broken():
    modes = _modes(
        evaluation={"hard_l1_er": 16.95, "pos_share": 0.09, "neg_share": 0.0048},
        stage3={"test_recall@10": 0.0645, "test_ndcg@10": 0.0355},
        probe={"g_h1": 0.12},
        structure={
            "collisions": {"level_1": {"largest_cell": 230}},
            "prefix_structure": {"l1_to_l2": {"mean_branches": 66.0}},
        },
    )
    assert modes == []


def test_a_grown_tail_is_reported_with_its_number():
    """The mode carries the measurement, so a reader sees how far it grew."""
    row = D.diagnose(
        mechanism_id="candidate_one",
        structure={"collisions": {"level_1": {"largest_cell": 240}}},
    )
    mode = next(m for m in row["modes"] if m["mode"] == "concentration_or_collisions")
    assert "240" in mode["evidence"]
    assert "237" in mode["evidence"]


def test_the_summary_counts_the_modes():
    rows = [
        D.diagnose(mechanism_id="a", structure={"collisions": {"level_1": {"largest_cell": 374}}}),
        D.diagnose(mechanism_id="b", screening={"judges": {"failed": ["gradient_source"]}}),
        D.diagnose(mechanism_id="c", evaluation={"hard_l1_er": 16.4}),
    ]
    summary = D.summarise(rows)
    assert summary["arms"] == 3
    assert summary["modes"]["concentration_or_collisions"] == 1
    assert summary["modes"]["invalid_gradient"] == 1
    assert summary["arms_with_no_diagnosis"] == 1
    # the mean magnitude is reported with the count, because a mode every arm carries
    # says nothing about which of them it explains
    assert summary["mean_magnitude"]["concentration_or_collisions"] == 137.0
