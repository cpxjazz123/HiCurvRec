"""Agent A's tests (issue #110, WP1).

The analysis has to be honest about what it knows: a number read from an artifact is
measured, a number that does not exist is missing, and the failure classification must
separate an implementation failure from a falsified mechanism from a measurement that has no
support.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_scientific_analysis.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import scientific_analysis as SA  # noqa: E402


def _manifests(**metrics):
    return [
        {
            "mechanism_id": "arm_one",
            "kind": "stage2",
            "metrics": metrics,
        }
    ]


def _structure(**overrides):
    structure = {
        "token_balance": {"level_1": {"distinct_codes": 255, "gini": 0.229}},
        "prefix_structure": {"l1_to_l2": {"mean_branches": 63.8}, "l2_to_l3": {"mean_branches": 1.45}},
        "collisions": {"level_1": {"largest_cell": 374}},
        "sequential_predictability": {"level_1": {"excess_over_the_marginal": 0.1045}},
        "stage3_slices": {"available": True, "levels": {"l1": {"accuracy": 0.09}}},
    }
    structure.update(overrides)
    return structure


def test_the_diagnostics_carry_the_evidence_status():
    diagnostics = SA.arm_diagnostics(
        "arm_one",
        _manifests(
            frozen_evaluation={
                "hard_l1_er": 16.9,
                "pos_share": 0.09,
                "neg_share": 0.0048,
                "d2": 0.043,
                "d2_block": {"denominator_positive": 563, "flags": ["low_support"]},
            },
            gh1_probe={"g_h1": 0.123, "prior_ce_l1": 5.33},
        ),
        _structure(),
    )
    assert diagnostics["enrichment_ratio"]["value"] == 16.9
    assert diagnostics["enrichment_ratio"]["cleared_the_gate"] is True
    assert diagnostics["enrichment_ratio"]["numerator_positive_share"]["kind"] == "measured"
    assert diagnostics["conditional_l2_discrimination"]["low_support"] is True
    assert diagnostics["conditional_l2_discrimination"]["support"] == 563
    assert diagnostics["history_gain"]["value"] == 0.123
    assert diagnostics["level0_usage"]["delta_largest_cell_vs_control"]["value"] == 374 - 237
    assert diagnostics["prefix_tree"]["delta_branches_vs_control"]["value"] == pytest.approx(
        63.8 - SA.CONTROL["l1_to_l2_branches"]
    )


def test_a_missing_number_is_missing_and_not_zero():
    diagnostics = SA.arm_diagnostics("arm_one", _manifests(), _structure())
    assert diagnostics["enrichment_ratio"]["kind"] == "missing"
    assert diagnostics["history_gain"]["kind"] == "missing"
    assert diagnostics["downstream"]["recall"]["value"] is None


def test_an_implementation_failure_is_decided_before_the_mechanism_is_judged():
    diagnostics = SA.arm_diagnostics("arm_one", _manifests(), _structure())
    failure = SA.classify_failure(
        diagnostics, {"judges": {"failed": ["gradient_source"]}}
    )
    assert failure["kind"] == "implementation_failure"


def test_an_inactive_export_is_an_implementation_failure():
    diagnostics = SA.arm_diagnostics("arm_one", _manifests(), _structure())
    failure = SA.classify_failure(
        diagnostics, {"export_sensitivity": {"verdict": "inert"}}
    )
    assert failure["kind"] == "implementation_failure"


def test_a_measured_arm_that_missed_is_a_falsified_mechanism():
    diagnostics = SA.arm_diagnostics(
        "arm_one",
        _manifests(
            frozen_evaluation={"hard_l1_er": 16.9, "d2": 0.04, "d2_block": {"denominator_positive": 5000}},
            stage3={"test_recall@10": 0.06, "test_ndcg@10": 0.03},
        ),
        _structure(),
    )
    failure = SA.classify_failure(diagnostics)
    assert failure["kind"] == "mechanism_falsified"
    assert "did not hold" in failure["evidence"]


def test_an_unsupported_measurement_is_low_support_not_a_conclusion():
    diagnostics = SA.arm_diagnostics(
        "arm_one",
        _manifests(frozen_evaluation={"hard_l1_er": 16.9}),
        _structure(),
    )
    failure = SA.classify_failure(diagnostics)
    assert failure["kind"] == "statistical_low_support"


def test_an_arm_that_beat_everything_has_no_failure():
    diagnostics = SA.arm_diagnostics(
        "arm_one",
        _manifests(
            frozen_evaluation={"hard_l1_er": 16.9, "d2": 0.04, "d2_block": {"denominator_positive": 5000}},
            stage3={"test_recall@10": 0.07, "test_ndcg@10": 0.04},
        ),
        _structure(),
    )
    failure = SA.classify_failure(diagnostics)
    assert failure["kind"] == "no_failure_identified"


def test_the_analysis_and_its_outputs_are_written(tmp_path):
    runs = _manifests(
        frozen_evaluation={"hard_l1_er": 16.9, "d2": 0.04, "d2_block": {"denominator_positive": 5000}},
        stage3={"test_recall@10": 0.06, "test_ndcg@10": 0.03},
    )
    analysis = SA.failure_analysis(runs, {"arm_one": _structure()})
    assert analysis["arm_count"] == 1
    assert analysis["measured_arms"] == 1
    questions = SA.research_questions(analysis)
    assert questions
    assert questions[0]["kind"] == "measured"
    hypotheses = SA.causal_hypotheses(analysis)
    assert hypotheses
    written = SA.write_analysis(analysis, tmp_path)
    assert json.loads(written["failure_analysis.json"].read_text())["arm_count"] == 1
    assert written["research_questions.json"].is_file()
    assert written["causal_hypotheses.json"].is_file()
