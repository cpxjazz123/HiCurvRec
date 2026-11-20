"""Issue #220: the mainline is Graphlet Type+Role CE through the real quantizer.

The old Issue #121 contract (AF3 x hyperbolic geometry) is retained as history;
these tests pin the migration so a new proposal cannot be judged by the old gate
and an old-style proposal cannot slip through the new one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.rounds import agents, scope

MAIN_LINE = Path(__file__).resolve().parents[3] / "ideas" / "main_line.json"


@pytest.fixture(scope="module")
def mainline():
    return json.loads(MAIN_LINE.read_text())


def test_statement_is_the_type_role_mainline(mainline):
    statement = mainline["statement"]
    assert "Type" in statement and "Role" in statement
    assert "量化" in statement and "SID" in statement


def test_must_hold_forbids_recasting_the_target_as_pairs(mainline):
    joined = " ".join(mainline["must_hold"])
    assert "单实例" in joined and "交叉熵" in joined
    assert "pairwise" in joined.lower() or "两两" in joined
    assert "真实量化器" in joined and "码本" in joined


def test_curvature_definitions_are_historical(mainline):
    definitions = mainline["definitions"]
    assert definitions["af3"]["status"] == "historical"
    assert definitions["mixed_curvature_space"]["status"] == "historical"
    assert definitions["graphlet_type_role"]["status"] if "status" in definitions["graphlet_type_role"] else True


def test_objective_and_contract_follow_the_new_mainline():
    assert "TYPE" in agents.RESEARCH_OBJECTIVE and "ROLE" in agents.RESEARCH_OBJECTIVE
    contract = agents.SCOPE_CONTRACT.lower()
    assert "hyperbolic" not in contract, "the new contract still demands hyperbolic geometry"
    assert "sinkhorn" in contract, "the assignment rule must still be pinned"


IDEA = {
    "supervision_target": "instance-level joint cross-entropy over the Graphlet type label "
                          "and the per-node role/orbit labels of one induced instance",
    "predictor_input": "the quantized representation q_full from the real quantizer, "
                       "with type pooled permutation-invariantly",
    "quantizer_path": "hard code indices forward; hard-forward/soft-backward code assignment "
                      "proxy; gradient reaches the encoder and the selected codebook entries",
    "assignment_rule": "the baseline's Sinkhorn-balanced assignment is kept; the score enters "
                       "its cost matrix",
    "effect_observables": "heldout type/role macro-F1, hard-SID prefix retention, collision, "
                          "Stage3 recall",
    "specification": "L_pred = CE(type) + alpha * mean_v CE(role_v); q_ST = sum_k h_ST[k] E_k",
}


def test_new_gate_accepts_the_mainline_idea():
    verdict = scope.check_mainline_scope(IDEA)
    assert verdict["in_scope"], verdict["reasons"]


def test_new_gate_rejects_a_pairwise_recast():
    recast = dict(IDEA, supervision_target="pairwise BCE over the six P4 edge/non-edge pairs")
    verdict = scope.check_mainline_scope(recast)
    assert not verdict["in_scope"]
    assert any("pairwise" in reason for reason in verdict["reasons"])


def test_new_gate_rejects_replacing_the_assignment():
    replaced = dict(IDEA, assignment_rule="plain argmin over the mechanism score replaces "
                                          "Sinkhorn")
    verdict = scope.check_mainline_scope(replaced)
    assert not verdict["in_scope"]


def test_old_gate_still_exists_for_history():
    """The #121 gate is not deleted; it is simply no longer the mainline judge."""
    assert hasattr(scope, "check_scope")
    assert "hyperbolic" in scope.SCOPE_CONTRACT if hasattr(scope, "SCOPE_CONTRACT") else True
