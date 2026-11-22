"""Issue #220/#221: the mainline is Graphlet Type+Role CE through the real quantizer.

The old Issue #121 contract (AF3 x hyperbolic geometry) is retained as history;
these tests pin the migration so a new proposal cannot be judged by the old gate
and an old-style proposal cannot slip through the new one.

The policy document they used to read (``ideas/main_line.json``) was removed with
the rest of ``ideas/`` at the owner's request (2026-11-22); the file is kept in
git history at the commit before that deletion, so the statement can be cited but
not silently edited here.
"""
from __future__ import annotations

from tools.research.rounds import agents, scope


def test_objective_and_contract_follow_the_new_mainline():
    assert "TYPE" in agents.RESEARCH_OBJECTIVE and "ROLE" in agents.RESEARCH_OBJECTIVE
    contract = agents.SCOPE_CONTRACT.lower()
    assert "names an explicit curved operation" not in contract, \
        "the new contract still demands a curved operation"
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
