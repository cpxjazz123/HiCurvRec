"""The success predicate's tests (issue #110, section 4 and the section 7 acceptance list).

The predicate is the only thing that may declare scientific success, so the tests are about
what it refuses: a Stage-3-only win, an ER-only win, a tie, a measurement from a different
implementation or a different seed, an incomplete Stage-3, a missing gate, a gate declared
not-applicable or insufficient-evidence, and a certificate bound to another proposal. One
test shows it accepts a genuine win, with every threshold strictly exceeded.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_success.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import success as S  # noqa: E402


def _gates(prefix: str = "H", names=S.HYPERBOLIC_GATES) -> dict:
    return {gate: {"state": "pass"} for gate in names}


def _evidence(**overrides):
    evidence = {
        "proposal_sha256": "p" * 64,
        "implementation_sha256": "i" * 64,
        "frozen_protocol_hash": "f" * 64,
        "seed": 42,
        "hyperbolic_certificate": {**_gates(), "proposal_sha256": "p" * 64},
        "cgv_certificate": {**_gates(prefix="G", names=S.CGV_GATES), "proposal_sha256": "p" * 64},
        "frozen_evaluation": {
            "hard_l1_er": 16.9,
            "pos_share": 0.09,
            "neg_share": 0.0048,
        },
        "stage3": {
            "test_recall@10": 0.0645,
            "test_ndcg@10": 0.0355,
            "n_eval": 57439,
            "complete": True,
        },
    }
    evidence.update(overrides)
    return evidence


def test_a_genuine_win_is_accepted():
    verdict = S.is_verified_triple_win(_evidence())
    assert verdict.won
    assert all(check.ok for check in verdict.checks)


def test_a_stage3_only_win_is_not_a_win():
    """The frozen ER is the first gate: without it there is no scientific success."""
    evidence = _evidence()
    evidence["frozen_evaluation"] = {"hard_l1_er": 16.2, "pos_share": 0.07, "neg_share": 0.005}
    verdict = S.is_verified_triple_win(evidence)
    assert not verdict.won
    assert "frozen_er" in [check.name for check in verdict.checks if not check.ok]


def test_an_er_only_win_is_not_a_win():
    evidence = _evidence()
    evidence["stage3"] = {"test_recall@10": 0.05, "test_ndcg@10": 0.02, "complete": True}
    verdict = S.is_verified_triple_win(evidence)
    assert not verdict.won
    failed = {check.name for check in verdict.checks if not check.ok}
    assert {"stage3_recall", "stage3_ndcg"} <= failed


def test_a_tie_is_not_a_win():
    """The thresholds are strict: equal is not above."""
    evidence = _evidence()
    evidence["frozen_evaluation"]["hard_l1_er"] = S.ER_THRESHOLD
    assert not S.is_verified_triple_win(evidence).won
    evidence = _evidence()
    evidence["stage3"]["test_recall@10"] = S.RECALL_THRESHOLD
    assert not S.is_verified_triple_win(evidence).won
    evidence = _evidence()
    evidence["stage3"]["test_ndcg@10"] = S.NDCG_THRESHOLD
    assert not S.is_verified_triple_win(evidence).won


def test_a_different_seed_is_not_a_win():
    assert not S.is_verified_triple_win(_evidence(seed=43)).won
    assert not S.is_verified_triple_win(_evidence(seed=None)).won


def test_a_missing_identity_is_not_a_win():
    for field in ("proposal_sha256", "implementation_sha256", "frozen_protocol_hash"):
        assert not S.is_verified_triple_win(_evidence(**{field: ""})).won


def test_a_missing_or_failing_gate_is_not_a_win():
    evidence = _evidence()
    evidence["hyperbolic_certificate"] = {**_gates(), "H2": {"state": "fail"}}
    assert not S.is_verified_triple_win(evidence).won
    evidence = _evidence()
    evidence["cgv_certificate"] = {**_gates(prefix="G", names=S.CGV_GATES), "G3": {"state": "fail"}}
    assert not S.is_verified_triple_win(evidence).won


def test_a_gate_declared_not_applicable_or_insufficient_is_not_a_win():
    for state in ("not_applicable", "insufficient_evidence"):
        evidence = _evidence()
        evidence["hyperbolic_certificate"] = {**_gates(), "H4": {"state": state}}
        assert not S.is_verified_triple_win(evidence).won, state


def test_a_certificate_bound_to_another_proposal_is_not_a_win():
    evidence = _evidence()
    evidence["cgv_certificate"] = {
        **_gates(prefix="G", names=S.CGV_GATES),
        "proposal_sha256": "z" * 64,
    }
    verdict = S.is_verified_triple_win(evidence)
    assert not verdict.won
    assert "cgv_certificate_identity" in {check.name for check in verdict.checks if not check.ok}


def test_an_incomplete_stage3_is_not_a_win():
    evidence = _evidence()
    evidence["stage3"] = {"test_recall@10": 0.07, "test_ndcg@10": 0.04, "complete": False}
    verdict = S.is_verified_triple_win(evidence)
    assert not verdict.won
    assert "stage3_complete" in {check.name for check in verdict.checks if not check.ok}


def test_a_missing_er_share_is_not_a_win():
    evidence = _evidence()
    evidence["frozen_evaluation"] = {"hard_l1_er": 16.9}
    verdict = S.is_verified_triple_win(evidence)
    assert not verdict.won
    assert "er_shares_recorded" in {check.name for check in verdict.checks if not check.ok}


def test_no_evidence_at_all_is_not_a_win():
    verdict = S.is_verified_triple_win(None)
    assert not verdict.won
    assert verdict.reason
