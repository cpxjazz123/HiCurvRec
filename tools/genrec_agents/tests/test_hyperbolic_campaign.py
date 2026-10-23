"""The new state machine's tests (issue #110, sections 3 and 4).

The machine has to make the illegal impossible: a candidate cannot skip the mathematics
review, cannot be launched without the geometry and the CGV certificates, and cannot run a
second official experiment of the same kind under the same identity. It also has to
distinguish progress from a stop: a blocked candidate records what it needs, and the counts
keep the kinds apart.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_hyperbolic_campaign.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents.hyperbolic_campaign import (  # noqa: E402
    Campaign,
    Candidate,
    TransitionError,
    experiment_id,
)


def _campaign(tmp_path) -> Campaign:
    return Campaign(namespace=tmp_path / "runs")


def _candidate(mechanism_id: str = "arm_one", **overrides) -> Candidate:
    fields = {
        "mechanism_id": mechanism_id,
        "proposal_sha256": "p" * 64,
        "family": "hyperbolic_history",
    }
    fields.update(overrides)
    return Candidate(**fields)


def test_a_candidate_moves_through_the_flow(tmp_path):
    campaign = _campaign(tmp_path)
    campaign.add(_candidate())
    assert campaign.candidate("arm_one").state == "proposed"
    for to_state in ("reviewed", "implemented", "screened", "sealed", "stage2_reserved"):
        campaign.transition("arm_one", to_state)
    assert campaign.candidate("arm_one").state == "stage2_reserved"


def test_an_illegal_transition_is_refused(tmp_path):
    campaign = _campaign(tmp_path)
    campaign.add(_candidate())
    with pytest.raises(TransitionError):
        campaign.transition("arm_one", "stage2_reserved")
    with pytest.raises(TransitionError):
        campaign.transition("arm_one", "analysed")


def test_a_second_official_run_of_the_same_mechanism_is_refused(tmp_path):
    campaign = _campaign(tmp_path)
    campaign.add(_candidate())
    campaign.transition("arm_one", "reviewed")
    campaign.transition("arm_one", "implemented")
    campaign.transition("arm_one", "screened")
    campaign.transition("arm_one", "sealed")
    identity = campaign.reserve_experiment(
        "arm_one", "stage2", implementation_sha256="i" * 64, frozen_protocol_hash="f" * 64
    )
    assert identity
    with pytest.raises(TransitionError) as raised:
        campaign.reserve_experiment(
            "arm_one", "stage2", implementation_sha256="i" * 64, frozen_protocol_hash="f" * 64
        )
    assert "already reserved" in str(raised.value)


def test_an_experiment_without_the_identity_cannot_be_reserved(tmp_path):
    campaign = _campaign(tmp_path)
    campaign.add(_candidate())
    with pytest.raises(TransitionError):
        campaign.reserve_experiment("arm_one", "stage2")


def test_the_identity_is_the_tuple_the_issue_names(tmp_path):
    """A renamed proposal with the same mathematics is the same experiment."""
    first = experiment_id(
        proposal_sha256="p" * 64,
        implementation_sha256="i" * 64,
        frozen_protocol_hash="f" * 64,
        seed=42,
        kind="stage2",
    )
    second = experiment_id(
        proposal_sha256="p" * 64,
        implementation_sha256="i" * 64,
        frozen_protocol_hash="f" * 64,
        seed=42,
        kind="stage2",
    )
    assert first == second
    other_seed = experiment_id(
        proposal_sha256="p" * 64,
        implementation_sha256="i" * 64,
        frozen_protocol_hash="f" * 64,
        seed=43,
        kind="stage2",
    )
    assert other_seed != first


def test_a_blocked_candidate_records_what_it_needs(tmp_path):
    campaign = _campaign(tmp_path)
    campaign.add(_candidate())
    campaign.transition(
        "arm_one",
        "blocked_literature",
        reason="no verified source is reachable",
        next_action="configure an academic provider or reuse a verified cache",
    )
    candidate = campaign.candidate("arm_one")
    assert candidate.is_blocked
    assert "no verified source" in candidate.blocked_reason
    assert "provider" in candidate.next_action
    summary = campaign.summary()
    assert summary["blocked"][0]["mechanism_id"] == "arm_one"


def test_the_state_survives_a_restart(tmp_path):
    campaign = _campaign(tmp_path)
    campaign.add(_candidate())
    campaign.transition("arm_one", "reviewed")
    again = _campaign(tmp_path)
    assert again.candidate("arm_one").state == "reviewed"
    assert again.candidate("arm_one").history


def test_the_counts_keep_the_kinds_of_progress_apart(tmp_path):
    campaign = _campaign(tmp_path)
    campaign.add(_candidate("a"))
    campaign.add(_candidate("b"))
    campaign.transition("a", "reviewed")
    campaign.transition("b", "rejected", reason="prior art")
    summary = campaign.summary()
    assert summary["candidates"] == 2
    assert summary["states"]["reviewed"] == 1
    assert summary["states"]["rejected"] == 1
    assert summary["blocked"] == []
