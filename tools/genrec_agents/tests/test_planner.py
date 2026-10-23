"""The planner's tests (issue #110, section 4 and acceptance test 11).

The acceptance test the issue names is that seventy-two planner ticks with no novel
candidate and no new evidence produce ZERO new science rounds and ZERO extra training, that
the status is preserved, and that the planner says why it is blocked. It is also a
regression for the old 48-119 loop, which burned the machine by treating a tick as a round.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 -m pytest \
        tools/genrec_agents/tests/test_planner.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import planner as P  # noqa: E402
from tools.genrec_agents.hyperbolic_campaign import Campaign  # noqa: E402


def _planner(tmp_path, **kwargs):
    namespace = tmp_path / "namespace"
    campaign = Campaign(namespace=namespace)
    return P.Planner(campaign, namespace=namespace, **kwargs)


def test_seventy_two_ticks_with_no_evidence_create_no_round_and_no_training(tmp_path):
    planner = _planner(tmp_path)
    statuses = planner.run(72)
    assert len(statuses) == 72
    assert planner.state.rounds_created == 0
    assert planner.state.trainings_dispatched == 0
    assert planner.state.round == 0
    assert all(status.round_created is False for status in statuses)
    assert all(status.training_dispatched is False for status in statuses)
    assert all(status.state == P.BLOCKED for status in statuses)


def test_the_planner_preserves_its_status_and_says_why_it_is_blocked(tmp_path):
    planner = _planner(tmp_path)
    planner.run(72)
    assert planner.state.status == P.BLOCKED
    assert planner.state.blocked_reason == P.NO_WORK
    assert planner.state.blocked_since
    # and the status survives a restart
    reloaded = P.Planner(Campaign(namespace=tmp_path / "namespace"), namespace=tmp_path / "namespace")
    assert reloaded.state.ticks == 72
    assert reloaded.state.status == P.BLOCKED
    assert reloaded.state.blocked_reason == P.NO_WORK


def test_a_novel_candidate_starts_exactly_one_round(tmp_path):
    def proposals():
        return [
            {"mechanism_id": "arm_new", "proposal_sha256": "a" * 64, "family": "hyperbolic"}
        ]

    planner = _planner(tmp_path, proposals=proposals)
    status = planner.tick()
    assert status.round_created is True
    assert planner.state.rounds_created == 1
    assert planner.state.round == 1
    round_dir = tmp_path / "namespace" / "rounds" / "round_001"
    assert (round_dir / "round.json").is_file()


def test_the_planner_dispatches_no_training_without_work(tmp_path):
    planner = _planner(tmp_path)
    planner.run(5)
    assert planner.state.trainings_dispatched == 0
    assert not (tmp_path / "namespace" / "rounds").exists()


def test_a_ready_candidate_is_advanced_and_the_round_is_recorded(tmp_path):
    advanced = []

    def advance(candidate, action):
        advanced.append((candidate.mechanism_id, action))
        return {"next_action": "", "kind": "stage2"}

    namespace = tmp_path / "namespace"
    campaign = Campaign(namespace=namespace)
    campaign.add(
        P.Candidate(
            mechanism_id="arm_ready",
            proposal_sha256="b" * 64,
            family="hyperbolic",
            next_action="screen",
        )
    )
    planner = P.Planner(campaign, namespace=namespace, advance=advance)
    status = planner.tick()
    assert status.state == P.WORKING
    assert status.round_created is True
    assert status.training_dispatched is True
    assert planner.state.trainings_dispatched == 1
    assert advanced and advanced[0][1]["action"] == "screen"


def test_a_new_measurement_is_recorded_without_a_new_round(tmp_path):
    seen = []

    def measurements():
        return [{"mechanism_id": "arm_x", "er": 16.4}]

    def advance(candidate, action):
        seen.append(action)
        return {"next_action": ""}

    namespace = tmp_path / "namespace"
    campaign = Campaign(namespace=namespace)
    campaign.add(P.Candidate(mechanism_id="arm_x", proposal_sha256="c" * 64))
    planner = P.Planner(
        campaign, namespace=namespace, measurements=measurements, advance=advance
    )
    status = planner.tick()
    assert status.action == "record_measurement"
    assert status.round_created is False
    assert planner.state.rounds_created == 0
    assert seen and seen[0]["action"] == "record_measurement"


def test_the_planner_state_is_written_atomically(tmp_path):
    planner = _planner(tmp_path)
    planner.tick()
    document = json.loads((tmp_path / "namespace" / "planner" / "state.json").read_text())
    assert document["ticks"] == 1
    assert document["status"] == P.BLOCKED
