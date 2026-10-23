"""The #109 lesson checks' tests (issue #110, section 0).

The eight lessons have to be re-checked, not remembered, and the checks have to be honest
about what they found: the previous campaign's proposals carried no CGV contract, its
mechanisms were Euclidean under hyperbolic names, and its success decision was made in
three places. The tests assert that the checks find those facts, so a later campaign
cannot quietly reintroduce them.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/harness2_genrec/tests/test_forensics_109.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.harness2_genrec import forensics_109 as F  # noqa: E402


def test_every_lesson_has_a_check():
    assert len(F.CHECKS) == 8
    report = F.run_all()
    assert len(report["checks"]) == 8
    for check in report["checks"]:
        assert check["verdict"] in ("ok", "gap", "clean")
        assert check["note"]


def test_the_false_progress_is_counted():
    """A portfolio file without an implementation is a proposal, not an experiment."""
    check = F.check_false_progress()
    assert check["portfolio_files_without_an_implementation"] > 50
    assert check["rounds_that_adopted"] > 50


def test_the_repeated_identities_are_counted():
    check = F.check_repeated_identities()
    assert check["proposals"] > 50
    assert check["names_repeated"] > 5


def test_the_lost_cgv_contract_is_found():
    """The previous campaign never carried the caller's contract."""
    check = F.check_cgv_contract()
    assert check["with_a_contract"] == 0
    assert check["verdict"] == "gap"


def test_the_euclidean_framing_is_found():
    """The previous campaign's mechanisms were Euclidean under hyperbolic names."""
    check = F.check_euclidean_framing()
    assert check["prior_with_a_hyperbolic_operator"] == 0
    assert check["prior_euclidean_only"] > 50
    # the new campaign has no arms yet, so there is no live Euclidean defect
    assert check["new_proposals"] == 0
    assert check["verdict"] == "ok"


def test_the_split_success_predicate_is_found():
    """Every success decision now asks the one shared predicate."""
    check = F.check_split_success_predicate()
    assert check["shared_predicate_modules"], "the shared predicate module is missing"
    assert check["inline_success_decisions_in_the_controller"] == 0
    assert check["verdict"] == "ok"


def test_the_repeated_stage3_and_work_checks_are_run():
    assert F.check_repeated_stage3()["shard_recovery_present"] is True
    assert F.check_repeated_work()["tree_recreations_in_the_candidate_runner"] >= 1
    surrogates = F.check_surrogates()
    assert surrogates["with_a_history_gain"] > 0
    assert surrogates["with_a_d2"] > 0
    assert surrogates["with_both_er_shares"] > 0
