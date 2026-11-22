"""Owner rule (2026-11-22): Stage2 always runs four-card DDP, and Stage2 runs serially.

Two properties, both enforced at the only launch path (`launcher.launch`), because the candidate
trees must not be able to choose either:

* ``world_size`` is the frozen four; a host that cannot give four cards fails closed rather than
  silently training one rank with the four-rank batch;
* a second Stage2 launch while one is live is refused, so two arms never share the host -- the
  concurrency that §19d0b once allowed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tools.research.stage2 import launcher  # noqa: E402


def test_world_size_is_frozen_at_four():
    from tools.research.stage2 import contract
    assert contract.WORLD_SIZE == 4
    assert launcher.required_world_size() == 4


def test_a_single_card_launch_is_refused():
    with pytest.raises(launcher.LaunchRefused):
        launcher.assert_four_cards(visible=1)


def test_a_four_card_host_is_accepted():
    launcher.assert_four_cards(visible=4)


def test_a_second_live_run_is_refused(tmp_path):
    lock = tmp_path / "stage2.lock"
    launcher.acquire_serial(lock, pid=1111, alive=lambda pid: True)
    with pytest.raises(launcher.LaunchRefused):
        launcher.acquire_serial(lock, pid=2222, alive=lambda pid: True)


def test_a_stale_lock_is_replaced(tmp_path):
    lock = tmp_path / "stage2.lock"
    launcher.acquire_serial(lock, pid=1111, alive=lambda pid: False)
    launcher.acquire_serial(lock, pid=2222, alive=lambda pid: pid == 2222)
    assert str(2222) in lock.read_text()


def test_releasing_the_lock_allows_the_next_run(tmp_path):
    lock = tmp_path / "stage2.lock"
    launcher.acquire_serial(lock, pid=1111, alive=lambda pid: True)
    launcher.release_serial(lock, pid=1111)
    launcher.acquire_serial(lock, pid=2222, alive=lambda pid: pid == 2222)
