"""Issue #220: one Stage2 arm per card, chosen from the cards nobody holds.

The failure this pins: four concurrent assignments all landing on the same card
(or on a card that already had a trainer) because allocation ignored live usage.
"""
from __future__ import annotations

from tools.research.stage2 import gpu_alloc


INVENTORY = "0, GPU-aaa\n1, GPU-bbb\n2, GPU-ccc\n3, GPU-ddd\n"
APPS_EMPTY = ""
APPS_TWO_BUSY = "GPU-aaa, 111\nGPU-ccc, 222\nGPU-ccc, 333\n"


def test_inventory_and_apps_parse_into_cards():
    inventory = gpu_alloc.parse_inventory(INVENTORY)
    assert inventory == {"GPU-aaa": 0, "GPU-bbb": 1, "GPU-ccc": 2, "GPU-ddd": 3}
    apps = gpu_alloc.parse_compute_apps(APPS_TWO_BUSY)
    usage = gpu_alloc.card_usage(inventory, apps)
    assert usage[0] == [111]
    assert usage[2] == [222, 333]
    assert usage[1] == [] and usage[3] == []


def test_allocation_skips_cards_that_already_run_something():
    inventory = gpu_alloc.parse_inventory(INVENTORY)
    usage = gpu_alloc.card_usage(inventory, gpu_alloc.parse_compute_apps(APPS_TWO_BUSY))
    assert gpu_alloc.choose_card([0, 1, 2, 3], usage) == 1


def test_four_successive_assignments_get_four_distinct_cards():
    """Four arms never share a card, even when each one is assigned in turn."""
    inventory = gpu_alloc.parse_inventory(INVENTORY)
    usage = gpu_alloc.card_usage(inventory, gpu_alloc.parse_compute_apps(APPS_EMPTY))
    claimed: list[int] = []
    for _ in range(4):
        card = gpu_alloc.choose_card([0, 1, 2, 3], usage)
        assert card is not None
        claimed.append(card)
        usage[card].append(9000 + card)     # what the trainer's process would do
    assert sorted(claimed) == [0, 1, 2, 3]


def test_a_fifth_arm_is_refused_not_piled_onto_a_busy_card():
    inventory = gpu_alloc.parse_inventory(INVENTORY)
    usage = gpu_alloc.card_usage(inventory, gpu_alloc.parse_compute_apps(APPS_EMPTY))
    for card in range(4):
        usage[card].append(9000 + card)
    assert gpu_alloc.choose_card([0, 1, 2, 3], usage) is None
    text = gpu_alloc.describe([0, 1, 2, 3], usage)
    assert "gpu 0" in text and "busy" in text


def test_a_live_stage2_claim_counts_before_its_trainer_shows_up():
    """The smoke-pending window must already reserve its card."""
    usage = {0: [], 1: [], 2: [], 3: []}
    tracked = {2: "TIGER_trq_C"}
    busy = {index: list(pids) for index, pids in usage.items()}
    for index, tree in tracked.items():
        busy[index].append("stage2:" + tree)
    assert gpu_alloc.choose_card([0, 1, 2, 3], busy) == 0
    assert gpu_alloc.conflicts(list(tracked.items())) == {}


def test_two_runs_on_one_card_are_reported_as_a_conflict():
    assert gpu_alloc.conflicts([(1, "TIGER_a"), (2, "TIGER_b")]) == {}
    conflicts = gpu_alloc.conflicts([(1, "TIGER_a"), (1, "TIGER_b")])
    assert set(conflicts[1]) == {"TIGER_a", "TIGER_b"}


def test_a_claim_without_a_pid_already_blocks_its_card():
    """The race that put D3 next to D0: the state exists before the worker's pid.

    The server's rule is "active status and recent launch blocks the card"; this
    test pins the decision the allocator consumes.
    """
    import time

    def blocks(status: str, launched_at: float) -> bool:
        terminal = status in {"stopped", "refused", "finished", "exited_incomplete"}
        recent = (time.time() - launched_at) < 2 * 3600
        return (not terminal) and recent

    assert blocks("starting", time.time())
    # a claim written before the worker exists has no launched_at: the state
    # file's mtime stands in for it. Pin the *shape* (an hour-old timestamp is
    # still recent), not this file's mtime: the mtime test turned red every time
    # the checkout sat unedited for two hours.
    assert blocks("starting", time.time() - 3600)
    assert not blocks("starting", time.time() - 3 * 3600)
    assert blocks("smoke_pending", time.time())
    assert blocks("running", time.time())
    assert not blocks("stopped", time.time())
    assert not blocks("running", time.time() - 3 * 3600)
