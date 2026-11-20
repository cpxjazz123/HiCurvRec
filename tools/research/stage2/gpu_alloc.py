"""Issue #220: give every Stage2 arm its own card.

The launcher used to hand each arm every visible card, so four concurrent arms
put sixteen ranks on four GPUs and per-epoch time grew 14-55x instead of ~4x.
This module owns the allocation rule: one arm per card, chosen from the cards no
other process is using.  It is pure so the rule can be tested without a GPU.

Policy (in order):
  1. a card with *any* compute process on it -- ours or another tenant's -- is busy;
  2. a card claimed by a live Stage2 run counts as busy even before its trainer
     shows up in nvidia-smi (the smoke is pending);
  3. the lowest free index wins, which makes the assignment deterministic;
  4. no free card means the launch is refused, never queued onto a busy card.
"""
from __future__ import annotations


def parse_inventory(text: str) -> dict[str, int]:
    """``nvidia-smi --query-gpu=index,uuid --format=csv,noheader`` -> uuid -> index."""
    inventory: dict[str, int] = {}
    for line in text.splitlines():
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            inventory[parts[1]] = int(parts[0])
        except ValueError:
            continue
    return inventory


def parse_compute_apps(text: str) -> dict[str, list[int]]:
    """``nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader`` per card."""
    apps: dict[str, list[int]] = {}
    for line in text.splitlines():
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        apps.setdefault(parts[0], []).append(pid)
    return apps


def card_usage(inventory: dict[str, int], apps: dict[str, list[int]]) -> dict[int, list[int]]:
    """Card index -> pids, for the cards the inventory knows about."""
    usage: dict[int, list[int]] = {index: [] for index in inventory.values()}
    for uuid, pids in apps.items():
        index = inventory.get(uuid)
        if index is not None:
            usage.setdefault(index, []).extend(pids)
    return usage


def merge_tracked(usage: dict[int, list[int]], tracked: dict[int, str]) -> dict[int, list[str]]:
    """Fold the live Stage2 claims into the pid picture, for reporting."""
    merged: dict[int, list[str]] = {index: [str(pid) for pid in pids]
                                    for index, pids in usage.items()}
    for index, tree in tracked.items():
        merged.setdefault(index, []).append("stage2:" + tree)
    return merged


def choose_card(visible: list[int], busy: dict[int, list[int]]) -> int | None:
    """Lowest visible card with no compute process and no live Stage2 claim."""
    for index in sorted(int(value) for value in visible):
        if not busy.get(index):
            return index
    return None


def describe(visible: list[int], busy: dict[int, list[int]]) -> str:
    """A refusal that names each card and what holds it."""
    parts = []
    for index in sorted(visible):
        holders = [str(item) for item in (busy.get(index) or [])]
        parts.append(f"gpu {index}: {'idle' if not holders else 'busy (' + ', '.join(holders) + ')'}")
    return "; ".join(parts)


def conflicts(pairs) -> dict[int, list[str]]:
    """Cards claimed by more than one live Stage2 run (must be empty).

    Accepts ``(card, tree)`` pairs rather than a dict, because the whole point is
    to notice the same card twice.
    """
    seen: dict[int, list[str]] = {}
    for index, tree in pairs:
        seen.setdefault(int(index), []).append(tree)
    return {index: trees for index, trees in seen.items() if len(trees) > 1}
