"""The Stage-3 metrics recovered from the per-rank shards.

A Stage-3 run writes one shard per rank and then merges them into ``test_final.json``. A
run that dies between those two steps leaves a complete measurement on disk that nothing
records: the shards hold every evaluated row's target rank, and the metric is a function of
those ranks.

The convention is not guessed. It is derived: a hit is a rank in ``[0, 10)``, the
denominator is every evaluated row, and the NDCG contribution of a hit at zero-indexed rank
``r`` is ``1 / log2(r + 2)``. On every archived arm that has both the shards and the final,
this reproduces the final's numbers exactly -- the test in this repository asserts it.

Recovering the metric from the shards is a READING of an artifact that exists, not a second
evaluation: no model is run, no label is read anew.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The number of ranks that count as a hit.
HIT_RANKS = 10

#: The number of ranks a Stage-3 run writes.
EXPECTED_SHARDS = 4


def shard_ranks(shard_dir: Path) -> List[Optional[int]]:
    """Every evaluated row's target rank, from the per-rank shards."""
    ranks: List[Optional[int]] = []
    for path in sorted(Path(shard_dir).glob("rank*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for record in document.get("per_user") or ():
            ranks.append(record.get("target_rank"))
    return ranks


def metrics_from_ranks(ranks: List[Optional[int]]) -> Dict[str, Any]:
    """The Recall@10 and NDCG@10 of a set of target ranks."""
    total = len(ranks)
    scored = [rank for rank in ranks if rank is not None and rank >= 0]
    hits = sum(1 for rank in scored if rank < HIT_RANKS)
    gain = sum(1.0 / math.log2(rank + 2) for rank in scored if rank < HIT_RANKS)
    return {
        "rows": total,
        "scored_rows": len(scored),
        "unscored_rows": total - len(scored),
        "test_recall@10": hits / total if total else 0.0,
        "test_ndcg@10": gain / total if total else 0.0,
    }


def metrics_from_shards(shard_dir: Path) -> Optional[Dict[str, Any]]:
    """The metrics of a Stage-3 run, from its shards, or None when they are incomplete."""
    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        return None
    shards = sorted(shard_dir.glob("rank*.json"))
    if len(shards) < EXPECTED_SHARDS:
        return None
    ranks = shard_ranks(shard_dir)
    if not ranks:
        return None
    result = metrics_from_ranks(ranks)
    result["shards"] = [path.name for path in shards]
    result["source"] = "the per-rank shards"
    return result


def run_directories(arm_dir: Path) -> List[Path]:
    """The Stage-3 run directories of an arm, oldest first."""
    root = Path(arm_dir) / "logs" / "Amazon_2023_Instruments"
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def latest_complete(arm_dir: Path) -> Optional[Dict[str, Any]]:
    """The metrics of an arm's most recent run whose shards are all there."""
    for run_dir in reversed(run_directories(arm_dir)):
        result = metrics_from_shards(run_dir / "_test_shards")
        if result is not None:
            result["run_dir"] = str(run_dir)
            return result
    return None


def final_of(run_dir: Path) -> Optional[Dict[str, Any]]:
    """The official final of a Stage-3 run, when it was written."""
    final = Path(run_dir) / "test_final.json"
    if not final.is_file():
        return None
    try:
        return json.loads(final.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


__all__ = [
    "EXPECTED_SHARDS",
    "HIT_RANKS",
    "final_of",
    "latest_complete",
    "metrics_from_ranks",
    "metrics_from_shards",
    "run_directories",
    "shard_ranks",
]
