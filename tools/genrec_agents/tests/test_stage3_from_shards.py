"""The shard-based recovery of the Stage-3 metrics (issue #109).

A Stage-3 run that dies between writing its shards and merging them leaves a complete
measurement that nothing records. The recovery reads those shards, and the test proves the
convention is the official one by reproducing the archived finals exactly.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_stage3_from_shards.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import stage3_from_shards as S3  # noqa: E402


def _shards(tmp_path: Path, ranks, shards: int = 4) -> Path:
    """A shard directory with the given ranks spread over the shards."""
    shard_dir = tmp_path / "_test_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    per_shard = (len(ranks) + shards - 1) // shards
    for index in range(shards):
        chunk = ranks[index * per_shard: (index + 1) * per_shard]
        (shard_dir / f"rank{index}.json").write_text(
            json.dumps({"per_user": [{"target_rank": rank} for rank in chunk]}),
            encoding="utf-8",
        )
    return shard_dir


def test_the_hits_and_the_denominator():
    """A hit is a rank in [0, 10) and the denominator is every evaluated row."""
    ranks = [0, 3, 9, 10, 20, None, -1]
    result = S3.metrics_from_ranks(ranks)
    assert result["rows"] == 7
    assert result["scored_rows"] == 5
    assert result["unscored_rows"] == 2
    assert result["test_recall@10"] == pytest.approx(3 / 7)
    # the NDCG contribution of a zero-indexed rank r is 1 / log2(r + 2)
    import math

    expected = sum(1.0 / math.log2(rank + 2) for rank in (0, 3, 9)) / 7
    assert result["test_ndcg@10"] == pytest.approx(expected)


def test_incomplete_shards_are_not_measured(tmp_path):
    """Three shards are not a measurement."""
    shard_dir = _shards(tmp_path, list(range(100)), shards=3)
    assert S3.metrics_from_shards(shard_dir) is None


def test_the_convention_reproduces_the_archived_finals():
    """The convention is the official one: it reproduces every archived final exactly."""
    base = REPO_ROOT / "results/stage3_T5Train"
    checked = 0
    for arm_dir in sorted(base.glob("TIGER_c107_*")):
        for run_dir in S3.run_directories(arm_dir):
            final = S3.final_of(run_dir)
            if final is None:
                continue
            recovered = S3.metrics_from_shards(run_dir / "_test_shards")
            if recovered is None:
                continue
            assert recovered["test_recall@10"] == pytest.approx(
                final["test_recall@10"], abs=1e-12
            )
            assert recovered["test_ndcg@10"] == pytest.approx(
                final["test_ndcg@10"], abs=1e-12
            )
            checked += 1
    assert checked >= 1, "no archived run had both the shards and the final to check against"


def test_a_run_without_a_final_is_still_measured(tmp_path):
    """The run that died between the shards and the merge is still a measurement."""
    shard_dir = _shards(tmp_path, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 60])
    run_dir = tmp_path / "Oct-22-2026_00-00-00"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "_test_shards").mkdir(exist_ok=True)
    for path in shard_dir.glob("rank*.json"):
        (run_dir / "_test_shards" / path.name).write_text(path.read_text(), encoding="utf-8")
    assert S3.final_of(run_dir) is None
    result = S3.metrics_from_shards(run_dir / "_test_shards")
    assert result is not None
    assert result["test_recall@10"] == pytest.approx(10 / 12)
    assert result["source"] == "the per-rank shards"
