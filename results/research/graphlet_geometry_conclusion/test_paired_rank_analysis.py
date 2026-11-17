"""Regression tests for the paired per-user analysis behind the geometry report.

The report's comparisons are only as good as two pieces of code: the mapping
from a shard's local row to the frozen global test row, and the rebuild of
recall/ndcg from the recorded `target_rank`.  Both are checked here on a
synthetic run, including the rejection of a run whose rebuilt metric disagrees
with its own `test_final.json`.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = importlib.util.spec_from_file_location(
    "paired_rank_analysis", HERE / "paired_rank_analysis.py")
analysis = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(analysis)


def write_run(root: Path, arm: str, stamp: str, world: int, ranks: list,
              recorded: dict) -> None:
    """A minimal run directory: one shard per rank plus the recorded metric."""
    run_dir = root / arm / "logs" / "dataset" / stamp
    shard_dir = run_dir / "_test_shards"
    shard_dir.mkdir(parents=True)
    for rank in range(world):
        rows = [{"local_row_index": i, "target_rank": ranks[rank + i * world]}
                for i in range(len(ranks) // world)]
        payload = {"rank": rank, "n_eval": len(rows),
                   "recall_sums": {"10": 0.0}, "ndcg_sums": {"10": 0.0},
                   "per_user": rows}
        (shard_dir / f"rank{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "test_final.json").write_text(json.dumps(recorded), encoding="utf-8")


def test_local_rows_map_to_the_frozen_global_index() -> None:
    """`rank + local_row_index * world_size` is the row identity the report assumes."""
    directory = Path(__file__).resolve().parent / "_test_tmp_identity"
    if directory.exists():
        raise SystemExit(f"stale test directory: {directory}")
    ranks = [0, 10, 2, None, 7, 1]          # global rows 0..5
    try:
        write_run(directory, "arm", "stamp", 2, ranks, {"n_eval": len(ranks)})
        loaded = analysis.load_run(str(directory / "arm" / "logs" / "dataset" / "stamp"))
        assert loaded["world"] == 2
        assert loaded["per_row"] == {index: value for index, value in enumerate(ranks)}
    finally:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)


def test_rebuild_matches_the_recorded_metric_and_rejects_a_mismatch() -> None:
    ranks = [0, 10, 2, None, 7, 1]
    n, recall, ndcg = analysis.rebuild_metrics({i: r for i, r in enumerate(ranks)})
    assert n == 6
    assert recall[10] == 4 / 6 and recall[5] == 3 / 6
    # The standard NDCG: 1/log2(rank+2) for each rank inside the cut.
    expected = (1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4) + 1.0 / math.log2(9)) / 6
    assert abs(ndcg[10] - expected) < 1e-12

    directory = Path(__file__).resolve().parent / "_test_tmp_rebuild"
    if directory.exists():
        raise SystemExit(f"stale test directory: {directory}")
    try:
        write_run(directory, "good", "stamp", 1, ranks,
                  {"n_eval": 6, "test_recall@10": recall[10], "test_recall@5": recall[5],
                   "test_ndcg@10": ndcg[10], "test_ndcg@5": ndcg[5]})
        write_run(directory, "bad", "stamp", 1, ranks,
                  {"n_eval": 6, "test_recall@10": 0.5, "test_recall@5": recall[5],
                   "test_ndcg@10": ndcg[10], "test_ndcg@5": ndcg[5]})
        saved = analysis.STAGE3
        analysis.STAGE3 = str(directory)
        try:
            runs, rejected = analysis.collect()
        finally:
            analysis.STAGE3 = saved
        assert set(runs) == {"good"}
        assert any(entry["arm"] == "bad" and entry["why"] == "rebuild mismatch"
                   for entry in rejected)
        assert runs["good"]["n"] == 6
    finally:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)


def test_arm_name_is_read_above_the_logs_directory() -> None:
    path = str(Path("/x/results/stage3_T5Train/TIGER_example/logs/ds/stamp"))
    assert analysis.arm_of(path) == "TIGER_example"
