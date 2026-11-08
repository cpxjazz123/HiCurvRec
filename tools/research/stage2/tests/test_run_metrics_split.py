"""Regression tests for the shared-metrics-stream fault (2026-11-08).

The trainer's collector truncates ``training_metrics.jsonl`` at the first record
of a run and then appends. On 2026-11-08 a second, duplicate run of the same arm
was started while the first was still alive: the second truncated the file, and
the first kept appending its tail. The result was one file holding two runs,
which silently merged their epoch sequences.

These tests reproduce that shape with no torch and no GPU. They run with the
system interpreter:

    python3 -m tools.research.stage2.tests.test_run_metrics_split
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.stage2 import run_metrics


def _row(event: str, t0: float, wall: float, epoch: int | None = None, **extra) -> dict:
    row = {"event": event, "timestamp": t0 + wall, "wall_time_s": wall}
    if epoch is not None:
        row["epoch"] = epoch
    row.update(extra)
    return row


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_two_runs_in_one_file_are_separated() -> str:
    """Interleaved rows keep their own run; no epoch is shared between runs."""
    run_a = 1_000.0
    run_b = 2_000.0
    rows = [
        _row("train", run_b, 0.1, 1),
        _row("train", run_a, 900.1, 4),
        _row("train", run_b, 1.1, 2),
        _row("train", run_a, 901.1, 5),
    ]
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "training_metrics.jsonl"
        _write(path, rows)
        runs = run_metrics.split_runs(path)
        assert set(runs) == {run_a, run_b}, "the two runs were not separated: %r" % (sorted(runs),)
        assert [r["epoch"] for r in runs[run_a]] == [4, 5]
        assert [r["epoch"] for r in runs[run_b]] == [1, 2]
    return "rows group by their own run start, not by file order"


def test_the_complete_run_is_the_one_returned() -> str:
    """A complete 1..epochs sequence wins over a truncated tail from another run."""
    delivered = 3_000.0
    older = 4_000.0
    rows = [_row("train_start", delivered, 0.0),
            _row("train", delivered, 1.0, 1),
            _row("train", older, 800.0, 2),
            _row("train", delivered, 2.0, 2),
            _row("train", older, 801.0, 3),
            _row("train_end", delivered, 3.0)]
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "training_metrics.jsonl"
        _write(path, rows)
        chosen = run_metrics.complete_run(path, epochs=2)
        assert all(row["timestamp"] - row["wall_time_s"] == delivered for row in chosen), \
            "the truncated run's rows leaked into the delivered stream"
        assert [r["epoch"] for r in chosen if r["event"] == "train"] == [1, 2]
    return "the run with a full epoch sequence and both ends is the delivered one"


def test_a_truncated_stream_is_not_reported_complete() -> str:
    """Two partial runs must raise, never merge into a plausible 1..epochs."""
    rows = [_row("train_start", 1_000.0, 0.0),
            _row("train", 1_000.0, 1.0, 1),
            _row("train", 2_000.0, 500.0, 2),
            _row("train", 2_000.0, 501.0, 3),
            _row("train_end", 2_000.0, 502.0)]
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "training_metrics.jsonl"
        _write(path, rows)
        try:
            run_metrics.complete_run(path, epochs=3)
        except run_metrics.NoCompleteRun as exc:
            assert "no run" in str(exc).lower()
        else:
            raise AssertionError("two partial runs were silently merged into one stream")
    return "an incomplete stream fails closed instead of inventing a run"


def test_clock_jitter_does_not_split_one_run() -> str:
    """One run's start time reads a few microseconds apart; it stays one run.

    ``timestamp`` and the wall offset come from two ``time.time`` calls, so the
    derived start wobbles by microseconds and can straddle a rounding boundary.
    """
    rows = [_row("train", 1_000.0, 0.0, 1), _row("train", 1_000.0, 1.0, 2)]
    rows[0]["timestamp"] += 0.004
    rows[1]["timestamp"] += 0.006
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "training_metrics.jsonl"
        _write(path, rows)
        runs = run_metrics.split_runs(path)
        assert len(runs) == 1, "clock jitter split one run into %d" % len(runs)
        assert [r["epoch"] for r in next(iter(runs.values()))] == [1, 2]
    return "sub-millisecond clock jitter is one run, not two"


def main() -> int:
    failures = 0
    for test in (test_two_runs_in_one_file_are_separated,
                 test_the_complete_run_is_the_one_returned,
                 test_a_truncated_stream_is_not_reported_complete,
                 test_clock_jitter_does_not_split_one_run):
        try:
            print("PASS " + test.__name__ + ": " + test(), flush=True)
        except AssertionError as exc:
            failures += 1
            print("FAIL " + test.__name__ + ": " + str(exc), flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("ERROR " + test.__name__ + ": " + type(exc).__name__ + ": " + str(exc),
                  flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
