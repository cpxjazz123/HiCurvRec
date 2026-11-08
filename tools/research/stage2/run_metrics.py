"""Recover one run's stream from a shared ``training_metrics.jsonl``.

The Stage 2 collector truncates the file at the first record of a run and then
appends. One run alone is therefore clean by construction. Two overlapping runs
are not: the later run truncates, the earlier run keeps appending its tail, and
the file ends up holding both -- the earlier tail first in file order, then the
later sequence. Reading such a file as "one run" silently mixes two runs.

Every record carries ``timestamp`` and ``wall_time_s`` (both from ``time.time``),
so ``timestamp - wall_time_s`` is the run's own start time. Grouping by that
value separates the runs exactly, without guessing from epoch order.

This module is pure standard library; it is used by the closing paperwork of
Issues #140-#143, where a duplicate run really did corrupt the stream.
"""
from __future__ import annotations

import json
from pathlib import Path

__all__ = ["NoCompleteRun", "run_start", "split_runs", "summarize", "complete_run",
           "write_run_stream"]


class NoCompleteRun(ValueError):
    """Raised when the file holds no single run with a full epoch sequence."""


# Two real runs are at least a process start apart (seconds). The derived start
# time wobbles by microseconds because ``timestamp`` and the wall offset are two
# ``time.time`` calls, so records are clustered instead of compared exactly.
_TOLERANCE_S = 0.5


def run_start(row: dict) -> float:
    """The run's own start time, from the row's clock and its wall offset."""
    return float(row["timestamp"]) - float(row.get("wall_time_s", 0.0))


def split_runs(path: Path | str) -> dict[float, list[dict]]:
    """Group the records of one file by the run that wrote them."""
    clusters: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            start = run_start(row)
            for cluster in clusters:
                if abs(start - cluster["center"]) <= _TOLERANCE_S:
                    cluster["total"] += start
                    cluster["count"] += 1
                    cluster["center"] = cluster["total"] / cluster["count"]
                    cluster["rows"].append(row)
                    break
            else:
                clusters.append({"center": start, "total": start, "count": 1,
                                 "rows": [row]})
    groups: dict[float, list[dict]] = {}
    for cluster in clusters:
        groups[round(cluster["center"], 2)] = sorted(
            cluster["rows"], key=lambda row: row.get("epoch", -1))
    return groups


def summarize(path: Path | str) -> dict[float, dict]:
    """Per-run counts and epoch ranges, for the audit record."""
    out = {}
    for start, rows in split_runs(path).items():
        events: dict[str, int] = {}
        for row in rows:
            events[row.get("event", "?")] = events.get(row.get("event", "?"), 0) + 1
        epochs = [row["epoch"] for row in rows if row.get("event") == "train"]
        out[start] = {"rows": len(rows), "events": events,
                      "first_epoch": min(epochs) if epochs else None,
                      "last_epoch": max(epochs) if epochs else None,
                      "contiguous": epochs == list(range(1, len(epochs) + 1))}
    return out


def complete_run(path: Path | str, epochs: int) -> list[dict]:
    """Return the rows of the one run whose train sequence is exactly 1..epochs.

    Fails closed: no complete run, or more than one, is an error. A truncated
    stream must never be passed off as a run.
    """
    candidates = []
    for start, rows in split_runs(path).items():
        train = [row["epoch"] for row in rows if row.get("event") == "train"]
        events = {row.get("event") for row in rows}
        if train == list(range(1, epochs + 1)) and {"train_start", "train_end"} <= events:
            candidates.append((start, rows))
    if not candidates:
        raise NoCompleteRun(
            "no run with a complete 1..%d train sequence and both ends in %s"
            % (epochs, path))
    if len(candidates) > 1:
        raise NoCompleteRun(
            "%d complete runs in %s; refusing to choose between them"
            % (len(candidates), path))
    return candidates[0][1]


def write_run_stream(path: Path | str, output: Path | str, epochs: int) -> dict:
    """Write the complete run's records to ``output`` and return its summary."""
    rows = complete_run(path, epochs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    start = run_start(rows[0])
    return {"run_start": start, "rows": len(rows), "output": str(output),
            "events": _counts(rows)}


def _counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("event", "?")
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> int:
    """Write one run's stream plus the audit of every run found in the file."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", help="the shared training_metrics.jsonl")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--out", help="where to write the chosen run's rows")
    parser.add_argument("--audit", help="where to write the per-run audit JSON")
    args = parser.parse_args()

    path = Path(args.metrics)
    audit = {"source": str(path), "epochs": args.epochs,
             "runs": {str(start): summary for start, summary in
                      sorted(summarize(path).items())}}
    try:
        rows = complete_run(path, args.epochs)
    except NoCompleteRun as exc:
        audit["delivered"] = None
        audit["reason"] = str(exc)
        if args.audit:
            Path(args.audit).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 1
    audit["delivered"] = {"run_start": run_start(rows[0]), "rows": len(rows),
                          "events": _counts(rows)}
    if args.out:
        write_run_stream(path, args.out, args.epochs)
        audit["delivered"]["output"] = str(args.out)
    if args.audit:
        Path(args.audit).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
