"""Read a run's metrics stream and report its three-code collision against the Stage3 gate.

The guard keeps every codeword alive; it cannot tell whether the identifier still carries
information. That is what `stage3_start` decides (collision < 0.40, CLAUDE.md 89), but the
decision arrives after the whole budget has been spent. This watcher reads the number the
trainer already records and reports it while the run is still going.

    python tools/research/stage2/collapse_watch.py <metrics.jsonl> [--threshold 0.40]
                                                  [--stop <pid>]

Exit codes: 0 the last recorded collision is below the gate, 2 it is at or above it,
1 no collision record or unreadable file. `--stop` is opt-in: it terminates the given
process when the gate is crossed, which is a policy choice (Stage2 metrics are diagnostic
in CLAUDE.md 19), so it is off unless the caller asks for it.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

#: The Stage3 admission constant (CLAUDE.md 89).
GATE = 0.40


def last_collision(path: Path) -> dict | None:
    if not path.is_file():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "collision" in record:
            latest = record
    return latest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--threshold", type=float, default=GATE)
    parser.add_argument("--stop", type=int, default=None,
                        help="terminate this pid when the gate is crossed")
    args = parser.parse_args(argv)

    record = last_collision(args.metrics)
    if record is None:
        print("no collision record in " + str(args.metrics))
        return 1
    collision = float(record["collision"])
    verdict = "REFUSED at the stage3 gate" if collision >= args.threshold else "allowed"
    print(f"epoch {record.get('epoch')}: collision {collision:.6f} ({verdict}, "
          f"gate {args.threshold})")
    if collision >= args.threshold:
        if args.stop is not None:
            try:
                os.kill(args.stop, signal.SIGTERM)
                print("stopped " + str(args.stop))
            except ProcessLookupError:
                print("process " + str(args.stop) + " is gone")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
