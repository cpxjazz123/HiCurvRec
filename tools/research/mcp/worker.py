"""One detached native research worker, using only internal stdin run-id IPC."""
import json
import os
import sys
import traceback

from ..workflows.pipeline import process_next, run_dir


def main():
    run_id = sys.stdin.readline().strip()
    root = run_dir(run_id)
    if not (root / "run.json").is_file():
        raise SystemExit("unknown research run")
    while not (root / "cancel.json").is_file():
        try:
            outcome = process_next(run_id)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return 1
        if outcome is None:
            break
        print(json.dumps(outcome, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
