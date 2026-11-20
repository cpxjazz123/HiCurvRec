"""Run Stage3 for every Stage2 arm that has a SID, one at a time, to completion.

The owner's 2026-11-20 instruction is unconditional Stage 3: no collision, prefix or
usage gate.  Stage 3 is a singleton on this host, so the queue waits for the live run to
end, launches the next arm through the same harness the MCP uses, and records every
test_final.json as it lands.  Progress goes to queue.log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STAGE2 = REPO / "results" / "stage2_RQ-VAE"
STAGE3 = REPO / "results" / "stage3_T5Train"
HARNESS = Path(__file__).resolve().parent / "run_stage3_direct.py"
LOG = Path(__file__).resolve().parent / "queue.log"
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10"
POLL_SECONDS = 60


def stamp(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def live_stage3() -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            continue
        # Match whole argument tokens: a shell command that merely mentions the
        # pattern (a monitoring grep, for instance) must not count as a live run.
        tokens = cmd.split()
        if any(token.endswith("run_stage3_direct.py") or token.endswith("train_HG-Rec.py")
               or "run_stage3_param_" in token and token.endswith(".py") for token in tokens):
            found.append(int(entry.name))
    return found


def result_file(variant: str) -> Path | None:
    """The trainer writes test_final.json under logs/<dataset>/<timestamp>/."""
    direct = STAGE3 / variant / "test_final.json"
    if direct.is_file():
        return direct
    found = sorted((STAGE3 / variant / "logs").glob("*/test_final.json"))
    found += sorted((STAGE3 / variant / "logs").glob("*/*/test_final.json"))
    return found[-1] if found else None


def finished(variant: str) -> bool:
    return result_file(variant) is not None


def main() -> None:
    arms = [d for d in sorted(STAGE2.iterdir())
            if d.is_dir() and (d / "item_sids.json").is_file()]
    pending = [d for d in arms if not finished(d.name)]
    stamp(f"queue start: {len(arms)} arms with SID, {len(pending)} pending")
    for directory in pending:
        variant = directory.name
        while live_stage3():
            time.sleep(POLL_SECONDS)
        if finished(variant):
            stamp(f"{variant}: already finished, skip")
            continue
        stamp(f"{variant}: launching")
        with (LOG.parent / f"{variant}.stdout.log").open("a") as handle:
            environment = dict(os.environ,
                               GENREC_STAGE3_SID=str(directory / "item_sids.json"),
                               GENREC_STAGE3_VARIANT=variant)
            process = subprocess.Popen(
                [PYTHON, str(HARNESS), str(directory / "item_sids.json"), variant],
                cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, env=environment)
        while process.poll() is None:
            time.sleep(POLL_SECONDS)
        result = result_file(variant)
        if result is not None:
            payload = json.loads(result.read_text())
            stamp(f"{variant}: done recall@10={payload.get('test_recall@10')}")
        else:
            stamp(f"{variant}: exited rc={process.returncode} without test_final.json")
    stamp("queue complete")


if __name__ == "__main__":
    main()
