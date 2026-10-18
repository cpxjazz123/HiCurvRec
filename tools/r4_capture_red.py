"""Capture the RED phase for every R4 arm against the CANONICAL Resample tree.

Every arm's test file is run with ``R4_TEST_TREE`` pointing at
``stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE``, so each failure is a measurement of the
unchanged parent rather than of a partially written arm. The output of each run is
written to that arm's ``tdd_red_green.md`` under a ``## RED`` heading.

    python3 tools/r4_capture_red.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
CANONICAL = REPO_ROOT / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
ARMS = [f"r4_{n:02d}" for n in range(1, 11)]
MECHANISM_MARKERS = ("not implemented", "is not implemented", "not applied", "no-op")


def run(arm: str) -> tuple[int, str, str]:
    script = REPO_ROOT / "results/stage2_RQ-VAE" / arm / f"test_{arm}.py"
    env = dict(os.environ)
    env["R4_TEST_TREE"] = str(CANONICAL)
    done = subprocess.run(
        [PYTHON, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
        env=env,
    )
    command = (
        f"R4_TEST_TREE={CANONICAL} {PYTHON} results/stage2_RQ-VAE/{arm}/test_{arm}.py"
    )
    return done.returncode, done.stdout + done.stderr, command


def write_red(arm: str, code: int, output: str, command: str) -> None:
    path = REPO_ROOT / "results/stage2_RQ-VAE" / arm / "tdd_red_green.md"
    failures = [
        line for line in output.splitlines() if line.startswith(("FAIL:", "ERROR:"))
    ]
    body = [
        f"# {arm.upper()} TDD record",
        "",
        "## RED",
        "",
        f"Command (run against the unchanged canonical Resample parent):",
        "",
        f"    {command}",
        "",
        f"Exit code: {code}",
        "",
        "```",
        output.rstrip(),
        "```",
        "",
        f"Behavioural failures observed on the parent: {len(failures)}",
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> int:
    problems = []
    for arm in ARMS:
        code, output, command = run(arm)
        write_red(arm, code, output, command)
        failures = [
            line for line in output.splitlines() if line.startswith(("FAIL:", "ERROR:"))
        ]
        mechanism_failures = [
            line for line in failures if any(marker in line for marker in MECHANISM_MARKERS)
        ]
        marker = any(marker in line for line in failures for marker in ("ImportError", "AttributeError"))
        status = "RED OK" if code != 0 and not marker else "RED PROBLEM"
        if status != "RED OK":
            problems.append(arm)
        print(f"{arm}: exit={code} {status} | {len(failures)} failures, "
              f"{len(mechanism_failures)} mechanism-specific", flush=True)
        for line in failures[:4]:
            print(f"    {line[:150]}", flush=True)
    print("\nPROBLEM ARMS:", problems if problems else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
