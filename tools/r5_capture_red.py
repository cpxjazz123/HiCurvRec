"""Capture RED for every R5 arm against pinned parent TIGER_r4_08 (#58)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
PARENT = REPO_ROOT / "stage2_RQ-VAE/TIGER_r4_08"
ARMS = [f"r5_{n:02d}" for n in range(1, 11)]


def run(arm: str):
    script = REPO_ROOT / "results/stage2_RQ-VAE" / arm / f"test_{arm}.py"
    env = dict(os.environ)
    env["R5_TEST_TREE"] = str(PARENT)
    done = subprocess.run(
        [PYTHON, str(script)], cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=7200, env=env,
    )
    command = f"R5_TEST_TREE={PARENT} {PYTHON} results/stage2_RQ-VAE/{arm}/test_{arm}.py"
    return done.returncode, done.stdout + done.stderr, command


def main() -> int:
    problems = []
    for arm in ARMS:
        code, output, command = run(arm)
        path = REPO_ROOT / "results/stage2_RQ-VAE" / arm / "tdd_red_green.md"
        failures = [ln for ln in output.splitlines() if ln.startswith(("FAIL:", "ERROR:"))]
        body = [
            f"# {arm.upper()} TDD record", "", "## RED", "",
            "Command (run against the pinned R4-08 parent):", "",
            f"    {command}", "", f"Exit code: {code}", "", "```", output.rstrip(), "```", "",
            f"Behavioural failures observed on the parent: {len(failures)}", "",
        ]
        path.write_text("\n".join(body), encoding="utf-8")
        bad_import = any("ImportError" in ln or "AttributeError" in ln for ln in failures)
        status = "RED OK" if code != 0 and not bad_import else "RED PROBLEM"
        if status != "RED OK":
            problems.append(arm)
        print(f"{arm}: exit={code} {status} | {len(failures)} failures", flush=True)
        for ln in failures[:4]:
            print(f"    {ln[:160]}", flush=True)
    print("\nPROBLEM ARMS:", problems if problems else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
