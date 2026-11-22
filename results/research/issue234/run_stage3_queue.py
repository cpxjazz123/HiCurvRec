#!/usr/bin/env python3
"""Drive the A1-A6 Stage3 queue serially through the MCP tool surface.

One rule only: never more than one live Stage3 run. The MCP server enforces that
itself (``a stage3 process is already live``), so this driver waits for the running
arm's ``test_final.json`` before calling ``stage3_start`` for the next one.

A6 is launched with ``waive_collision`` because its three-level collision 0.4118
is above the frozen 0.40 gate and the owner asked for the run anyway; the verdict
keeps the real number and ``collision_gate_waived``.

Usage (no arguments, per CLAUDE.md 9-10):
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/issue234/run_stage3_queue.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARMS = [f"TIGER_i234_A{n}" for n in range(1, 7)]
#: Arms whose three-level collision is at or above the 0.40 gate; they start waived.
WAIVED = {"TIGER_i234_A6"}
POLL_SECONDS = 60
#: One arm is ~40 min (cold import ~5 min + 150 epochs + final test); 90 min is the fence.
ARM_TIMEOUT_SECONDS = 90 * 60


def sid_of(arm: str) -> Path:
    return REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json"


def result_dir(arm: str) -> Path:
    return REPO / "results" / "stage3_T5Train" / arm


def test_final(arm: str) -> Path | None:
    hits = sorted(result_dir(arm).glob("**/test_final.json"))
    return hits[-1] if hits else None


def live_stage3() -> list[int]:
    out = subprocess.run(["pgrep", "-f", "run_stage3_param_"],
                         capture_output=True, text=True)
    pids = [int(line) for line in out.stdout.split() if line.strip().isdigit()]
    return [pid for pid in pids if pid != 0]


def call_start(arm: str) -> dict:
    """Start one arm through the same code path the MCP tool uses.

    Importing the server module is not the tool call; the tool is a plain function,
    so the driver binds to it directly and the server keeps owning the gate.
    """
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tools" / "genrec-mcp"))
    import server  # noqa: PLC0415 - the MCP implementation lives outside this script

    return server.tool_stage3_start(str(sid_of(arm)),
                                    waive_collision=arm in WAIVED)


def read_result(arm: str) -> dict | None:
    path = test_final(arm)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def wait_for(arm: str) -> dict | None:
    started = time.time()
    while time.time() - started < ARM_TIMEOUT_SECONDS:
        result = read_result(arm)
        if result is not None:
            return result
        if not live_stage3():
            # The run died without writing its test: report it, never retry silently.
            result = read_result(arm)
            if result is not None:
                return result
            raise SystemExit(f"{arm}: stage3 exited without test_final.json")
        time.sleep(POLL_SECONDS)
    raise SystemExit(f"{arm}: still running after {ARM_TIMEOUT_SECONDS}s")


def main() -> int:
    report = []
    for arm in ARMS:
        done = read_result(arm)
        if done is not None:
            report.append({"arm": arm, "reused": True, "test": done})
            print(f"[skip] {arm} already has test_final.json", flush=True)
            continue
        while live_stage3():
            time.sleep(POLL_SECONDS)
        started = call_start(arm)
        print(f"[start] {arm} -> {json.dumps({k: started.get(k) for k in ('started', 'refused', 'pid', 'variant', 'collision', 'collision_gate_waived')})}",
              flush=True)
        if not started.get("started"):
            raise SystemExit(f"{arm}: refused: {started.get('refused')}")
        result = wait_for(arm)
        print(f"[done] {arm} test_R@10={result['test_recall@10']:.6f}", flush=True)
        report.append({"arm": arm,
                       "reused": False,
                       "collision_gate_waived": started.get("collision_gate_waived", False),
                       "collision": started.get("collision"),
                       "test": result})
    out = REPO / "results" / "research" / "issue234" / "stage3_queue_report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
