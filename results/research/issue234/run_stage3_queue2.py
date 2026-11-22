#!/usr/bin/env python3
"""Drive the A4-A6 Stage3 queue serially through the MCP tool surface.

Why this is a separate script from `run_stage3_queue.py`: importing
`tools/genrec-mcp/server.py` in-process blocks (the module is an MCP stdio server),
which is exactly how the first driver stalled between A2 and A3.  Here the tool call
happens in a *subprocess* that imports the server, calls `tool_stage3_start` once, and
prints the result as JSON; the driver only waits on files.

One rule: never more than one live Stage3 run. The MCP server enforces that itself,
so the driver waits for the running arm's `test_final.json` before starting the next.

A6 is launched with ``waive_collision`` because its three-level collision 0.4118 is
above the frozen 0.40 gate and the owner asked for the run anyway; the verdict keeps
the real number and ``collision_gate_waived``.

Usage (no arguments, per CLAUDE.md 9-10):
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/issue234/run_stage3_queue2.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARMS = [f"TIGER_i234_A{n}" for n in range(4, 7)]
WAIVED = {"TIGER_i234_A6"}
POLL_SECONDS = 60
ARM_TIMEOUT_SECONDS = 100 * 60

CALLER = r"""
import json, sys
sys.path.insert(0, {repo!r})
sys.path.insert(0, {tools!r})
import server
print(json.dumps(server.tool_stage3_start({sid!r}, waive_collision={waive})))
"""


def sid_of(arm: str) -> Path:
    return REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json"


def test_final(arm: str) -> Path | None:
    hits = sorted((REPO / "results" / "stage3_T5Train" / arm).glob("**/test_final.json"))
    return hits[-1] if hits else None


def live_stage3() -> list[int]:
    out = subprocess.run(["pgrep", "-f", "run_stage3_param_"],
                         capture_output=True, text=True)
    return [int(line) for line in out.stdout.split() if line.strip().isdigit()]


def call_start(arm: str) -> dict:
    """Call the MCP tool in its own process and return its JSON verdict."""
    code = CALLER.format(repo=str(REPO), tools=str(REPO / "tools" / "genrec-mcp"),
                         sid=str(sid_of(arm)), waive=str(arm in WAIVED))
    done = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=3600)
    if done.returncode != 0:
        raise SystemExit(f"{arm}: tool call failed: {done.stderr[-800:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def read_result(arm: str) -> dict | None:
    path = test_final(arm)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def wait_for(arm: str) -> dict:
    started = time.time()
    while time.time() - started < ARM_TIMEOUT_SECONDS:
        result = read_result(arm)
        if result is not None:
            return result
        if not live_stage3():
            result = read_result(arm)
            if result is not None:
                return result
            raise SystemExit(f"{arm}: stage3 exited without test_final.json")
        time.sleep(POLL_SECONDS)
    raise SystemExit(f"{arm}: still running after {ARM_TIMEOUT_SECONDS}s")


def main() -> int:
    report = []
    for arm in ARMS:
        while live_stage3():
            time.sleep(POLL_SECONDS)
        started = call_start(arm)
        print(f"[start] {arm} -> started={started.get('started')} pid={started.get('pid')} "
              f"waived={started.get('collision_gate_waived')} refused={started.get('refused')}",
              flush=True)
        if not started.get("started"):
            raise SystemExit(f"{arm}: refused: {started.get('refused')}")
        result = wait_for(arm)
        print(f"[done] {arm} test_R@10={result['test_recall@10']:.6f}", flush=True)
        report.append({"arm": arm,
                       "collision_gate_waived": started.get("collision_gate_waived", False),
                       "collision": started.get("collision"),
                       "test": result})
    out = REPO / "results" / "research" / "issue234" / "stage3_queue2_report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
