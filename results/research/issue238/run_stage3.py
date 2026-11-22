#!/usr/bin/env python3
"""Run Stage3 for the two exploration arms, serially, after their Stage2 finished.

Stage3 is a global singleton (one live run), so the arms are queued.  The MCP tool is
called in a subprocess because importing the stdio server in-process blocks.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARMS = ["TIGER_g237_struct_rq", "TIGER_g237_sem_rq"]
POLL = 60
TIMEOUT = 110 * 60

CALLER = r"""
import json, sys
sys.path.insert(0, {repo!r})
sys.path.insert(0, {tools!r})
import server
print(json.dumps(server.tool_stage3_start({sid!r})))
"""


def stage2_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "train_rqvae.py"], capture_output=True, text=True)
    return bool([p for p in out.stdout.split() if p.strip().isdigit()])


def stage3_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "run_stage3_param_"], capture_output=True, text=True)
    return bool([p for p in out.stdout.split() if p.strip().isdigit()])


def test_final(arm: str):
    hits = sorted((REPO / "results" / "stage3_T5Train" / arm).glob("**/test_final.json"))
    return hits[-1] if hits else None


def start(arm: str) -> dict:
    sid = REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json"
    code = CALLER.format(repo=str(REPO), tools=str(REPO / "tools" / "genrec-mcp"), sid=str(sid))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600)
    if done.returncode != 0:
        raise SystemExit(f"{arm}: tool call failed: {done.stderr[-800:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def wait(arm: str) -> dict:
    started = time.time()
    while time.time() - started < TIMEOUT:
        path = test_final(arm)
        if path:
            return json.loads(path.read_text(encoding="utf-8"))
        if not stage3_running():
            raise SystemExit(f"{arm}: stage3 exited without test_final.json")
        time.sleep(POLL)
    raise SystemExit(f"{arm}: timeout after {TIMEOUT}s")


def main() -> int:
    while stage2_running():
        print("[wait] stage2 still running", flush=True)
        time.sleep(POLL)
    report = []
    for arm in ARMS:
        while stage3_running():
            time.sleep(POLL)
        started = start(arm)
        print(f"[start] {arm} -> started={started.get('started')} "
              f"collision={started.get('collision')} refused={started.get('refused')}", flush=True)
        if not started.get("started"):
            report.append({"arm": arm, "refused": started.get("refused")})
            continue
        result = wait(arm)
        print(f"[done] {arm} test_R@10={result['test_recall@10']:.6f}", flush=True)
        report.append({"arm": arm, "collision": started.get("collision"), "test": result})
    out = REPO / "results" / "research" / "issue238" / "stage3_report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
