#!/usr/bin/env python3
"""Drive one exploration arm: Stage2 (if needed) then Stage3, through the MCP servers.

Stage3 is a global singleton, so this script is run one arm at a time.  The MCP tools are
called in subprocesses -- importing a stdio server in-process blocks.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "tools" / "genrec-mcp"
POLL = 60
STAGE2_TIMEOUT = 200 * 60
STAGE3_TIMEOUT = 110 * 60

CALL = r"""
import json, sys
sys.path.insert(0, {repo!r}); sys.path.insert(0, {tools!r})
import server
print(json.dumps(server.tool_{tool}({args})))
"""


def call(tool: str, args: str) -> dict:
    code = CALL.format(repo=str(REPO), tools=str(TOOLS), tool=tool, args=args)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600)
    if done.returncode != 0:
        raise SystemExit(f"{tool}: failed: {done.stderr[-1200:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def sid_path(arm: str) -> Path:
    return REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json"


def stage2_log_epoch(arm: str) -> str:
    log = REPO / "results" / "stage2_RQ-VAE" / arm / "logs" / "train_migrated.log"
    if not log.is_file():
        return "no log"
    best = "?"
    for line in log.read_text(errors="ignore").splitlines()[-4000:]:
        i = line.find("epoch=")
        if i >= 0:
            best = line[i:i + 16]
    return best


def wait_sid(arm: str) -> None:
    started, last = time.time(), 0.0
    while time.time() - started < STAGE2_TIMEOUT:
        if sid_path(arm).is_file():
            print(f"[{arm}] item_sids.json present", flush=True)
            return
        if time.time() - last > 600:
            print(f"[{arm}] stage2 {stage2_log_epoch(arm)}", flush=True)
            last = time.time()
        time.sleep(POLL)
    raise SystemExit(f"{arm}: stage2 timed out")


def test_final(arm: str):
    hits = sorted((REPO / "results" / "stage3_T5Train" / arm).glob("**/test_final.json"))
    return hits[-1] if hits else None


def main() -> int:
    arm = sys.argv[1]
    print(f"[{arm}] stage2_start", flush=True)
    print(json.dumps(call("stage2_start", repr(str(REPO / "stage2_RQ-VAE" / arm)))), flush=True)
    wait_sid(arm)
    print(f"[{arm}] stage3_start", flush=True)
    print(json.dumps(call("stage3_start", repr(str(sid_path(arm))))), flush=True)
    started = time.time()
    while time.time() - started < STAGE3_TIMEOUT:
        path = test_final(arm)
        if path:
            result = json.loads(path.read_text(encoding="utf-8"))
            print(f"[{arm}] test_final {json.dumps(result)}", flush=True)
            out = REPO / "results" / "research" / "issue238" / f"{arm}_stage3_result.json"
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return 0
        time.sleep(POLL)
    raise SystemExit(f"{arm}: stage3 timed out")


if __name__ == "__main__":
    raise SystemExit(main())
