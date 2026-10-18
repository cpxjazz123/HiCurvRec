"""Launch R5 Stage2 arms with up to MAX_CONCURRENT parallel jobs.

Packs multiple arms onto visible GPUs via round-robin when arm count
exceeds card count. Cap is 10 concurrent Stage2 jobs (user policy).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
LAUNCH_LOG = Path("/tmp/r5_launches.json")
MAX_CONCURRENT = 10


def visible_cards():
    done = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return [ln.strip() for ln in done.stdout.splitlines() if ln.strip()]


def already_running(arm: str) -> bool:
    done = subprocess.run(
        ["pgrep", "-af", f"TIGER_{arm}/train_rqvae.py"],
        capture_output=True, text=True,
    )
    return bool(done.stdout.strip())


def count_running_stage2() -> int:
    done = subprocess.run(
        ["pgrep", "-af", "TIGER_r5_.*/train_rqvae.py"],
        capture_output=True, text=True,
    )
    lines = [ln for ln in done.stdout.splitlines() if "train_rqvae.py" in ln and "pgrep" not in ln]
    return len(lines)


def launch(arm: str, card: str) -> dict:
    tree = REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}"
    results = tree / "results"
    results.mkdir(parents=True, exist_ok=True)
    log = results / "launcher.log"
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": card,
        "GENREC_SINGLE_GPU": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
    })
    handle = log.open("ab")
    proc = subprocess.Popen(
        [PYTHON, str(tree / "train_rqvae.py")],
        cwd=str(tree), env=env, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {"arm": arm, "card": card, "pid": proc.pid, "log": str(log), "started_at": time.time()}


def main() -> int:
    arms = sys.argv[1:]
    if not arms:
        print("usage: r5_launch_wave.py <arm> ...", file=sys.stderr)
        return 2
    cards = visible_cards()
    if not cards:
        print("refusing: no visible GPUs", file=sys.stderr)
        return 1
    free = [a for a in arms if not already_running(a)]
    skipped = [a for a in arms if a not in free]
    for a in skipped:
        print(f"{a}: already running, skipping")
    running = count_running_stage2()
    slots = max(0, MAX_CONCURRENT - running)
    if slots <= 0:
        print(f"refusing: already at MAX_CONCURRENT={MAX_CONCURRENT}", file=sys.stderr)
        return 1
    if len(free) > slots:
        print(
            f"note: launching first {slots}/{len(free)} free arms "
            f"(running={running}, MAX_CONCURRENT={MAX_CONCURRENT})",
            flush=True,
        )
        free = free[:slots]
    payload = json.loads(LAUNCH_LOG.read_text()) if LAUNCH_LOG.is_file() else []
    for idx, arm in enumerate(free):
        card = cards[idx % len(cards)]
        entry = launch(arm, card)
        payload = [e for e in payload if e["arm"] != arm] + [entry]
        print(f"{arm}: card {card} pid {entry['pid']} log {entry['log']}", flush=True)
    LAUNCH_LOG.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
