"""Launch one R4 wave: one arm per visible GPU, in the tree's own single-card mode.

Each arm's trainer honours the single-card launch contract (``GENREC_SINGLE_GPU=1``) and
scales its per-process batch to the 4-card global batch, so four arms can share the four
L40S cards without either changing a hyperparameter or forking 4 ranks per arm. The
command is the tree's documented entrypoint; nothing is reimplemented here.

    python3 tools/r4_launch_wave.py r4_01 r4_02 r4_03 r4_04
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
LAUNCH_LOG = Path("/tmp/r4_launches.json")


def visible_cards() -> list[str]:
    done = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return [line.strip() for line in done.stdout.splitlines() if line.strip()]


def already_running(arm: str) -> bool:
    done = subprocess.run(
        ["pgrep", "-af", f"TIGER_{arm}/train_rqvae.py"],
        capture_output=True, text=True,
    )
    return bool(done.stdout.strip())


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
        "NCCL_TIMEOUT": "3600",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
    })
    handle = log.open("ab")
    proc = subprocess.Popen(
        [PYTHON, str(tree / "train_rqvae.py")],
        cwd=str(tree), env=env, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        "arm": arm, "card": card, "pid": proc.pid,
        "log": str(log), "started_at": time.time(),
        "tree": str(tree),
    }


def record(entry: dict) -> None:
    payload = json.loads(LAUNCH_LOG.read_text()) if LAUNCH_LOG.is_file() else []
    payload = [e for e in payload if e["arm"] != entry["arm"]]
    payload.append(entry)
    LAUNCH_LOG.write_text(json.dumps(payload, indent=2))


def main() -> int:
    arms = sys.argv[1:]
    if not arms:
        print("usage: r4_launch_wave.py <arm> [<arm> ...]", file=sys.stderr)
        return 2
    cards = visible_cards()
    free = []
    for arm in arms:
        if already_running(arm):
            print(f"{arm}: already running, skipping")
            continue
        free.append(arm)
    if len(free) > len(cards):
        print(f"refusing: {len(free)} arms but only {len(cards)} cards", file=sys.stderr)
        return 1
    for arm, card in zip(free, cards):
        entry = launch(arm, card)
        record(entry)
        print(f"{arm}: card {card} pid {entry['pid']} log {entry['log']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
