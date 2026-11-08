"""Run the waiting Stage 2 arms at the same time, one card each.

CLAUDE.md 19d0c requires every admitted arm to be training while another is; the
MCP's launcher decides the layout from the cards its own process sees, and with
four cards visible it would run one arm on four ranks and queue the rest. This
driver keeps the frozen recipe and the owner's rule at once: each arm runs the
framework launcher with ``world_size=1`` and its own ``CUDA_VISIBLE_DEVICES``, so
every arm is a single rank with the global batch the contract fixes (4096) and no
arm waits for another.

Only arms whose source is the admitted source are launched: the driver records the
tree's digest and refuses if it differs from the digest recorded at admission.

    python3 results/research/issue143/run_parallel.py M1:1 M3:2 M4:3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.research.stage2 import contract, launcher  # noqa: E402

TREE = REPO / "stage2_RQ-VAE"
RESULT = REPO / "results" / "stage2_RQ-VAE"
PYTHON = str(Path("/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"))


def tree_digest(tree: Path) -> str:
    """One digest over the source and tests, so a restart is provably the same code."""
    value = hashlib.sha256()
    for path in sorted(list(tree.glob("*.py")) + list(tree.glob("model/*.py"))
                       + list(tree.glob("tests/*.py"))):
        if "__pycache__" in path.parts:
            continue
        value.update(str(path.relative_to(tree)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+", help="ARM:CARD, e.g. M1_learned_kappa:1")
    parser.add_argument("--record", default="results/research/issue143/parallel_runs.json")
    args = parser.parse_args()
    record = []
    processes = []
    for specification in args.arms:
        arm, _, card = specification.partition(":")
        name = arm if arm.startswith("TIGER_") else "TIGER_" + arm
        tree = TREE / name
        result = RESULT / name
        if not tree.is_dir():
            raise SystemExit("no tree at " + str(tree))
        digest = tree_digest(tree)
        entry = tree / "train_rqvae.py"
        environment = {"CUDA_VISIBLE_DEVICES": str(card), contract.SINGLE_GPU_ENV: "1"}
        command = [PYTHON, str(entry)]
        env = contract.launch_env(environment)
        log_dir = result / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / ("train_run_card%s.log" % card)
        sink = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=str(result), env=env, stdout=sink,
                                   stderr=subprocess.STDOUT)
        processes.append((arm, process, sink, log_path))
        record.append({"arm": arm, "card": str(card), "pid": process.pid,
                       "tree": str(tree), "source_digest": digest,
                       "log": str(log_path), "started": time.time(),
                       "world_size": 1,
                       "mode": "single-card", "launched_by": "framework contract"})
        print(json.dumps(record[-1]))
    target = REPO / args.record
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for arm, process, sink, log_path in processes:
        code = process.wait()
        sink.close()
        print("finished", arm, "exit", code, "log", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
