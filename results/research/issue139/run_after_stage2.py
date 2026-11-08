"""After the five Stage 2 runs finish: record layers, measure distortion, re-report.

The goal has three acceptance criteria per mechanism, and the middle one -- the
structural distortion of the SID, measured on one footing -- only becomes
possible once all five identifiers exist. This script waits for them, records
the layered metrics for each, runs the shared distortion measurement over all
five at once, and regenerates the report, so that the numbers are produced by the
same code every time instead of by hand.

It starts nothing and changes no run: every step reads what the runs wrote.

Usage:
    nohup python3 results/research/issue139/run_after_stage2.py > settle.log 2>&1 &
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARMS = ("F0", "G", "E", "A", "L")
PY = sys.executable


def log(message: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def metrics_path(arm: str) -> Path:
    return REPO / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm) / "logs" / "training_metrics.jsonl"


#: When this script started. Only a run that began after it counts: the arms
#: carry four-card runs from earlier today whose final event is on disk, and a
#: check that accepted those would report a fresh run as finished before it began.
SCRIPT_START = time.time()


def training_now(arm: str) -> bool:
    """Whether a trainer for this arm's tree is running right now."""
    done = subprocess.run(["pgrep", "-f", "TIGER_mixed_rq_%s/train_rqvae.py" % arm],
                          capture_output=True, text=True)
    return bool((done.stdout or "").strip())


def complete(arm: str) -> bool:
    """Whether this arm has an identifier and is not being trained any more.

    The metrics stream cannot answer this: a cancelled run truncates it, so the
    records of a finished run disappear (F0's did), and the arms carry four-card
    runs from earlier today whose final event is still there. What is true at
    this moment is simpler -- the identifier exists and no trainer for that tree
    is running -- and it is what the settlement actually needs.
    """
    sid = REPO / "results" / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm) / "item_sids.json"
    if not sid.is_file():
        return False
    return not training_now(arm)


def record_layers(arm: str) -> str:
    sid = REPO / "results" / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm) / "item_sids.json"
    if not sid.is_file():
        return "no SID"
    done = subprocess.run(
        [PY, "-c",
         "import sys; sys.path.insert(0, %r); from pathlib import Path;"
         "from tools.research.science import sid_layer_analysis as A;"
         "print(A.record_single_sid(Path(%r)))" % (str(REPO), str(sid))],
        capture_output=True, text=True, timeout=1800)
    return ((done.stdout or done.stderr).strip().splitlines() or ["?"])[-1][:120]


def main() -> int:
    log("waiting for the five Stage 2 runs")
    deadline = time.time() + 6 * 3600
    pending = list(ARMS)
    while pending and time.time() < deadline:
        pending = [arm for arm in ARMS if not complete(arm)]
        if pending:
            log("still running: " + ", ".join(pending))
            time.sleep(120)
    if pending:
        log("gave up waiting for: " + ", ".join(pending))
    for arm in ARMS:
        if complete(arm):
            log("%s layers: %s" % (arm, record_layers(arm)))
    log("measuring the structural distortion of all five")
    done = subprocess.run([PY, str(OUT / "measure_structure.py"), *ARMS],
                          capture_output=True, text=True, timeout=3600)
    for line in (done.stdout or "").splitlines():
        if line[:2] in ("F0", "G ", "E ", "A ", "L ") or line.startswith("HYPOTHESIS"):
            log(line)
    if done.returncode != 0:
        log("measurement failed: " + (done.stderr or "")[-300:])
    log("regenerating the report")
    done = subprocess.run([PY, str(OUT / "write_report.py")], capture_output=True, text=True, timeout=1800)
    log(((done.stdout or "").strip().splitlines() or ["?"])[-1][:200])
    log("settle done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
