"""Run the five Stage 3 jobs, one at a time, each in its own process.

The earlier all-in-one driver wedged inside a call it made in its own process.
This runner keeps every framework call in a separate, time-limited subprocess, so
a stuck call is a recorded failure rather than a dead pipeline, and it waits for
the last Stage 2 arm before it touches the GPUs.

Since 2026-11-09 the framework's `tool_stage3_start` performs no admission
checks (owner policy): it starts any SID under the Stage 2 result root, records
the layer-record outcome as a diagnostic, and the runner records whatever it
answers. The frozen Stage 3 protocol itself is still checked.

Usage:
    nohup python3 results/research/issue139/run_stage3.py > stage3.log 2>&1 &
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARMS = ("F0", "G", "E", "A", "L")
PY = sys.executable

CALL = """
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("genrec_server",
    r"%(server)s")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
answer = module.tool_stage3_start(%(sid)r)
print("ANSWER " + json.dumps({k: answer.get(k) for k in
      ("started", "variant", "refused", "error", "pid", "run_log")}, default=str))
"""


def log(message: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def sid_of(arm: str) -> Path:
    return REPO / "results" / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm) / "item_sids.json"


def stage2_done(arm: str) -> bool:
    """Whether this arm has an identifier and no Stage 2 trainer is running.

    Reading the metrics stream does not answer it: a cancelled run truncates the
    stream, so F0's finished run left no final event behind, and asking for one
    would keep the whole Stage 3 queue waiting forever. What the queue needs is
    simpler: the identifier exists and nobody is training that tree.
    """
    sid = sid_of(arm)
    if not sid.is_file():
        return False
    done = subprocess.run(["pgrep", "-f", "TIGER_mixed_rq_%s/train_rqvae.py" % arm],
                          capture_output=True, text=True)
    return not bool((done.stdout or "").strip())


def stage3_result(arm: str) -> dict | None:
    folder = REPO / "results" / "stage3_T5Train" / ("TIGER_mixed_rq_" + arm)
    found = sorted(folder.glob("logs/*/*/test_final.json")) if folder.is_dir() else []
    return json.loads(found[-1].read_text()) if found else None


def start(arm: str) -> dict:
    script = CALL % {"server": str(REPO / "tools" / "genrec-mcp" / "server.py"),
                     "sid": str(sid_of(arm))}
    done = subprocess.run([PY, "-c", script], capture_output=True, text=True, timeout=1800)
    for line in (done.stdout or "").splitlines():
        if line.startswith("ANSWER "):
            return json.loads(line[len("ANSWER "):])
    return {"error": ((done.stdout or "") + (done.stderr or ""))[-300:]}


def main() -> int:
    report = {}
    # The GPUs are shared: no Stage 3 may run while a Stage 2 arm is still training.
    for arm in ARMS:
        while not stage2_done(arm):
            log("waiting for the Stage 2 of %s" % arm)
            time.sleep(120)
    log("all five Stage 2 arms are finished; starting Stage 3")
    for arm in ARMS:
        existing = stage3_result(arm)
        if existing is not None:
            log("%s already has a Stage 3 result: R@10=%.6f"
                % (arm, existing.get("test_recall@10", -1)))
            report[arm] = {"state": "already", "recall@10": existing.get("test_recall@10")}
            continue
        log("dispatching Stage 3 for %s" % arm)
        answer = start(arm)
        if not answer.get("started"):
            log("%s refused: %s" % (arm, str(answer)[:300]))
            report[arm] = {"state": "refused", "answer": answer}
            continue
        deadline = time.time() + 6 * 3600
        while time.time() < deadline:
            result = stage3_result(arm)
            if result is not None:
                log("%s Stage 3 done: R@10=%.6f" % (arm, result.get("test_recall@10", -1)))
                report[arm] = {"state": "done", "recall@10": result.get("test_recall@10"),
                               "recall_block": result}
                # The rule: every finished Stage 3 is archived to R2 with the Stage 2
                # code, the SID, the checkpoint and the complete recall record.
                archive = subprocess.run(
                    [PY, str(OUT / "upload_stage3.py"), arm],
                    capture_output=True, text=True, timeout=3600)
                log("%s archive: %s" % (arm, ((archive.stdout or archive.stderr).strip()
                                              .splitlines() or ["?"])[-1][:200]))
                break
            time.sleep(120)
        else:
            log("%s Stage 3 timed out" % arm)
            report[arm] = {"state": "timeout"}
    (OUT / "stage3_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log("stage 3 report written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
