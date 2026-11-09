"""Run the pending Stage 3 queue, one arm at a time, and archive each to R2.

Since 2026-11-09 Stage 3 has no admission checks, so the queue is simply every
Stage 2 result with an item_sids.json, a checkpoint and a source tree under
results/stage2_RQ-VAE/ that has no test_final.json yet, ordered by the SID's
completion time (newest first). The order is frozen in
results/research/stage3_queue.json.

Each framework call runs in its own subprocess through the server's own
tool_stage3_start (the same function the MCP exposes); a live Stage 3 is waited
out, never duplicated. After the run's test_final.json appears, upload_stage3.py
stages the four archive entries and uploads them.

    setsid nohup python3 results/research/stage3_queue/run_queue.py \
        > results/research/stage3_queue/run.log 2>&1 &

The poll interval is 55 s: under the repository's 60 s supervision cap.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
QUEUE = REPO / "results" / "research" / "stage3_queue.json"
RESULT2 = REPO / "results" / "stage2_RQ-VAE"
RESULT3 = REPO / "results" / "stage3_T5Train"
SERVER = REPO / "tools" / "genrec-mcp" / "server.py"
UPLOAD = REPO / "results" / "research" / "issue139" / "upload_stage3.py"
PY = sys.executable
POLL_SECONDS = 55
DEADLINE_SECONDS = 6 * 3600

CALL = """
import importlib.util, json
spec = importlib.util.spec_from_file_location("genrec_server", r"%(server)s")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
answer = module.tool_stage3_start(%(sid)r)
print("ANSWER " + json.dumps({k: answer.get(k) for k in
      ("started", "variant", "refused", "error", "pid", "run_log")}, default=str))
"""


def log(message: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def sid_of(arm: str) -> Path:
    return RESULT2 / arm / "item_sids.json"


def stage3_result(arm: str) -> dict | None:
    """Nested arms (``tree/variant``) get a ``__`` folder from the trainer."""
    for name in (arm, arm.replace("/", "__")):
        folder = RESULT3 / name
        found = sorted(folder.glob("logs/*/*/test_final.json")) if folder.is_dir() else []
        if found:
            newest = found[-1]
            sid = RESULT2 / arm / "item_sids.json"
            if sid.is_file() and newest.stat().st_mtime < sid.stat().st_mtime:
                continue  # that result predates the retrained SID: not a pairing
            return json.loads(newest.read_text())
    return None


def stage3_live() -> bool:
    """A launcher may still be importing with no GPU memory yet, so ask both."""
    import subprocess as _sp

    if _sp.run(["pgrep", "-f", "run_stage3_param"], capture_output=True).returncode == 0:
        return True
    return gpu_busy()


def gpu_busy() -> bool:
    """Any sizeable GPU allocation means a Stage 3 is live, whatever its name."""
    done = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True)
    used = [int(x) for x in (done.stdout or "").split() if x.strip().isdigit()]
    return bool(used) and max(used) > 5_000


def start(arm: str) -> dict:
    script = CALL % {"server": str(SERVER), "sid": str(sid_of(arm))}
    done = subprocess.run([PY, "-c", script], capture_output=True, text=True, timeout=1800)
    for line in (done.stdout or "").splitlines():
        if line.startswith("ANSWER "):
            return json.loads(line[len("ANSWER "):])
    return {"error": ((done.stdout or "") + (done.stderr or ""))[-300:]}


def upload(arm: str) -> str:
    done = subprocess.run([PY, str(UPLOAD), arm], capture_output=True, text=True,
                          timeout=3600)
    tail = ((done.stdout or "") + (done.stderr or "")).strip().splitlines()
    return tail[-1][:220] if tail else "no output"


def wait_for_result(arm: str, deadline: float) -> dict | None:
    while time.time() < deadline:
        result = stage3_result(arm)
        if result is not None:
            return result
        time.sleep(POLL_SECONDS)
    return None


def run_arm(arm: str, report: dict) -> None:
    existing = stage3_result(arm)
    if existing is not None:
        log("%s already has a Stage 3 result: R@10=%s" % (arm, existing.get("test_recall@10")))
        report[arm] = {"state": "already", "recall@10": existing.get("test_recall@10")}
        return
    if not sid_of(arm).is_file():
        log("%s has no item_sids.json" % arm)
        report[arm] = {"state": "no_sid"}
        return
    started = None
    deadline = time.time() + DEADLINE_SECONDS
    while time.time() < deadline:
        result = stage3_result(arm)
        if result is not None:
            break
        if stage3_live() and stage3_result(arm) is None:
            time.sleep(POLL_SECONDS)
            continue
        answer = start(arm)
        if answer.get("started"):
            started = answer
            log("%s dispatched pid=%s mode=%s" % (arm, answer.get("pid"), answer.get("mode")))
            break
        refused = str(answer.get("refused") or answer.get("error") or answer)
        if "already live" in refused:
            time.sleep(POLL_SECONDS)
            continue
        log("%s refused: %s" % (arm, refused[:300]))
        report[arm] = {"state": "refused", "answer": answer}
        return
    result = wait_for_result(arm, deadline)
    if result is None:
        log("%s Stage 3 timed out" % arm)
        report[arm] = {"state": "timeout", "launch": started}
        return
    recall = result.get("test_recall@10")
    log("%s done: R@10=%s" % (arm, recall))
    archive = upload(arm)
    log("%s archive: %s" % (arm, archive))
    report[arm] = {"state": "done", "recall@10": recall, "recall": result,
                   "archive": archive, "launch": started}


def main() -> int:
    queue = json.loads(QUEUE.read_text())["queue"]
    report_path = OUT / "queue_report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    log("queue: %d arms, newest first" % len(queue))
    for entry in queue:
        arm = entry["arm"]
        if report.get(arm, {}).get("state") == "done":
            continue
        run_arm(arm, report)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log("queue finished; report at %s" % report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
