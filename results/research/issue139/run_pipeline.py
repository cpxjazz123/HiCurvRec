"""Drive issue #139 from five Stage 2 runs to five Stage 3 runs, unattended.

The owner's goal, in order: complete the Stage 2 of all five mechanisms, record
the structural distortion of the five resulting SIDs, and run the corresponding
Stage 3 for each of them.

This script does not reimplement any of that. It calls the framework's own entry
points -- the same `tool_stage2_start` and `tool_stage3_start` the MCP exposes,
with the same admission checks and the same smoke -- and it waits on the
artifacts those calls produce. Nothing here bypasses a gate: if the admission
refuses an arm, the arm is recorded as refused and the pipeline moves on.

Usage:
    nohup python3 results/research/issue139/run_pipeline.py \
        --stage2-wait E --stage2-start A L --skip-stage3 > pipeline.log 2>&1 &
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue139"
ARMS = ("F0", "G", "L", "E", "A")
PY = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print("[%s] %s" % (stamp, message), flush=True)


def server():
    spec = importlib.util.spec_from_file_location(
        "genrec_server", REPO / "tools" / "genrec-mcp" / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree(arm: str) -> Path:
    return REPO / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm)


def result(arm: str) -> Path:
    return REPO / "results" / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm)


def metrics_path(arm: str) -> Path:
    return tree(arm) / "logs" / "training_metrics.jsonl"


def state_path(arm: str) -> Path:
    return Path.home() / ".cache" / "genrec-mcp" / ("stage2__TIGER_mixed_rq_" + arm + ".json")


def last_run_finished(arm: str) -> bool:
    """Whether the newest run of this arm recorded its final event."""
    path = metrics_path(arm)
    if not path.is_file():
        return False
    events = []
    for line in path.open():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    starts = [index for index, event in enumerate(events) if event.get("event") == "train_start"]
    if not starts:
        return False
    return any(event.get("event") == "train_end" for event in events[starts[-1]:])


def last_epoch(arm: str) -> int:
    path = metrics_path(arm)
    if not path.is_file():
        return 0
    events = []
    for line in path.open():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    starts = [index for index, event in enumerate(events) if event.get("event") == "train_start"]
    if not starts:
        return 0
    epochs = [event for event in events[starts[-1]:] if event.get("event") == "train"]
    return epochs[-1]["epoch"] if epochs else 0


def wait_for_stage2(arm: str, timeout: int = 4 * 3600) -> str:
    deadline = time.time() + timeout
    last_report = 0
    while time.time() < deadline:
        if last_run_finished(arm):
            return "finished"
        state = state_path(arm)
        if state.is_file():
            payload = json.loads(state.read_text())
            if payload.get("status") == "refused":
                return "refused: " + str(payload.get("refused"))[:200]
            alive = payload.get("worker_alive")
            if payload.get("status") == "running" and alive is False and not last_run_finished(arm):
                # the worker is gone and no final event was written
                time.sleep(30)
                if last_run_finished(arm):
                    return "finished"
                return "worker gone without a final event: " + crash_tail(arm)
        epoch = last_epoch(arm)
        if epoch and epoch != last_report and epoch % 500 == 0:
            last_report = epoch
            log("%s epoch %d/3000" % (arm, epoch))
        time.sleep(20)
    return "timeout"


def start_stage2(arm: str) -> str:
    log("dispatching Stage 2 for %s" % arm)
    module = server()
    answer = module.tool_stage2_start(str(tree(arm)), "machine:issue139-" + arm.lower())
    if answer.get("started"):
        return "started"
    # The first version of this line returned "refused: None" and hid the real
    # reason, so the whole answer is logged when a dispatch does not start.
    verdict = answer.get("agent_verification") or {}
    failed = [check for check in (verdict.get("checks") or [])
              if check.get("status") != "PASS"]
    reason = answer.get("refused") or verdict.get("refused") or "no reason in the answer"
    log("%s dispatch did not start: %s | failing checks: %s"
        % (arm, str(reason)[:300], [c.get("name") for c in failed] or "none"))
    return "refused: " + str(reason)[:300]


def crash_tail(arm: str, limit: int = 3) -> str:
    """The last error lines of the arm's launch log, for a run that died."""
    log_path = result(arm) / "logs" / "train_migrated.log"
    if not log_path.is_file():
        return "no launch log"
    lines = [line for line in log_path.read_text(errors="replace").splitlines()
             if any(marker in line for marker in ("Error", "Traceback", "ValueError",
                                                  "RuntimeError", "SIGABRT", "signal"))
             and "UserWarning" not in line]
    return " || ".join(lines[-limit:])[:400] or "no error line in the launch log"


def record_layers(arm: str) -> str:
    sid = result(arm) / "item_sids.json"
    if not sid.is_file():
        return "no SID"
    done = subprocess.run([PY, "-c",
                           "import sys; sys.path.insert(0, %r);"
                           "from pathlib import Path;"
                           "from tools.research.science import sid_layer_analysis as A;"
                           "print(A.record_single_sid(Path(%r)))" % (str(REPO), str(sid))],
                          capture_output=True, text=True, timeout=1800)
    return (done.stdout or done.stderr).strip().splitlines()[-1][:200]


def measure(arm: str) -> str:
    done = subprocess.run([PY, str(OUT / "measure_structure.py"), arm],
                          capture_output=True, text=True, timeout=3600)
    lines = [line for line in (done.stdout or "").splitlines()
             if line.startswith(("F0 ", "G  ", "L  ", "E  ", "A  ", "HYPOTHESIS"))]
    return " | ".join(lines)[:300] or (done.stderr or "").strip()[-200:]


def stage3_dir(arm: str) -> Path:
    return REPO / "results" / "stage3_T5Train" / ("TIGER_mixed_rq_" + arm)


def wait_for_stage3(arm: str, timeout: int = 6 * 3600) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not any(("stage3" in str(path) and arm in str(path))
                   for path in []):
            pass
        found = list(stage3_dir(arm).glob("logs/*/*/test_final.json"))
        if found:
            payload = json.loads(found[-1].read_text())
            return "finished R@10=%.6f" % payload.get("test_recall@10", -1)
        time.sleep(60)
    return "timeout"


def start_stage3(arm: str) -> str:
    sid = result(arm) / "item_sids.json"
    if not sid.is_file():
        return "no SID"
    log("dispatching Stage 3 for %s" % arm)
    module = server()
    answer = module.tool_stage3_start(str(sid))
    if answer.get("started"):
        return "started"
    return "refused: " + str(answer.get("refused") or answer.get("error"))[:200]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-wait", nargs="*", default=[],
                        help="arms whose Stage 2 is already running")
    parser.add_argument("--stage2-start", nargs="*", default=[],
                        help="arms whose Stage 2 this script has to dispatch")
    parser.add_argument("--stage3", nargs="*", default=list(ARMS),
                        help="arms to carry into Stage 3")
    parser.add_argument("--skip-stage3", action="store_true")
    args = parser.parse_args()

    report = {"stage2": {}, "distortion": {}, "stage3": {}}
    for arm in list(args.stage2_wait) + list(args.stage2_start):
        outcome = wait_for_stage2(arm) if arm in args.stage2_wait else None
        if outcome is None:
            outcome = start_stage2(arm)
            if outcome == "started":
                outcome = wait_for_stage2(arm)
        log("%s Stage 2: %s" % (arm, outcome))
        report["stage2"][arm] = outcome
        if outcome == "finished":
            log("%s layered record: %s" % (arm, record_layers(arm)))
            log("%s distortion: %s" % (arm, measure(arm)))
            report["distortion"][arm] = "recorded"

    if not args.skip_stage3:
        for arm in args.stage3:
            if report["stage2"].get(arm) != "finished":
                log("%s skipped in Stage 3: Stage 2 did not finish" % arm)
                report["stage3"][arm] = "skipped"
                continue
            outcome = start_stage3(arm)
            if outcome == "started":
                outcome = wait_for_stage3(arm)
            log("%s Stage 3: %s" % (arm, outcome))
            report["stage3"][arm] = outcome

    (OUT / "pipeline_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log("pipeline report written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
