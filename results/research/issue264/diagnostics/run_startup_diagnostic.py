"""One short isolated startup diagnostic for Issue #264.

It launches the real Stage3 trainer through the optimized parent harness, using
the existing frozen SID, writes only under the non-formal `issue264_diag`
variant, and stops immediately after the first real train record. It never runs
the 150-epoch schedule, never evaluates test, and does not touch #263 artifacts.
"""
from __future__ import annotations
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SID = ROOT / "results/stage2_RQ-VAE/_runs/stage2_ae668bc04dc44d52a8bed5f9408e8e74/TIGER_frozen_topology_gnn_rq/result/item_sids.json"
RESULT = ROOT / "results/stage3_T5Train/issue264_diag"
HARNESS = ROOT / "results/research/issue264/diagnostics/real_harness"
CACHE = "/dev/shm/genrec-stage3-deps-issue264"
VARIANT = "issue264_diag"


def load_server():
    spec = importlib.util.spec_from_file_location("issue264_shared_server", ROOT / "tools/genrec-mcp/server.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["issue264_shared_server"] = module
    spec.loader.exec_module(module)
    return module


def main():
    import shutil
    if RESULT.exists():
        shutil.rmtree(RESULT)
    HARNESS.mkdir(parents=True, exist_ok=True)
    server = load_server()
    server.HARNESS_DIR = HARNESS
    harness = server.write_stage3_harness(SID, VARIANT, False)
    env = os.environ.copy()
    env["GENREC_STAGE3_MODE"] = "ddp"
    env["GENREC_STAGE3_DEP_CACHE"] = CACHE
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    start = time.time()
    child = subprocess.Popen([sys.executable, str(harness)], cwd=str(ROOT / "stage3_T5Train"),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, env=env)
    metric_path = None
    first_train = None
    deadline = time.time() + 600
    try:
        while time.time() < deadline:
            candidates = list((RESULT / "logs").rglob("training_metrics.jsonl"))
            if candidates and metric_path is None:
                metric_path = candidates[0]
            if metric_path is not None:
                for line in metric_path.read_text(errors="ignore").splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("event") == "train":
                        first_train = row
                        break
            if first_train is not None:
                break
            if child.poll() is not None:
                break
            time.sleep(0.25)
    finally:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=30)
    payload = {
        "schema": "genrec.issue264.startup_diagnostic/1",
        "variant": VARIANT,
        "sid": str(SID),
        "start_epoch": start,
        "child_pid": child.pid,
        "metric_path": str(metric_path) if metric_path else None,
        "first_train": first_train,
        "first_train_after_s": (first_train["timestamp"] - start) if first_train else None,
        "child_exit": child.returncode,
        "formal_training":"not run; variant is issue264_diag and process was stopped after first train record",
        "test_evaluated": False,
    }
    (ROOT / "results/research/issue264/DIAGNOSTIC_STARTUP.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if first_train is None:
        raise SystemExit("no first train record; inspect issue264_diag logs")


if __name__ == "__main__":
    main()
