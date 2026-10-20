"""Track F harness v2 (issue #88): run the REAL Stage2 trainer, not a replay of it.

The #87 pilot was invalid because it replayed frozen one-step bundles: one step-32 bundle was
carried through every "active" step, so no step ever saw a real behaviour-CE minibatch, the
source-conditioned mask had no training graph to exclude against, and the "hard" metric was an
argmax over a soft posterior built on raw item embeddings.

This harness runs the HOST trainer itself, with the auxiliary loss composed inside the host's own
loss graph by ``tools/r7_f_host_hook.py``. The host's own DataLoaders, DistributedSampler,
behaviour CE, false-negative mask, R6-04 soft surrogate or R4-08 PCGrad, AdamW and scheduler all
run unmodified. The only difference between the arms is ``R7_F_LAMBDA``.

  F0  R6-04  lambda=0.00   the zero-lambda no-op control
  F1  R6-04  lambda=0.10   the treatment
  F2  R4-08  lambda=0.00   the zero-lambda no-op control
  F3  R4-08  lambda=0.10   the treatment

The zero-lambda arm is the issue's exact no-op control: the hook executes, the separation term is
computed and added as zero, and the resulting loss, gradients, clipping, optimizer moments, LR,
parameters and hard SID must match the untouched host.

Every arm is audited BEFORE any effectiveness number is computed: the step count, the applied
weight schedule, the per-step hashes, the LR progression and the snapshot inventory.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TOOLS = REPO_ROOT / "tools"
OUTPUT = REPO_ROOT / "results/stage2_RQ-VAE/r7_f_harness_v2"
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

ARMS = {
    "F0": {"host": "r6_04", "lambda": 0.0},
    "F1": {"host": "r6_04", "lambda": 0.10},
    "F2": {"host": "r4_08", "lambda": 0.0},
    "F3": {"host": "r4_08", "lambda": 0.10},
}
BASELINE_STEPS = 32
ACTIVE_STEPS = 128
TOTAL_STEPS = BASELINE_STEPS + ACTIVE_STEPS
SNAPSHOT_STEPS = (0, 8, 16, 32, 64, 96, 128, 160)
TAU_FLOOR = 1e-3


class InvalidProtocol(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


def host_tree(host: str) -> Path:
    """An ISOLATED copy of the host tree, so the preflight never writes into a historical arm.

    The trainer derives its output directory from EXPERIMENT_ARM, so running the real trainer in
    the original tree would overwrite that arm's SID export and checkpoint. It did exactly that
    once during this issue; this function makes the mistake impossible to repeat.
    """
    source = REPO_ROOT / f"stage2_RQ-VAE/TIGER_{host}"
    scratch = REPO_ROOT / f"stage2_RQ-VAE/_r7f_scratch/TIGER_{host}_{host}"
    if scratch.is_dir():
        return scratch
    import shutil
    scratch.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, scratch,
                    ignore=shutil.ignore_patterns("logs", "__pycache__", "*.pth", "*.npy",
                                                  "item_sids.json"))
    trainer = scratch / "train_rqvae.py"
    text = trainer.read_text()
    # give the scratch tree its own arm name so its outputs land in their own directory
    import re
    text = re.sub(r'EXPERIMENT_ARM = "[^"]+"',
                  f'EXPERIMENT_ARM = "r7f_scratch_{host}"', text, count=1)
    trainer.write_text(text)
    return scratch


def run_arm(arm: str, host: str, lam: float, *, seed: int = 42) -> dict[str, Any]:
    """Run the real host trainer with the hook and collect its per-step provenance."""
    tree = host_tree(host)
    arm_dir = OUTPUT / arm / f"seed{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    step_log = arm_dir / "steps.jsonl"
    snapshot_dir = arm_dir / "snapshots"
    if step_log.exists():
        step_log.unlink()
    env = dict(os.environ)
    env.update({
        # The pilot is single-process on ONE device, and says so; the issue allows a single-GPU
        # pilot provided the scope is labelled honestly. GENREC_SINGLE_GPU is the host's own
        # switch, so the production 4-GPU DDP path stays untouched.
        "GENREC_SINGLE_GPU": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "R7_F_ENABLED": "1",
        "R7_F_LAMBDA": str(lam),
        "R7_F_STEP_LOG": str(step_log),
        "R7_F_SNAPSHOT_DIR": str(snapshot_dir),
        "R7_F_SNAPSHOT_STEPS": ",".join(str(v) for v in SNAPSHOT_STEPS),
        "R7_F_CANDIDATES": "128",
        "R7_F_STEP_SCHEDULE": "1",
        "R7_F_MAX_STEPS": str(TOTAL_STEPS),
        "R7_F_BASELINE_STEPS": str(BASELINE_STEPS),
        "R7_F_ACTIVE_WEIGHT": "0.20",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{TOOLS}:{tree}:{os.environ.get('PYTHONPATH', '')}",
    })
    # Run the host trainer IN-PROCESS. The host's own __main__ forks a 4-GPU torchrun, which is
    # not what this pilot measures; calling main() directly keeps the native DataLoaders, the
    # native loss assembly, the native optimizer and the native scheduler, on one device.
    os.environ.update(env)
    command = [PYTHON, "-c",
               "import sys; sys.path.insert(0, %r); "
               "import train_rqvae as T; T.main()" % str(tree)]
    started = time.time()
    completed = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True)
    elapsed = round(time.time() - started, 2)
    stdout_path = arm_dir / "stdout.log"
    stderr_path = arm_dir / "stderr.log"
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    record = {
        "arm": arm, "host": host, "lambda": lam, "seed": seed,
        "tree": str(tree),
        "trainer_sha256": hashlib.sha256((tree / "train_rqvae.py").read_bytes()).hexdigest(),
        "hook_sha256": hashlib.sha256((TOOLS / "r7_f_host_hook.py").read_bytes()).hexdigest(),
        "command": " ".join(command), "exit_code": completed.returncode,
        "elapsed_s": elapsed,
        "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
        "step_log": str(step_log),
    }
    if completed.returncode != 0:
        record["status"] = "invalid_protocol"
        record["reason"] = f"the host trainer exited with code {completed.returncode}"
        (arm_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
        return record
    audit = audit_steps(arm_dir, step_log, snapshot_dir, lam=lam)
    record["audit"] = audit
    record["status"] = ("validated_execution" if audit["execution_ok"] else "invalid_protocol")
    if not audit["execution_ok"]:
        record["reason"] = "; ".join(audit["problems"])
    (arm_dir / "run.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def audit_steps(arm_dir: Path, step_log: Path, snapshot_dir: Path, *,
                 lam: float) -> dict[str, Any]:
    """Check the execution contract BEFORE any effectiveness metric is computed."""
    problems: list[str] = []
    if not step_log.is_file():
        return {"execution_ok": False, "problems": ["the per-step provenance log was never written"],
                "steps": 0, "active_steps": 0}
    records = [json.loads(line) for line in step_log.read_text().splitlines() if line.strip()]
    if len(records) != TOTAL_STEPS:
        problems.append(f"{len(records)} steps were logged, the protocol requires EXACTLY "
                        f"{TOTAL_STEPS} COMPLETED optimizer steps; extra inactive updates are a "
                        f"step-count defect, not a rounding artefact")
    active = [r for r in records if r.get("applied_weight", 0.0) > 0.0]
    if len(active) != ACTIVE_STEPS:
        problems.append(f"{len(active)} steps carried a positive behaviour weight, the protocol "
                        f"requires exactly {ACTIVE_STEPS}")
    weights = sorted({round(float(r.get("applied_weight", 0.0)), 6) for r in records})
    if not set(weights) <= {0.0, 0.20}:
        problems.append(f"the applied behaviour weight took values {weights}; the protocol allows "
                        f"only 0.0 and 0.20")
    for record in records:
        if record.get("applied_weight", 0.0) > 0.0:
            if not record.get("pair_item_ids_sha256"):
                problems.append(f"step {record.get('step')} was active but recorded no pair item ids")
                break
            # The zero-lambda control has no valid pairs BY DESIGN: the term is not evaluated.
            # Only the treatment arm is required to produce a nonempty valid set.
            if lam > 0.0 and not record.get("valid_negative_count"):
                problems.append(f"step {record.get('step')} was active but had no valid unrelated "
                                f"pairs, so the separation term had nothing to minimise")
                break
    steps_seen = [record.get("step") for record in records]
    if steps_seen != list(range(1, len(records) + 1)):
        problems.append("the step counter is not the global optimizer step 1..N (it resets or "
                        "repeats), so snapshots and the weight schedule would be misaligned")
    # the loader stream must actually progress
    hashes = [r.get("item_batch_sha256") for r in records]
    if len(set(hashes)) < max(2, len(hashes) // 4):
        problems.append("the item-batch hashes do not progress: the loader stream is not advancing")
    lrs = [r.get("lr") for r in records if r.get("lr") is not None]
    if lrs and len(set(round(v, 12) for v in lrs)) < 2:
        problems.append("the learning rate never changed across the run, so the scheduler did not "
                        "advance")
    snapshots = sorted(p.name for p in snapshot_dir.glob("step_*")) if snapshot_dir.is_dir() else []
    return {
        "execution_ok": not problems,
        "problems": problems,
        "steps": len(records),
        "active_steps": len(active),
        "applied_weight_values": weights,
        "distinct_item_batch_hashes": len(set(hashes)),
        "distinct_pair_item_id_hashes": len({r.get("pair_item_ids_sha256") for r in records
                                             if r.get("pair_item_ids_sha256")}),
        "lr_first": lrs[0] if lrs else None,
        "lr_last": lrs[-1] if lrs else None,
        "snapshots": snapshots,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    arms = list(ARMS)
    if len(sys.argv) > 1:
        arms = [name for name in sys.argv[1:] if name in ARMS] or arms
    records = []
    for arm in arms:
        host, lam = ARMS[arm]["host"], ARMS[arm]["lambda"]
        print(f"=== {arm} ({host}, lambda={lam}) ===", flush=True)
        record = run_arm(arm, host, lam)
        records.append(record)
        print(f"{arm}: {record['status']} exit={record['exit_code']} "
              f"steps={record.get('audit', {}).get('steps')}", flush=True)
        for problem in record.get("audit", {}).get("problems", []):
            print(f"    PROBLEM {problem}", flush=True)
    (OUTPUT / "runs.json").write_text(json.dumps(records, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
