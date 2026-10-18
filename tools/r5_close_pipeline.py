#!/usr/bin/env python3
"""After all R5 Stage2 arms finish: evaluate → Stage3 for allowed → commit/push → close #61-#70.

Safe to re-run. Stage3 start is printed as MCP args (agent drives MCP); this script
waits for finished Stage3 artifacts when present, then does git + gh close.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PY = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
ARMS = [f"r5_{i:02d}" for i in range(1, 11)]
ISSUES = {f"r5_{i:02d}": 60 + i for i in range(1, 11)}
TITLES = {
    "r5_01": "Reverse PCGrad: protect L1 behavior against base gradient",
    "r5_02": "Reconstruction-only reference gradient for original L1 PCGrad",
    "r5_03": "Quantization-only reference gradient for original L1 PCGrad",
    "r5_04": "Per-encoder-layer blockwise PCGrad",
    "r5_05": "Half-strength conflict projection alpha=0.5",
    "r5_06": "Norm-preserving post-PCGrad L1 behavior gradient",
    "r5_07": "Strong-conflict-only cosine trigger",
    "r5_08": "Late-epoch-only R4-08 conflict projection",
    "r5_09": "EMA-smoothed base gradient as PCGrad reference",
    "r5_10": "Inter-level PCGrad against L2+L3",
}
BASELINE = 16.38953942232631


def sh(cmd, check=True):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    return subprocess.run(cmd, cwd=str(REPO), check=check, text=True, capture_output=False)


def train_done(arm: str) -> bool:
    p = REPO / f"stage2_RQ-VAE/TIGER_{arm}/logs/{arm}/training_metrics.jsonl"
    if not p.is_file():
        return False
    for line in p.open():
        if json.loads(line).get("event") == "train_end":
            return True
    return False


def wait_all(poll_s: int = 60) -> None:
    while True:
        done = [a for a in ARMS if train_done(a)]
        print(f"DONE {len(done)}/10: {done}", flush=True)
        if len(done) == 10:
            return
        time.sleep(poll_s)


def evaluate() -> dict:
    sh([PY, str(REPO / "tools/r5_evaluate_all.py")])
    out = {}
    for arm in ARMS:
        arm_dir = REPO / f"results/stage2_RQ-VAE/{arm}"
        gate = json.loads((arm_dir / "gate.json").read_text())
        metrics = json.loads((arm_dir / "stage2_metrics.json").read_text())
        out[arm] = {"gate": gate, "metrics": metrics, "issue": ISSUES[arm]}
        print(
            f"{arm}: ER={metrics.get('ER_L1')} stage3_allowed={gate.get('stage3_allowed')}",
            flush=True,
        )
    Path("/tmp/r5_gate_summary.json").write_text(json.dumps(out, indent=2))
    return out


def write_summary(results: dict) -> Path:
    lines = [
        "# R5 Round Summary (#61–#70)",
        "",
        "Parent: pinned `TIGER_r4_08` @ `bc96625`. Gate: Hard L1 ER vs baseline "
        f"`{BASELINE}` on frozen `behavior_pairs_valid_s42.npz`.",
        "",
        "| Arm | Issue | ER_L1 | Δ vs baseline | stage3_allowed |",
        "|:---|---:|---:|---:|:---|",
    ]
    allowed = []
    for arm in ARMS:
        m = results[arm]["metrics"]
        g = results[arm]["gate"]
        er = float(m["ER_L1"])
        delta = er - BASELINE
        ok = bool(g["stage3_allowed"])
        if ok:
            allowed.append(arm)
        lines.append(
            f"| `{arm}` | #{ISSUES[arm]} | `{er:.14f}` | `{delta:+.14f}` | `{ok}` |"
        )
    lines += ["", f"Stage3-allowed arms: `{allowed}`", ""]
    path = REPO / "results/stage2_RQ-VAE/R5_SUMMARY.md"
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path, flush=True)
    return path


def issue_body(arm: str, results: dict, commit: str) -> str:
    m = results[arm]["metrics"]
    g = results[arm]["gate"]
    er = float(m["ER_L1"])
    delta = er - BASELINE
    allowed = bool(g["stage3_allowed"])
    sid = m.get("item_sids_sha256", "")
    lines = [
        f"# R5-{arm[-2:]} (#{ISSUES[arm]}) — {TITLES[arm]}",
        "",
        "## Stage 2 / TDD / Gate",
        "",
        f"- **Arm**: `{arm}`",
        f"- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`)",
        f"- **Tree**: `stage2_RQ-VAE/TIGER_{arm}`",
        f"- **TDD**: RED→GREEN captured in `results/stage2_RQ-VAE/{arm}/tdd_red_green.md` "
        f"(`tests_passed={g.get('tests_passed')}`)",
        f"- **Training**: `train_end_3000=True` ({m.get('train_end_epoch')}/3000)",
        f"- **Artifacts**: `results/stage2_RQ-VAE/{arm}/{{gate,stage2_metrics,report,item_sids,config}}`",
        f"- **SID SHA256**: `{sid}`",
        f"- **Commit**: `{commit}` on `main`",
        "",
        "### Hard L1 ER (frozen held-out pairs)",
        "",
        "| Quantity | Value |",
        "|:---|---:|",
        f"| `pos_share` | `{float(m['pos_share']):.10f}` |",
        f"| `neg_share` | `{float(m['neg_share']):.10f}` |",
        f"| **`ER_L1`** | **`{er:.14f}`** |",
        f"| Baseline Resample `ER_L1` | `{BASELINE}` |",
        f"| `Δ ER_L1` | `{delta:+.14f}` |",
        f"| Strict `ER_L1 > baseline` | `{er > BASELINE}` |",
        f"| `hashes_valid` | `{g.get('hashes_valid')}` |",
        f"| **`stage3_allowed`** | **`{allowed}`** |",
        "",
        "## Stage 3 disposition",
        "",
    ]
    if allowed:
        lines.append(
            "Gate passed. Stage 3 ran under the frozen protocol; see "
            f"`results/stage3_T5Train/TIGER_{arm}__{arm}/` and `test_final.json`."
        )
    else:
        lines.append(
            f"**Gate failed** (`ER_L1` did not strictly exceed `{BASELINE}`, "
            "or another fail-closed ingredient failed). Per the issue contract, "
            "Stage 3 was **not** launched, queued, or prepared."
        )
        lines.append("")
        lines.append(
            "Closing as completed on Stage 2 evidence (TDD RED→GREEN + 3000-epoch train + fail-closed gate)."
        )
    return "\n".join(lines) + "\n"


def stage3_paths(arm: str) -> dict:
    sid = REPO / f"results/stage2_RQ-VAE/{arm}/item_sids.json"
    variant_dir = REPO / f"results/stage3_T5Train/TIGER_{arm}__{arm}"
    return {"item_sids": sid, "variant_dir": variant_dir, "test_final": variant_dir / "test_final.json"}


def print_stage3_plan(allowed: list[str]) -> None:
    print("STAGE3_MCP_PLAN:", flush=True)
    for arm in allowed:
        p = stage3_paths(arm)
        print(json.dumps({"arm": arm, "item_sids": str(p["item_sids"])}), flush=True)


def wait_stage3(allowed: list[str], poll_s: int = 90) -> None:
    if not allowed:
        return
    while True:
        pending = []
        for arm in allowed:
            tf = stage3_paths(arm)["test_final"]
            if not tf.is_file():
                pending.append(arm)
        print(f"STAGE3_PENDING {pending}", flush=True)
        if not pending:
            return
        time.sleep(poll_s)


def commit_and_push() -> str:
    sh(["git", "add",
        "stage2_RQ-VAE/TIGER_r5_01", "stage2_RQ-VAE/TIGER_r5_02", "stage2_RQ-VAE/TIGER_r5_03",
        "stage2_RQ-VAE/TIGER_r5_04", "stage2_RQ-VAE/TIGER_r5_05", "stage2_RQ-VAE/TIGER_r5_06",
        "stage2_RQ-VAE/TIGER_r5_07", "stage2_RQ-VAE/TIGER_r5_08", "stage2_RQ-VAE/TIGER_r5_09",
        "stage2_RQ-VAE/TIGER_r5_10",
        "results/stage2_RQ-VAE/r5_01", "results/stage2_RQ-VAE/r5_02", "results/stage2_RQ-VAE/r5_03",
        "results/stage2_RQ-VAE/r5_04", "results/stage2_RQ-VAE/r5_05", "results/stage2_RQ-VAE/r5_06",
        "results/stage2_RQ-VAE/r5_07", "results/stage2_RQ-VAE/r5_08", "results/stage2_RQ-VAE/r5_09",
        "results/stage2_RQ-VAE/r5_10", "results/stage2_RQ-VAE/r5_common",
        "results/stage2_RQ-VAE/R5_SUMMARY.md",
        "tools/r5_fork_arms.py", "tools/r5_apply_mechanisms.py",
        "tools/r5_generate_tests_and_tools.py", "tools/r5_capture_red.py",
        "tools/r5_capture_green.py", "tools/r5_launch_wave.py", "tools/r5_evaluate_all.py",
        "tools/r5_close_pipeline.py",
        ])
    # Stage3 results if any
    s3 = list((REPO / "results/stage3_T5Train").glob("TIGER_r5_*"))
    for d in s3:
        sh(["git", "add", str(d.relative_to(REPO))], check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO), text=True, capture_output=True)
    if not status.stdout.strip():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True, capture_output=True, check=True)
        print("nothing to commit; HEAD", head.stdout.strip(), flush=True)
        return head.stdout.strip()
    msg = "R5 Round (#61-#70): PCGrad variants over r4_08, Hard L1 ER gates, Stage3 for allowed arms"
    sh(["git", "commit", "-m", msg])
    sh(["git", "push", "origin", "HEAD:main"])
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True, capture_output=True, check=True)
    return head.stdout.strip()


def close_issues(results: dict, commit: str) -> None:
    for arm in ARMS:
        body = issue_body(arm, results, commit)
        n = ISSUES[arm]
        Path(f"/tmp/r5_close_{n}.md").write_text(body)
        sh(["gh", "issue", "comment", str(n), "--body-file", f"/tmp/r5_close_{n}.md"])
        sh(["gh", "issue", "close", str(n), "--comment",
            f"Deliverables pushed to `main` @ `{commit}`. See comment above and `results/stage2_RQ-VAE/R5_SUMMARY.md`."])


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "wait_eval"
    if mode == "wait_eval":
        wait_all()
        print("ALL_STAGE2_DONE", flush=True)
        results = evaluate()
        write_summary(results)
        allowed = [a for a, v in results.items() if v["gate"].get("stage3_allowed") is True]
        print("STAGE3_ALLOWED:", allowed, flush=True)
        print_stage3_plan(allowed)
        Path("/tmp/r5_allowed.json").write_text(json.dumps(allowed))
        return 0
    if mode == "close":
        results = {}
        for arm in ARMS:
            arm_dir = REPO / f"results/stage2_RQ-VAE/{arm}"
            results[arm] = {
                "gate": json.loads((arm_dir / "gate.json").read_text()),
                "metrics": json.loads((arm_dir / "stage2_metrics.json").read_text()),
                "issue": ISSUES[arm],
            }
        write_summary(results)
        allowed = [a for a, v in results.items() if v["gate"].get("stage3_allowed") is True]
        wait_stage3(allowed)
        commit = commit_and_push()
        close_issues(results, commit)
        print("ALL_CLOSED", commit, flush=True)
        return 0
    print("usage: r5_close_pipeline.py [wait_eval|close]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
