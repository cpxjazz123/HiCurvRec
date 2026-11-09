"""Re-train the legacy Stage 2 trees behind the >0.06 Stage 3 runs, so each can be
archived with its four pieces.

The Stage 2 code of those arms is gone from the worktree; the trees below still
exist in git history (last commit that contains them).  Each tree is restored
from that commit, run with its own frozen constants (3000 epochs, seed 42,
4-card DDP on port 50202, global batch 4096), and its SID/checkpoint stay in
results/stage2_RQ-VAE/<tree>/ as the tree writes them.

The numbers will not match the historical 0.06 runs: those used per-variant
copies whose constants were never committed, and several trees encode their
default variant through EXPERIMENT_ARM.
"""
import hashlib
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TREES = {
    "TIGER_BEHAVIOR_R48": "ec7971bed4a4",
    "TIGER_BEHAVIOR_SAMPLE_RQ-VAE": "3168c6ada1ef",
    "TIGER_FORK_RQ-VAE": "d6d5c2d2435d",
    "TIGER_HYPERBOLIC_RQ-VAE": "1b6a93aacfbc",
    "TIGER_behavior_residual_geometry_transport": "40a81eda282c",
    "TIGER_bootstrap_code_stability": "a2fc64a2fe32",
    "TIGER_c106_directed_assignment_usage_balance": "0a7da38c3e7c",
    "TIGER_c106_directed_codeword_margin_geometry": "0a7da38c3e7c",
    "TIGER_c1_behaviour_conditioned_residual_target": "ace0a07eadf7",
    "curvature_RQ-VAE": "d261d06c905b",
    "curvature_RQ-VAE_iter48": "00a769430b1a",
}
PY39 = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
OUT = REPO / "results" / "research" / "stage3_queue"
REPORT = OUT / "retrain_report.json"
FREE_MEMORY_MIB = 2000


def log(message: str) -> None:
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), message)
    print(line, flush=True)
    with (OUT / "retrain.log").open("a") as handle:
        handle.write(line + "\n")


def restore(tree: str, commit: str) -> str:
    """Fresh copy of the committed tree; returns the train_rqvae.py sha256."""
    dest = REPO / "stage2_RQ-VAE" / tree
    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    data = subprocess.run(["git", "archive", commit, "stage2_RQ-VAE/%s" % tree],
                          cwd=REPO, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(REPO)
    script = dest / "train_rqvae.py"
    return hashlib.sha256(script.read_bytes()).hexdigest() if script.is_file() else ""


def gpu_busy() -> bool:
    query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
    used = [int(x) for x in query.stdout.split() if x.strip().isdigit()]
    return bool(used) and max(used) > FREE_MEMORY_MIB


def wait_for_gpu() -> None:
    while gpu_busy():
        time.sleep(30)


def sids(tree: str) -> list[Path]:
    return sorted((REPO / "results" / "stage2_RQ-VAE" / tree).rglob("item_sids.json"))


def main() -> None:
    report = json.loads(REPORT.read_text()) if REPORT.is_file() else {}
    for tree, commit in TREES.items():
        if report.get(tree, {}).get("state") == "done":
            log("%s already done" % tree)
            continue
        sha = restore(tree, commit)
        log("%s restored from %s (train sha %s)" % (tree, commit, sha[:12]))
        wait_for_gpu()
        logfile = OUT / ("retrain_%s.log" % tree)
        started = time.time()
        with logfile.open("wb") as handle:
            process = subprocess.run([PY39, "train_rqvae.py"],
                                     cwd=REPO / "stage2_RQ-VAE" / tree,
                                     stdout=handle, stderr=subprocess.STDOUT)
        produced = sids(tree)
        report[tree] = {
            "commit": commit,
            "train_sha256": sha,
            "returncode": process.returncode,
            "wall_time_s": round(time.time() - started, 1),
            "log": str(logfile.relative_to(REPO)),
            "sids": [str(p.relative_to(REPO)) for p in produced],
            "state": "done" if produced and process.returncode == 0 else "failed",
        }
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        log("%s %s rc=%s sids=%d" % (tree, report[tree]["state"],
                                     process.returncode, len(produced)))


if __name__ == "__main__":
    main()
