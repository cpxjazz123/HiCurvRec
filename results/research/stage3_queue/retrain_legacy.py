"""Re-train the legacy Stage 2 trees behind the >0.06 Stage 3 runs, all at once.

The Stage 2 code of those arms is gone from the worktree; each tree below still
exists in git history (the last commit that contains it).  Every tree is
restored from that commit and then run as ONE process, so the eleven arms can
share four GPUs instead of queueing for them:

  * one rank per arm, pinned with CUDA_VISIBLE_DEVICES (three arms per card);
  * the per-rank batch is raised x4 (1024 -> 4096 items, pair batch 512 -> 2048)
    so the effective batch stays the frozen 4096 -- the single-card branch of
    rule 19d0b, not a smaller recipe;
  * each arm gets its own torchrun port (50220+i) so eleven DDP launchers never
    fight over 50202;
  * GENREC_STAGE2_LAUNCHED=1 stops the newer trees from forking a second time.

Nothing waits for a free GPU beyond the one Stage 3 already running at launch
time: the driver sleeps until that job exits, then starts all eleven together.
"""
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TREES = [  # (tree, last commit containing it)
    ("TIGER_BEHAVIOR_R48", "ec7971bed4a4"),
    ("TIGER_BEHAVIOR_SAMPLE_RQ-VAE", "3168c6ada1ef"),
    ("TIGER_FORK_RQ-VAE", "d6d5c2d2435d"),
    ("TIGER_HYPERBOLIC_RQ-VAE", "1b6a93aacfbc"),
    ("TIGER_behavior_residual_geometry_transport", "40a81eda282c"),
    ("TIGER_bootstrap_code_stability", "a2fc64a2fe32"),
    ("TIGER_c106_directed_assignment_usage_balance", "0a7da38c3e7c"),
    ("TIGER_c106_directed_codeword_margin_geometry", "0a7da38c3e7c"),
    ("TIGER_c1_behaviour_conditioned_residual_target", "ace0a07eadf7"),
    ("curvature_RQ-VAE", "d261d06c905b"),
    ("curvature_RQ-VAE_iter48", "00a769430b1a"),
]
PY39 = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
TORCHRUN = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/torchrun"
OUT = REPO / "results" / "research" / "stage3_queue"
REPORT = OUT / "retrain_report.json"
GPU_SLOTS = 4
BASE_PORT = 50220
FREE_MEMORY_MIB = 2000


def log(message: str) -> None:
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), message)
    print(line, flush=True)
    with (OUT / "retrain.log").open("a") as handle:
        handle.write(line + "\n")


def restore(tree: str, commit: str) -> Path:
    dest = REPO / "stage2_RQ-VAE" / tree
    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    data = subprocess.run(["git", "archive", commit, "stage2_RQ-VAE/%s" % tree],
                          cwd=REPO, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(REPO)
    return dest


def single_card_batch(dest: Path) -> str:
    """Keep the effective batch at 4096 when the four ranks collapse to one."""
    script = dest / "train_rqvae.py"
    text = script.read_text()
    text = re.sub(r"^BATCH_SIZE_PER_RANK = 1024\b", "BATCH_SIZE_PER_RANK = 4096",
                  text, flags=re.M)
    text = re.sub(r"^PAIR_BATCH_SIZE_PER_RANK = 512\b",
                  "PAIR_BATCH_SIZE_PER_RANK = 2048", text, flags=re.M)
    script.write_text(text)
    return hashlib.sha256(script.read_bytes()).hexdigest()


def gpu_busy() -> bool:
    query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
    used = [int(x) for x in query.stdout.split() if x.strip().isdigit()]
    return bool(used) and max(used) > FREE_MEMORY_MIB


def sids(tree: str) -> list[Path]:
    return sorted((REPO / "results" / "stage2_RQ-VAE" / tree).rglob("item_sids.json"))


def main() -> None:
    report = json.loads(REPORT.read_text()) if REPORT.is_file() else {}
    started = {}
    for index, (tree, commit) in enumerate(TREES):
        if report.get(tree, {}).get("state") == "done":
            log("%s already done" % tree)
            continue
        dest = restore(tree, commit)
        sha = single_card_batch(dest)
        report[tree] = {"commit": commit, "train_sha256": sha,
                        "state": "waiting_gpu", "gpu": index % GPU_SLOTS,
                        "port": BASE_PORT + index}
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        log("%s restored from %s (train sha %s), slot GPU %d port %d"
            % (tree, commit, sha[:12], index % GPU_SLOTS, BASE_PORT + index))

    while gpu_busy():
        time.sleep(30)
    log("GPUs free; starting %d arms together" % sum(
        1 for t, _ in TREES if report.get(t, {}).get("state") != "done"))

    running = {}
    for index, (tree, _) in enumerate(TREES):
        entry = report.get(tree, {})
        if entry.get("state") == "done":
            continue
        dest = REPO / "stage2_RQ-VAE" / tree
        logfile = OUT / ("retrain_%s.log" % tree)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(index % GPU_SLOTS)
        env["GENREC_STAGE2_LAUNCHED"] = "1"
        env["NCCL_IB_DISABLE"] = "1"
        env["OMP_NUM_THREADS"] = "8"
        handle = logfile.open("wb")
        process = subprocess.Popen(
            [TORCHRUN, "--standalone", "--nproc_per_node=1",
             "--master_port=%d" % (BASE_PORT + index), "train_rqvae.py"],
            cwd=dest, stdout=handle, stderr=subprocess.STDOUT, env=env)
        running[tree] = (process, handle, logfile, time.time())
        log("%s started pid=%d" % (tree, process.pid))

    while running:
        time.sleep(30)
        for tree, (process, handle, logfile, began) in list(running.items()):
            if process.poll() is None:
                continue
            handle.close()
            del running[tree]
            produced = sids(tree)
            report[tree].update({
                "returncode": process.returncode,
                "wall_time_s": round(time.time() - began, 1),
                "log": str(logfile.relative_to(REPO)),
                "sids": [str(p.relative_to(REPO)) for p in produced],
                "state": "done" if produced and process.returncode == 0 else "failed",
            })
            REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            log("%s %s rc=%s sids=%d" % (tree, report[tree]["state"],
                                         process.returncode, len(produced)))
    log("all arms finished")


if __name__ == "__main__":
    main()
