#!/usr/bin/env python3
# Issue #210: per-config child runner (called by run_sequential.sh)
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ITEM_EMB_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage1_hyp_v2/item_emb_u32.npy"
STAGE2_MAIN = Path("/fs04/ar57/wenyu/GeneRec/taskA/stage2/taskA_stage2.py")
SNAPSHOT = Path("/tmp/taskA_stage2_issue210_snapshot.py")
LOG_BASE = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue210_equal_codebook")


def main(codebook_tuple, tag, gpu_id, epochs=1000):
    product_dir = LOG_BASE / f"taskA_stage2_{tag}"
    product_dir.mkdir(parents=True, exist_ok=True)
    print(f"[child {tag}] GPU={gpu_id} codebook={codebook_tuple} epochs={epochs}")

    if not SNAPSHOT.exists():
        SNAPSHOT.write_text(STAGE2_MAIN.read_text())
        print(f"[snapshot] created {SNAPSHOT}")

    runner = Path(f"/tmp/taskA_stage2_{tag}_v2.py")
    content = SNAPSHOT.read_text()
    new_list = eval(codebook_tuple)
    old_line = re.search(r"^CODEBOOK_SIZES = .*$", content, re.MULTILINE).group(0)
    new_line = f'CODEBOOK_SIZES = {new_list}  # Issue #210 Equal-Codebook: {tag}'
    content = content.replace(old_line, new_line)
    runner.write_text(content)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3",
        "-u", str(runner),
        "--no_mlr",
        "--item_emb_npy", ITEM_EMB_NPY,
        "--epochs", str(epochs),
        "--batch_size", "1024",
        "--world_size", "1",
        "--product_dir", str(product_dir),
    ]
    log_path = product_dir / "train.log"
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                cwd="/fs04/ar57/wenyu/GeneRec", env=env)
    rc = proc.wait()
    if runner.exists():
        runner.unlink()
    return rc


if __name__ == "__main__":
    codebook_tuple = sys.argv[1]
    tag = sys.argv[2]
    gpu_id = int(sys.argv[3])
    epochs = int(sys.argv[4])
    sys.exit(main(codebook_tuple, tag, gpu_id, epochs))
