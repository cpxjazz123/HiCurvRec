"""L2 codebook capacity ablation on the production RQ-VAE.

L1 and L3 keep 256 codes and the assignment rule keeps the frozen bucket
Sinkhorn; only the second level's codebook changes:

    256  baseline
    128  light compression
     64  forces sharing: 256*64 = 16384 prefixes < 24587 items
     32  stronger compression: 8192 prefixes, ~3 items each

No cone supervision and no prefix loss, so capacity is the only variable.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/l2_capacity_arms"
SEED = 42
ARMS = [
    ("L2_256_baseline", 256),
    ("L2_128", 128),
    ("L2_64", 64),
    ("L2_32", 32),
]


def _arm_dir(name: str) -> Path:
    return RESULTS / f"{name}_s{SEED}"


def _is_complete(name: str) -> bool:
    directory = _arm_dir(name)
    return (directory / "item_sids.json").is_file() and (
        directory / "logs/training_metrics.jsonl"
    ).is_file()


def _load_trainer(gpu: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    sys.path.insert(0, str(PKG))
    import curvature_config as experiment

    spec = importlib.util.spec_from_file_location(
        f"generec_train_rqvae_cap_gpu{gpu}", PKG / "train_rqvae.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return experiment, trainer


def _run_arm(job: tuple[int, str, int]) -> dict:
    gpu, name, l2 = job
    arm_dir = _arm_dir(name)
    arm_dir.mkdir(parents=True, exist_ok=True)
    experiment, trainer = _load_trainer(gpu)
    experiment.MECHANISM_NAME = f"l2_capacity_{name}"
    experiment.STAGE2_RESULT_DIR = arm_dir
    experiment.RQVAE_OUT_DIR = arm_dir / "out/rqvae/instruments"
    experiment.RQVAE_CKPT_PATH = experiment.RQVAE_OUT_DIR / "rqvae_best.pth"
    experiment.RAW_SIDS_NPY = experiment.RQVAE_OUT_DIR / "sids_raw.npy"
    experiment.SIDS_NPY = arm_dir / "dataset/Instruments/sids_for_hgrec.npy"
    experiment.ITEM_SIDS_JSON = arm_dir / "item_sids.json"
    experiment.STAGE2_LOG_DIR = arm_dir / "logs"
    experiment.CATEGORY_CONE_ENABLED = False

    import torch as _torch

    _torch.set_num_threads(2)
    trainer.CODEBOOK_SIZE = (256, l2, 256)
    trainer.BATCH_SIZE_PER_RANK = 1024
    trainer.MAX_GLOBAL_STEPS = 18_000
    trainer.EVAL_INTERVAL_STEPS = 9_000
    trainer.CATEGORY_CONE_ENABLED = False
    trainer.BEHAVIOUR_LOSS_ENABLED = False
    trainer.LOG_DIR = arm_dir / "logs"
    trainer.METRICS_PATH = trainer.LOG_DIR / "training_metrics.jsonl"
    trainer.SNAPSHOT_STEPS = {trainer.MAX_GLOBAL_STEPS: "arm"}
    started = time.time()
    trainer.main()
    return {
        "arm": name,
        "l2_codes": l2,
        "codebook_size": [256, l2, 256],
        "gpu": gpu,
        "seconds": round(time.time() - started, 1),
    }


def _worker(gpu: int, queue: mp.Queue, results: mp.Queue) -> None:
    while True:
        try:
            name, size = queue.get_nowait()
        except Exception:
            return
        try:
            results.put(_run_arm((gpu, name, size)))
        except Exception as error:
            results.put({"arm": name, "gpu": gpu, "error": repr(error)})
            raise


def main() -> None:
    import torch

    gpu_count = torch.cuda.device_count()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pending = [(name, size) for name, size in ARMS if not _is_complete(name)]
    print(f"[cap] {len(pending)} arms over {gpu_count} gpus: {[n for n,_ in pending]}", flush=True)
    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    results: mp.Queue = context.Queue()
    for job in pending:
        queue.put(job)
    workers = [context.Process(target=_worker, args=(g, queue, results)) for g in range(gpu_count)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if any(worker.exitcode for worker in workers):
        raise RuntimeError(f"workers failed: {[w.exitcode for w in workers]}")
    finished = []
    while not results.empty():
        finished.append(results.get())
    for outcome in finished:
        print(f"[cap] finished {outcome}", flush=True)
    with (RESULTS / "arms.json").open("w") as handle:
        json.dump({"arms": finished, "seed": SEED}, handle, indent=2, sort_keys=True)
    print("[cap] complete", flush=True)


if __name__ == "__main__":
    main()
